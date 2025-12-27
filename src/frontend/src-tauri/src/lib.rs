use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, State};

struct OmegaState {
  child: Mutex<Option<Child>>,
}

fn find_repo_root() -> Option<PathBuf> {
  let mut candidates: Vec<PathBuf> = Vec::new();
  if let Ok(cwd) = std::env::current_dir() {
    candidates.push(cwd);
  }
  if let Ok(exe) = std::env::current_exe() {
    if let Some(parent) = exe.parent() {
      candidates.push(parent.to_path_buf());
    }
  }

  for base in candidates {
    let mut dir = Some(base.as_path());
    for _ in 0..10 {
      if let Some(d) = dir {
        let cand = d.to_path_buf();
        if cand.join("scripts").join("start_omega.py").exists() && cand.join("scripts").join("stop.py").exists() {
          return Some(cand);
        }
        dir = d.parent();
      }
    }
  }
  None
}

fn python_bin() -> String {
  std::env::var("METABONK_TAURI_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

fn run_stop_all(repo_root: &Path) {
  let py = python_bin();
  let _ = Command::new(py)
    .arg("scripts/stop.py")
    .arg("--all")
    .current_dir(repo_root)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status();
}

fn child_is_running(child: &mut Child) -> bool {
  match child.try_wait() {
    Ok(Some(_)) => false,
    Ok(None) => true,
    Err(_) => false,
  }
}

fn drain_stream(app: AppHandle, stream: impl std::io::Read + Send + 'static, event: &'static str) {
  std::thread::spawn(move || {
    let reader = BufReader::new(stream);
    for line in reader.lines() {
      match line {
        Ok(l) => {
          let _ = app.emit(event, l);
        }
        Err(_) => break,
      }
    }
  });
}

fn stop_omega_inner(app: Option<&AppHandle>, state: &OmegaState) {
  let repo_root = find_repo_root();
  let mut guard = state.child.lock().unwrap();
  if let Some(mut child) = guard.take() {
    let _ = child.kill();
    let _ = child.wait();
  }
  if let Some(root) = repo_root {
    // Ensure any stragglers (Proton/Wine helpers, ffmpeg, etc.) are killed to avoid VRAM leaks.
    run_stop_all(&root);
  }
  if let Some(app) = app {
    let _ = app.emit("omega-exit", true);
  }
}

#[tauri::command]
fn omega_running(state: State<'_, OmegaState>) -> bool {
  let mut guard = state.child.lock().unwrap();
  match guard.as_mut() {
    None => false,
    Some(child) => {
      if child_is_running(child) {
        true
      } else {
        *guard = None;
        false
      }
    }
  }
}

#[tauri::command]
fn start_omega(app: AppHandle, state: State<'_, OmegaState>, mode: String, workers: u32) -> Result<(), String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/start_omega.py)".to_string())?;

  // Stop any previously-running job before starting a new one. This prevents VRAM leaks and stuck input backends.
  run_stop_all(&repo_root);

  let mut guard = state.child.lock().unwrap();
  if let Some(child) = guard.as_mut() {
    if child_is_running(child) {
      return Err("omega is already running".to_string());
    }
    *guard = None;
  }

  let py = python_bin();
  let mut cmd = Command::new(py);
  cmd.arg("-u")
    .arg("scripts/start_omega.py")
    .arg("--mode")
    .arg(mode)
    .arg("--workers")
    .arg(workers.to_string())
    .arg("--no-ui")
    .current_dir(&repo_root)
    // Bake in the capture fixes: Synthetic Eye + focus enforcement.
    .env("METABONK_SYNTHETIC_EYE", "1")
    .env("METABONK_EYE_FORCE_FOCUS", "1")
    .env("METABONK_EYE_IMPORT_OPAQUE_OPTIMAL", "1")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

  let mut child = cmd.spawn().map_err(|e| format!("failed to spawn omega: {e}"))?;

  if let Some(out) = child.stdout.take() {
    drain_stream(app.clone(), out, "omega-stdout");
  }
  if let Some(err) = child.stderr.take() {
    drain_stream(app.clone(), err, "omega-stderr");
  }

  *guard = Some(child);
  Ok(())
}

#[tauri::command]
fn stop_omega(app: AppHandle, state: State<'_, OmegaState>) -> Result<(), String> {
  stop_omega_inner(Some(&app), &state);
  Ok(())
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  tauri::Builder::default()
    .manage(OmegaState {
      child: Mutex::new(None),
    })
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![omega_running, start_omega, stop_omega])
    .on_window_event(|window, event| {
      if let tauri::WindowEvent::CloseRequested { .. } = event {
        let app = window.app_handle();
        let state = app.state::<OmegaState>();
        stop_omega_inner(Some(&app), &state);
      }
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
