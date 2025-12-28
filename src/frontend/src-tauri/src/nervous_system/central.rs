use std::path::Path;
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter, Manager, State};

use super::synaptic::SynapticState;
use super::util::{child_is_running, drain_stream, find_repo_root, python_bin, run_stop_all};

pub struct OmegaState {
  pub child: Mutex<Option<Child>>,
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
pub fn omega_running(state: State<'_, OmegaState>) -> bool {
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
pub fn start_omega(
  app: AppHandle,
  state: State<'_, OmegaState>,
  syn: State<'_, SynapticState>,
  mode: String,
  workers: u32,
  env_id: Option<String>,
  synthetic_eye: Option<bool>,
  synthetic_eye_lockstep: Option<bool>,
  obs_backend: Option<String>,
  use_discovered_actions: Option<bool>,
  silicon_cortex: Option<bool>,
) -> Result<(), String> {
  syn.gate.lock().unwrap().fire("start_omega")?;

  let repo_root =
    find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/start_omega.py)".to_string())?;

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
  let now_s = SystemTime::now()
    .duration_since(UNIX_EPOCH)
    .unwrap_or_default()
    .as_secs();
  let run_id = format!("run-tauri-{now_s}");
  let run_dir = repo_root.join("runs").join(&run_id);
  let _ = std::fs::create_dir_all(run_dir.join("logs"));

  let mut cmd = Command::new(py);
  cmd.arg("-u")
    .arg("scripts/start_omega.py")
    .arg("--mode")
    .arg(mode)
    .arg("--workers")
    .arg(workers.to_string())
    .arg("--no-ui")
    .current_dir(&repo_root)
    .env("METABONK_RUN_ID", &run_id)
    .env("METABONK_RUN_DIR", run_dir.to_string_lossy().to_string())
    .env("MEGABONK_LOG_DIR", run_dir.to_string_lossy().to_string())
    // Bake in the capture fixes: Synthetic Eye + focus enforcement (override-able via command args).
    .env(
      "METABONK_SYNTHETIC_EYE",
      if synthetic_eye.unwrap_or(true) { "1" } else { "0" },
    )
    .env("METABONK_EYE_FORCE_FOCUS", "1")
    .env("METABONK_EYE_IMPORT_OPAQUE_OPTIMAL", "1")
    // Discovery artifacts are keyed by env id.
    .env("METABONK_ENV_ID", env_id.unwrap_or_else(|| "MegaBonk".to_string()))
    // Optional overrides.
    .env(
      "METABONK_SYNTHETIC_EYE_LOCKSTEP",
      if synthetic_eye_lockstep.unwrap_or(false) { "1" } else { "0" },
    )
    // UI telemetry: structured meta events forwarded from worker logs.
    .env("METABONK_EMIT_META_EVENTS", "1")
    .env("METABONK_EMIT_THOUGHTS", "1")
    .env("METABONK_FORWARD_META_EVENTS", "1")
    // Stream HUD: bake mind text into the encoded stream when supported (ffmpeg drawtext).
    .env("METABONK_STREAM_OVERLAY", "1")
    .stdout(Stdio::piped())
    .stderr(Stdio::piped());

  if let Some(v) = obs_backend.as_ref().map(|s| s.trim().to_string()).filter(|s| !s.is_empty()) {
    cmd.env("METABONK_OBS_BACKEND", v);
  }
  if let Some(use_disc) = use_discovered_actions {
    cmd.env("METABONK_USE_DISCOVERED_ACTIONS", if use_disc { "1" } else { "0" });
  }
  if let Some(enable) = silicon_cortex {
    cmd.env("METABONK_SILICON_CORTEX", if enable { "1" } else { "0" });
  }

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
pub fn stop_omega(app: AppHandle, state: State<'_, OmegaState>, syn: State<'_, SynapticState>) -> Result<(), String> {
  syn.gate.lock().unwrap().fire("stop_omega")?;
  stop_omega_inner(Some(&app), &state);
  Ok(())
}

pub fn on_close(app: &AppHandle) {
  let state = app.state::<OmegaState>();
  stop_omega_inner(Some(app), &state);
}
