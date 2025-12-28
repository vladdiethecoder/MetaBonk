use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};
use std::sync::Mutex;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter, Manager, State};
use sysinfo::System;

struct OmegaState {
  child: Mutex<Option<Child>>,
}

struct DiscoveryState {
  child: Mutex<Option<Child>>,
}

fn discovery_dir(repo_root: &Path, env_id: &str) -> PathBuf {
  repo_root.join("cache").join("discovery").join(env_id)
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
          let trimmed = l.trim_start();
          if trimmed.starts_with('{') && l.contains("\"__meta_event\"") {
            if let Ok(v) = serde_json::from_str::<serde_json::Value>(&l) {
              if let Some(kind) = v.get("__meta_event").and_then(|x| x.as_str()) {
                match kind {
                  "reasoning_trace" => {
                    let _ = app.emit("agent-thought", v);
                    continue;
                  }
                  _ => {
                    let _ = app.emit("omega-meta", v);
                    continue;
                  }
                }
              }
            }
          }
          let _ = app.emit(event, l);
        }
        Err(_) => break,
      }
    }
  });
}

fn parse_json_from_output(raw: &str) -> Result<serde_json::Value, String> {
  let s = raw.trim();
  if s.is_empty() {
    return Err("no output".to_string());
  }
  if let Ok(v) = serde_json::from_str::<serde_json::Value>(s) {
    return Ok(v);
  }
  let start = s.find('{').ok_or_else(|| "json start not found".to_string())?;
  let end = s.rfind('}').ok_or_else(|| "json end not found".to_string())?;
  if end <= start {
    return Err("invalid json bounds".to_string());
  }
  serde_json::from_str::<serde_json::Value>(&s[start..=end]).map_err(|e| format!("failed to parse json: {e}"))
}

fn read_json_file(path: &Path) -> Result<serde_json::Value, String> {
  let content = std::fs::read_to_string(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;
  serde_json::from_str::<serde_json::Value>(&content).map_err(|e| format!("failed to parse json {path:?}: {e}"))
}

fn parse_env_exports(path: &Path) -> Result<serde_json::Value, String> {
  let content = std::fs::read_to_string(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;
  let mut map = serde_json::Map::new();
  for line in content.lines() {
    let line = line.trim();
    if !line.starts_with("export ") {
      continue;
    }
    let rest = line.trim_start_matches("export ").trim();
    if let Some((k, v)) = rest.split_once('=') {
      let key = k.trim().to_string();
      let mut val = v.trim().to_string();
      if (val.starts_with('"') && val.ends_with('"')) || (val.starts_with('\'') && val.ends_with('\'')) {
        val = val[1..val.len() - 1].to_string();
      }
      map.insert(key, serde_json::Value::String(val));
    }
  }
  Ok(serde_json::Value::Object(map))
}

fn read_gpu_stats() -> Option<serde_json::Value> {
  let output = Command::new("nvidia-smi")
    .arg("--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu")
    .arg("--format=csv,noheader,nounits")
    .output()
    .ok()?;
  if !output.status.success() {
    return None;
  }
  let stdout = String::from_utf8_lossy(&output.stdout);
  let line = stdout.lines().next()?.trim();
  if line.is_empty() {
    return None;
  }
  let parts: Vec<&str> = line.split(',').map(|s| s.trim()).collect();
  let util = parts.get(0).and_then(|v| v.parse::<f32>().ok());
  let mem_used = parts.get(1).and_then(|v| v.parse::<f32>().ok());
  let mem_total = parts.get(2).and_then(|v| v.parse::<f32>().ok());
  let temp = parts.get(3).and_then(|v| v.parse::<f32>().ok());
  Some(serde_json::json!({
    "util_pct": util,
    "mem_used_mb": mem_used,
    "mem_total_mb": mem_total,
    "temp_c": temp,
  }))
}

fn spawn_system_telemetry(app: AppHandle) {
  std::thread::spawn(move || {
    let mut sys = System::new();
    loop {
      sys.refresh_cpu();
      sys.refresh_memory();
      let cpu = sys.global_cpu_info().cpu_usage();
      let mem_used_mb = sys.used_memory() as f32 / 1024.0;
      let mem_total_mb = sys.total_memory() as f32 / 1024.0;
      let payload = serde_json::json!({
        "ts": SystemTime::now()
          .duration_since(UNIX_EPOCH)
          .unwrap_or_default()
          .as_secs_f64(),
        "cpu": {
          "usage_pct": cpu,
          "cores": sys.cpus().len(),
        },
        "memory": {
          "used_mb": mem_used_mb,
          "total_mb": mem_total_mb,
        },
        "gpu": read_gpu_stats(),
      });
      let _ = app.emit("system-telemetry", payload);
      std::thread::sleep(Duration::from_millis(1000));
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
fn start_omega(
  app: AppHandle,
  state: State<'_, OmegaState>,
  mode: String,
  workers: u32,
  env_id: Option<String>,
  synthetic_eye: Option<bool>,
  synthetic_eye_lockstep: Option<bool>,
  obs_backend: Option<String>,
  use_discovered_actions: Option<bool>,
  silicon_cortex: Option<bool>,
) -> Result<(), String> {
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
fn stop_omega(app: AppHandle, state: State<'_, OmegaState>) -> Result<(), String> {
  stop_omega_inner(Some(&app), &state);
  Ok(())
}

fn stop_discovery_inner(app: Option<&AppHandle>, state: &DiscoveryState) {
  let mut guard = state.child.lock().unwrap();
  if let Some(mut child) = guard.take() {
    let _ = child.kill();
    let _ = child.wait();
  }
  if let Some(app) = app {
    let _ = app.emit("discovery-exit", true);
  }
}

#[tauri::command]
fn discovery_running(state: State<'_, DiscoveryState>) -> bool {
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
fn discovery_status(env_id: String) -> Result<serde_json::Value, String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;
  let env_name = env_id.trim();
  let cache_dir = discovery_dir(&repo_root, env_name);
  let phase0 = cache_dir.join("input_space.json").exists();
  let phase1 = cache_dir.join("effect_map.json").exists();
  let phase2 = cache_dir.join("action_clusters.json").exists();
  let phase3 = cache_dir.join("learned_action_space.json").exists();
  let suggested = cache_dir.join("suggested_env.json").exists();
  let ppo_cfg = cache_dir.join("ppo_config.sh").exists();
  let out = serde_json::json!({
    "env_id": env_name,
    "cache_dir": cache_dir.to_string_lossy(),
    "phase_0_complete": phase0,
    "phase_1_complete": phase1,
    "phase_2_complete": phase2,
    "phase_3_complete": phase3,
    "ready_for_training": phase3,
    "artifacts": {
      "input_space_json": cache_dir.join("input_space.json").to_string_lossy(),
      "effect_map_json": cache_dir.join("effect_map.json").to_string_lossy(),
      "action_clusters_json": cache_dir.join("action_clusters.json").to_string_lossy(),
      "learned_action_space_json": cache_dir.join("learned_action_space.json").to_string_lossy(),
      "ppo_config_sh": cache_dir.join("ppo_config.sh").to_string_lossy(),
      "suggested_env_json": cache_dir.join("suggested_env.json").to_string_lossy(),
    },
    "has_suggested_env": suggested,
    "has_ppo_config": ppo_cfg,
  });
  Ok(out)
}

#[tauri::command]
fn load_action_space(env_id: String) -> Result<serde_json::Value, String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;
  let env_name = env_id.trim();
  let action_path = discovery_dir(&repo_root, env_name).join("learned_action_space.json");
  if !action_path.exists() {
    return Err("learned_action_space.json not found for env".to_string());
  }
  read_json_file(&action_path)
}

#[tauri::command]
fn load_ppo_config(env_id: String) -> Result<serde_json::Value, String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;
  let env_name = env_id.trim();
  let cfg_path = discovery_dir(&repo_root, env_name).join("ppo_config.sh");
  if !cfg_path.exists() {
    return Err("ppo_config.sh not found for env".to_string());
  }
  parse_env_exports(&cfg_path)
}

#[tauri::command]
fn start_discovery(
  app: AppHandle,
  state: State<'_, DiscoveryState>,
  env_id: String,
  env_adapter: Option<String>,
  exploration_budget: Option<u32>,
  hold_frames: Option<u32>,
) -> Result<(), String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;

  let mut guard = state.child.lock().unwrap();
  if let Some(child) = guard.as_mut() {
    if child_is_running(child) {
      return Err("discovery already running".to_string());
    }
    *guard = None;
  }

  let py = python_bin();
  let adapter = env_adapter.unwrap_or_else(|| "mock".to_string());
  let mut cmd = Command::new(py);
  cmd.arg("-u")
    .arg("scripts/run_autonomous.py")
    .arg("--env")
    .arg(env_id)
    .arg("--phase")
    .arg("all")
    .arg("--env-adapter")
    .arg(adapter);
  if let Some(b) = exploration_budget {
    cmd.arg("--exploration-budget").arg(b.to_string());
  }
  if let Some(h) = hold_frames {
    cmd.arg("--hold-frames").arg(h.to_string());
  }

  cmd.current_dir(&repo_root).stdout(Stdio::piped()).stderr(Stdio::piped());

  let mut child = cmd.spawn().map_err(|e| format!("failed to spawn discovery: {e}"))?;
  if let Some(out) = child.stdout.take() {
    drain_stream(app.clone(), out, "discovery-stdout");
  }
  if let Some(err) = child.stderr.take() {
    drain_stream(app.clone(), err, "discovery-stderr");
  }
  *guard = Some(child);
  Ok(())
}

#[tauri::command]
fn stop_discovery(app: AppHandle, state: State<'_, DiscoveryState>) -> Result<(), String> {
  stop_discovery_inner(Some(&app), &state);
  Ok(())
}

#[tauri::command]
async fn run_synthetic_eye_bench(
  id: Option<String>,
  width: Option<u32>,
  height: Option<u32>,
  lockstep: Option<bool>,
  seconds: Option<f64>,
  full_bridge: Option<bool>,
) -> Result<serde_json::Value, String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;
  let py = python_bin();

  let args_id = id.unwrap_or_else(|| "tauri-bench".to_string());
  let args_width = width.unwrap_or(1280);
  let args_height = height.unwrap_or(720);

  let do_lockstep = lockstep.unwrap_or(false);
  let do_full = full_bridge.unwrap_or(true);

  tauri::async_runtime::spawn_blocking(move || {
    let mut cmd = Command::new(py);
    cmd.arg("-u")
      .arg("scripts/synthetic_eye_bench.py")
      .arg("--id")
      .arg(args_id)
      .arg("--width")
      .arg(args_width.to_string())
      .arg("--height")
      .arg(args_height.to_string())
      .current_dir(&repo_root)
      .stdout(Stdio::piped())
      .stderr(Stdio::piped());

    if do_lockstep {
      cmd.arg("--lockstep");
    }
    if !do_full {
      cmd.arg("--no-full-bridge");
    }
    if let Some(s) = seconds {
      if s > 0.0 {
        cmd.arg("--seconds").arg(format!("{s}"));
      }
    }

    let out = cmd.output().map_err(|e| format!("failed to run bench: {e}"))?;
    if !out.status.success() {
      let stdout = String::from_utf8_lossy(&out.stdout);
      let stderr = String::from_utf8_lossy(&out.stderr);
      return Err(format!("bench failed (exit={})\nstdout:\n{}\nstderr:\n{}", out.status, stdout, stderr));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_json_from_output(&stdout)
  })
  .await
  .map_err(|e| format!("bench join failed: {e}"))?
}

#[cfg_attr(mobile, tauri::mobile_entry_point)]
pub fn run() {
  // WebKitGTK can crash on NVIDIA/Wayland when DMA-BUF renderer is enabled.
  // Default to disabling it on Linux unless the user explicitly overrides.
  if cfg!(target_os = "linux") {
    if std::env::var_os("WEBKIT_DISABLE_DMABUF_RENDERER").is_none() {
      std::env::set_var("WEBKIT_DISABLE_DMABUF_RENDERER", "1");
    }
    if std::env::var_os("WEBKIT_DISABLE_COMPOSITING_MODE").is_none() {
      std::env::set_var("WEBKIT_DISABLE_COMPOSITING_MODE", "1");
    }
  }
  tauri::Builder::default()
    .manage(OmegaState {
      child: Mutex::new(None),
    })
    .manage(DiscoveryState {
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
      spawn_system_telemetry(app.handle().clone());
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      omega_running,
      start_omega,
      stop_omega,
      discovery_running,
      discovery_status,
      load_action_space,
      load_ppo_config,
      start_discovery,
      stop_discovery,
      run_synthetic_eye_bench
    ])
    .on_window_event(|window, event| {
      if let tauri::WindowEvent::CloseRequested { .. } = event {
        let app = window.app_handle();
        let state = app.state::<OmegaState>();
        stop_omega_inner(Some(&app), &state);
        let discovery = app.state::<DiscoveryState>();
        stop_discovery_inner(Some(&app), &discovery);
      }
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}
