use std::process::{Child, Command, Stdio};
use std::sync::Mutex;

use tauri::{AppHandle, Emitter, Manager, State};

use super::synaptic::SynapticState;
use super::util::{
  child_is_running, discovery_dir, drain_stream, find_repo_root, parse_env_exports, parse_json_from_output,
  python_bin, read_json_file,
};

pub struct DiscoveryState {
  pub child: Mutex<Option<Child>>,
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
pub fn discovery_running(state: State<'_, DiscoveryState>) -> bool {
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
pub fn discovery_status(env_id: String) -> Result<serde_json::Value, String> {
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
pub fn load_action_space(env_id: String) -> Result<serde_json::Value, String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;
  let env_name = env_id.trim();
  let action_path = discovery_dir(&repo_root, env_name).join("learned_action_space.json");
  if !action_path.exists() {
    return Err("learned_action_space.json not found for env".to_string());
  }
  read_json_file(&action_path)
}

#[tauri::command]
pub fn load_ppo_config(env_id: String) -> Result<serde_json::Value, String> {
  let repo_root = find_repo_root().ok_or_else(|| "failed to locate repo root (missing scripts/)".to_string())?;
  let env_name = env_id.trim();
  let cfg_path = discovery_dir(&repo_root, env_name).join("ppo_config.sh");
  if !cfg_path.exists() {
    return Err("ppo_config.sh not found for env".to_string());
  }
  parse_env_exports(&cfg_path)
}

#[tauri::command]
pub fn start_discovery(
  app: AppHandle,
  state: State<'_, DiscoveryState>,
  syn: State<'_, SynapticState>,
  env_id: String,
  env_adapter: Option<String>,
  exploration_budget: Option<u32>,
  hold_frames: Option<u32>,
) -> Result<(), String> {
  syn.gate.lock().unwrap().fire("start_discovery")?;

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
pub fn stop_discovery(
  app: AppHandle,
  state: State<'_, DiscoveryState>,
  syn: State<'_, SynapticState>,
) -> Result<(), String> {
  syn.gate.lock().unwrap().fire("stop_discovery")?;
  stop_discovery_inner(Some(&app), &state);
  Ok(())
}

#[tauri::command]
pub async fn run_synthetic_eye_bench(
  syn: State<'_, SynapticState>,
  id: Option<String>,
  width: Option<u32>,
  height: Option<u32>,
  lockstep: Option<bool>,
  seconds: Option<f64>,
  full_bridge: Option<bool>,
) -> Result<serde_json::Value, String> {
  syn.gate.lock().unwrap().fire("run_synthetic_eye_bench")?;
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
      return Err(format!(
        "bench failed (exit={})\nstdout:\n{}\nstderr:\n{}",
        out.status, stdout, stderr
      ));
    }
    let stdout = String::from_utf8_lossy(&out.stdout);
    parse_json_from_output(&stdout)
  })
  .await
  .map_err(|e| format!("bench join failed: {e}"))?
}

pub fn on_close(app: &AppHandle) {
  let state = app.state::<DiscoveryState>();
  stop_discovery_inner(Some(app), &state);
}
