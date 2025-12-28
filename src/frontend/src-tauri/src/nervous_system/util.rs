use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};
use std::process::{Child, Command, Stdio};

use tauri::{AppHandle, Emitter, Manager};

pub fn discovery_dir(repo_root: &Path, env_id: &str) -> PathBuf {
  repo_root.join("cache").join("discovery").join(env_id)
}

pub fn find_repo_root() -> Option<PathBuf> {
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

pub fn resolve_metabonk_root(app: &AppHandle) -> Result<PathBuf, String> {
  if let Some(root) = find_repo_root() {
    return Ok(root);
  }
  let resource_dir = app
    .path()
    .resource_dir()
    .map_err(|e| format!("failed to resolve resource_dir: {e}"))?;
  for cand in [resource_dir.clone(), resource_dir.join("resources")] {
    if cand.join("scripts").join("start_omega.py").exists() && cand.join("scripts").join("stop.py").exists() {
      return Ok(cand);
    }
  }
  Err("failed to locate MetaBonk root (missing scripts/)".to_string())
}

pub fn python_bin() -> String {
  std::env::var("METABONK_TAURI_PYTHON").unwrap_or_else(|_| "python3".to_string())
}

pub fn run_stop_all(repo_root: &Path) {
  let py = python_bin();
  let _ = Command::new(py)
    .arg("scripts/stop.py")
    .arg("--all")
    .current_dir(repo_root)
    .stdout(Stdio::null())
    .stderr(Stdio::null())
    .status();
}

pub fn child_is_running(child: &mut Child) -> bool {
  match child.try_wait() {
    Ok(Some(_)) => false,
    Ok(None) => true,
    Err(_) => false,
  }
}

pub fn drain_stream(app: AppHandle, stream: impl std::io::Read + Send + 'static, event: &'static str) {
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

pub fn parse_json_from_output(raw: &str) -> Result<serde_json::Value, String> {
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

pub fn read_json_file(path: &Path) -> Result<serde_json::Value, String> {
  let content = std::fs::read_to_string(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;
  serde_json::from_str::<serde_json::Value>(&content).map_err(|e| format!("failed to parse json {path:?}: {e}"))
}

pub fn parse_env_exports(path: &Path) -> Result<serde_json::Value, String> {
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
