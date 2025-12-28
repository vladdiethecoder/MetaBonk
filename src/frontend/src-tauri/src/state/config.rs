use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};
use std::sync::Mutex;

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AppConfig {
  pub gpu_enabled: bool,
  pub default_model: String,
  pub data_dir: PathBuf,
  pub log_level: String,
}

impl Default for AppConfig {
  fn default() -> Self {
    Self {
      gpu_enabled: true,
      default_model: "omega_protocol".to_string(),
      data_dir: PathBuf::from("./data"),
      log_level: "info".to_string(),
    }
  }
}

pub struct ConfigState {
  path: PathBuf,
  config: Mutex<AppConfig>,
}

impl ConfigState {
  pub fn load_or_default(path: PathBuf) -> Self {
    let cfg = read_json(&path).unwrap_or_default();
    Self {
      path,
      config: Mutex::new(cfg),
    }
  }

  pub fn get(&self) -> AppConfig {
    self.config.lock().unwrap().clone()
  }

  pub fn set(&self, config: AppConfig) -> Result<(), String> {
    write_json(&self.path, &config)?;
    *self.config.lock().unwrap() = config;
    Ok(())
  }

  pub fn path(&self) -> PathBuf {
    self.path.clone()
  }
}

fn read_json(path: &Path) -> Result<AppConfig, String> {
  let content = std::fs::read_to_string(path).map_err(|e| format!("failed to read {path:?}: {e}"))?;
  serde_json::from_str::<AppConfig>(&content).map_err(|e| format!("failed to parse {path:?}: {e}"))
}

fn write_json(path: &Path, config: &AppConfig) -> Result<(), String> {
  if let Some(parent) = path.parent() {
    std::fs::create_dir_all(parent).map_err(|e| format!("failed to create {parent:?}: {e}"))?;
  }
  let content = serde_json::to_string_pretty(config).map_err(|e| format!("failed to serialize config: {e}"))?;
  std::fs::write(path, content).map_err(|e| format!("failed to write {path:?}: {e}"))
}
