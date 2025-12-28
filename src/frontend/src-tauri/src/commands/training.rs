use serde::{Deserialize, Serialize};
use tauri::{AppHandle, State};

#[derive(Debug, Serialize, Deserialize)]
pub struct TrainingConfig {
  pub model: String,
  pub epochs: u32,
  pub batch_size: u32,
  pub learning_rate: f64,
}

#[tauri::command]
pub fn start_training(
  app: AppHandle,
  omega: State<'_, crate::OmegaState>,
  syn: State<'_, crate::SynapticState>,
  config: TrainingConfig,
) -> Result<String, String> {
  syn.gate.lock().unwrap().fire("start_training")?;
  // Map the doc's "training" entrypoint to MetaBonk's existing Omega stack launcher.
  // The detailed hyperparameters are currently controlled inside the Python stack; we persist
  // the requested config alongside the run for traceability.
  let run_id = crate::nervous_system::central::start_omega_inner(
    app.clone(),
    omega,
    "train".to_string(),
    1,
    None,
    Some(true),
    Some(false),
    None,
    None,
    None,
  )?;

  if let Ok(root) = crate::nervous_system::util::resolve_metabonk_root(&app) {
    let run_dir = root.join("runs").join(&run_id);
    let _ = std::fs::create_dir_all(&run_dir);
    if let Ok(json) = serde_json::to_string_pretty(&config) {
      let _ = std::fs::write(run_dir.join("training_config.json"), json);
    }
  }

  Ok(run_id)
}

#[tauri::command]
pub fn get_training_status(omega: State<'_, crate::OmegaState>, training_id: String) -> Result<String, String> {
  let mut guard = omega.child.lock().unwrap();
  match guard.as_mut() {
    None => Ok("stopped".to_string()),
    Some(job) => {
      if job.run_id != training_id {
        return Ok("unknown".to_string());
      }
      if crate::nervous_system::util::child_is_running(&mut job.child) {
        Ok("running".to_string())
      } else {
        *guard = None;
        Ok("stopped".to_string())
      }
    }
  }
}
