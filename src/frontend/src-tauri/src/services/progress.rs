use serde::Serialize;
use tauri::{AppHandle, Emitter};

#[derive(Clone, Serialize)]
pub struct ProgressEvent {
  pub epoch: u32,
  pub loss: f64,
  pub accuracy: f64,
}

pub fn emit_progress(app: &AppHandle, event: ProgressEvent) {
  let _ = app.emit("training-progress", event);
}
