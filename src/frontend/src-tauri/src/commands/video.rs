use std::io::{BufRead, BufReader};
use std::process::{Command, Stdio};
use std::sync::atomic::Ordering;
use std::time::{SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter, State};

use crate::services::progress::{emit_progress, ProgressEvent};

fn parse_pretrain_progress(line: &str) -> Option<ProgressEvent> {
  let s = line.trim();
  let epoch_idx = s.find("epoch ")?;
  let after = &s[epoch_idx + "epoch ".len()..];
  let frac = after.split_whitespace().next()?;
  let epoch = frac.split('/').next()?.parse::<u32>().ok()?;

  let loss = if let Some(idx) = s.find("wm_recon=") {
    s[idx + "wm_recon=".len()..].split_whitespace().next()?.parse::<f64>().ok()?
  } else if let Some(idx) = s.find("loss=") {
    s[idx + "loss=".len()..].split_whitespace().next()?.parse::<f64>().ok()?
  } else {
    return None;
  };

  Some(ProgressEvent {
    epoch,
    loss,
    accuracy: 0.0,
  })
}

fn drain_stream_with_progress(
  app: AppHandle,
  stream: impl std::io::Read + Send + 'static,
  event: &'static str,
  parse_progress: bool,
) {
  std::thread::spawn(move || {
    let reader = BufReader::new(stream);
    for line in reader.lines() {
      let Ok(line) = line else { break };
      let _ = app.emit(event, line.clone());
      if parse_progress {
        if let Some(evt) = parse_pretrain_progress(&line) {
          emit_progress(&app, evt);
        }
      }
    }
  });
}

#[tauri::command]
pub fn video_processing_running(state: State<'_, crate::VideoProcessingState>) -> bool {
  state.running.load(Ordering::SeqCst)
}

#[tauri::command]
pub fn process_videos(
  app: AppHandle,
  state: State<'_, crate::VideoProcessingState>,
  syn: State<'_, crate::SynapticState>,
  video_dir: String,
  fps: u32,
  resize: [u32; 2],
) -> Result<String, String> {
  syn.gate.lock().unwrap().fire("process_videos")?;
  if state
    .running
    .compare_exchange(false, true, Ordering::SeqCst, Ordering::SeqCst)
    .is_err()
  {
    return Err("video processing already running".to_string());
  }

  let run_id = format!(
    "run-video-{}",
    SystemTime::now()
      .duration_since(UNIX_EPOCH)
      .unwrap_or_default()
      .as_secs()
  );
  *state.run_id.lock().unwrap() = Some(run_id.clone());
  let run_id_for_thread = run_id.clone();
  let running = state.running.clone();
  let run_id_slot = state.run_id.clone();

  std::thread::spawn(move || {
    let result = (|| -> Result<serde_json::Value, String> {
      let root = crate::nervous_system::util::resolve_metabonk_root(&app)?;
      let py = crate::nervous_system::util::python_bin();

      let npz_dir = root.join("rollouts").join("video_demos");
      let labeled_dir = root.join("rollouts").join("video_demos_labeled");
      let pt_dir = root.join("rollouts").join("video_rollouts");
      let _ = std::fs::create_dir_all(&npz_dir);
      let _ = std::fs::create_dir_all(&labeled_dir);
      let _ = std::fs::create_dir_all(&pt_dir);

      let _ = app.emit(
        "video-processing-start",
        serde_json::json!({
          "run_id": run_id_for_thread,
          "video_dir": video_dir,
          "npz_dir": npz_dir.to_string_lossy(),
          "labeled_dir": labeled_dir.to_string_lossy(),
          "pt_dir": pt_dir.to_string_lossy(),
        }),
      );

      // 1) Extract trajectories from videos -> NPZ rollouts.
      let mut extract = Command::new(&py);
      extract
        .arg("-u")
        .arg("scripts/video_to_trajectory.py")
        .arg("--video-dir")
        .arg(&video_dir)
        .arg("--output-dir")
        .arg(npz_dir.to_string_lossy().to_string())
        .arg("--fps")
        .arg(fps.to_string())
        .arg("--resize")
        .arg(resize[0].to_string())
        .arg(resize[1].to_string())
        .current_dir(&root)
        .env("PYTHONPATH", root.to_string_lossy().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

      let mut child = extract.spawn().map_err(|e| format!("failed to spawn video_to_trajectory: {e}"))?;
      if let Some(out) = child.stdout.take() {
        drain_stream_with_progress(app.clone(), out, "video-processing-stdout", false);
      }
      if let Some(err) = child.stderr.take() {
        drain_stream_with_progress(app.clone(), err, "video-processing-stderr", false);
      }
      let status = child
        .wait()
        .map_err(|e| format!("failed to wait video_to_trajectory: {e}"))?;
      if !status.success() {
        return Err(format!("video_to_trajectory failed (exit={status})"));
      }

      // 2) Run offline pretraining pipeline on the produced rollouts.
      let mut pretrain = Command::new(&py);
      pretrain
        .arg("-u")
        .arg("scripts/video_pretrain.py")
        .arg("--phase")
        .arg("all")
        .arg("--npz-dir")
        .arg(npz_dir.to_string_lossy().to_string())
        .arg("--labeled-npz-dir")
        .arg(labeled_dir.to_string_lossy().to_string())
        .arg("--pt-dir")
        .arg(pt_dir.to_string_lossy().to_string())
        .arg("--device")
        .arg("cuda")
        .current_dir(&root)
        .env("PYTHONPATH", root.to_string_lossy().to_string())
        .stdout(Stdio::piped())
        .stderr(Stdio::piped());

      let mut child = pretrain.spawn().map_err(|e| format!("failed to spawn video_pretrain: {e}"))?;
      if let Some(out) = child.stdout.take() {
        drain_stream_with_progress(app.clone(), out, "video-pretrain-stdout", true);
      }
      if let Some(err) = child.stderr.take() {
        drain_stream_with_progress(app.clone(), err, "video-pretrain-stderr", false);
      }
      let status = child.wait().map_err(|e| format!("failed to wait video_pretrain: {e}"))?;
      if !status.success() {
        return Err(format!("video_pretrain failed (exit={status})"));
      }

      Ok(serde_json::json!({
        "run_id": run_id_for_thread,
        "npz_dir": npz_dir.to_string_lossy(),
        "labeled_dir": labeled_dir.to_string_lossy(),
        "pt_dir": pt_dir.to_string_lossy(),
      }))
    })();

    let _ = match &result {
      Ok(payload) => app.emit(
        "video-processing-exit",
        serde_json::json!({ "ok": true, "run_id": run_id_for_thread, "payload": payload }),
      ),
      Err(err) => app.emit(
        "video-processing-exit",
        serde_json::json!({ "ok": false, "run_id": run_id_for_thread, "error": err }),
      ),
    };

    *run_id_slot.lock().unwrap() = None;
    running.store(false, Ordering::SeqCst);
  });

  Ok(run_id)
}
