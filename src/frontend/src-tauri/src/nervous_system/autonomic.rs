use std::process::Command;
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use tauri::{AppHandle, Emitter};
use sysinfo::System;

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

pub fn spawn_system_telemetry(app: AppHandle) {
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

