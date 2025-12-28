mod nervous_system;

use std::sync::Mutex;

use tauri::Manager;

use nervous_system::central::OmegaState;
use nervous_system::peripheral::DiscoveryState;
use nervous_system::synaptic::{SynapticCommandGate, SynapticState};

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
    .manage(SynapticState {
      gate: Mutex::new(SynapticCommandGate::new()),
    })
    .setup(|app| {
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }
      nervous_system::autonomic::spawn_system_telemetry(app.handle().clone());
      Ok(())
    })
    .invoke_handler(tauri::generate_handler![
      nervous_system::central::omega_running,
      nervous_system::central::start_omega,
      nervous_system::central::stop_omega,
      nervous_system::peripheral::discovery_running,
      nervous_system::peripheral::discovery_status,
      nervous_system::peripheral::load_action_space,
      nervous_system::peripheral::load_ppo_config,
      nervous_system::peripheral::start_discovery,
      nervous_system::peripheral::stop_discovery,
      nervous_system::peripheral::run_synthetic_eye_bench
    ])
    .on_window_event(|window, event| {
      if let tauri::WindowEvent::CloseRequested { .. } = event {
        let app = window.app_handle();
        nervous_system::central::on_close(&app);
        nervous_system::peripheral::on_close(&app);
      }
    })
    .run(tauri::generate_context!())
    .expect("error while running tauri application");
}

