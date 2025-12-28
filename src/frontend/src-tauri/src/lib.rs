mod nervous_system;
mod commands;
mod python;
mod services;
mod state;

use std::sync::atomic::AtomicBool;
use std::sync::{Arc, Mutex};

use tauri::{Manager, State};

use nervous_system::central::OmegaState;
use nervous_system::peripheral::DiscoveryState;
use nervous_system::synaptic::{SynapticCommandGate, SynapticState};

struct VideoProcessingState {
  running: Arc<AtomicBool>,
  run_id: Arc<Mutex<Option<String>>>,
}

#[tauri::command]
fn get_app_config(state: State<'_, state::config::ConfigState>) -> state::config::AppConfig {
  state.get()
}

#[tauri::command]
fn set_app_config(state: State<'_, state::config::ConfigState>, config: state::config::AppConfig) -> Result<(), String> {
  state.set(config)
}

#[tauri::command]
fn app_config_path(state: State<'_, state::config::ConfigState>) -> String {
  state.path().to_string_lossy().to_string()
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
    .manage(SynapticState {
      gate: Mutex::new(SynapticCommandGate::new()),
    })
    .manage(VideoProcessingState {
      running: Arc::new(AtomicBool::new(false)),
      run_id: Arc::new(Mutex::new(None)),
    })
    .setup(|app| {
      app.handle().plugin(tauri_plugin_dialog::init())?;
      if cfg!(debug_assertions) {
        app.handle().plugin(
          tauri_plugin_log::Builder::default()
            .level(log::LevelFilter::Info)
            .build(),
        )?;
      }

      let cfg_path = app
        .handle()
        .path()
        .app_config_dir()
        .map(|dir| dir.join("config.json"))
        .unwrap_or_else(|_| {
          nervous_system::util::find_repo_root()
            .unwrap_or_else(|| std::path::PathBuf::from("."))
            .join("configs")
            .join("tauri_app_config.json")
        });
      app.manage(state::config::ConfigState::load_or_default(cfg_path));

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
      nervous_system::peripheral::run_synthetic_eye_bench,
      get_app_config,
      set_app_config,
      app_config_path,
      commands::training::start_training,
      commands::training::get_training_status,
      commands::video::video_processing_running,
      commands::video::process_videos
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
