use clap::Parser;

#[derive(Parser, Debug, Clone)]
#[command(name = "metabonk_smithay_eye", about = "MetaBonk synthetic eye (DMABuf + explicit sync exporter)")]
pub struct Args {
    /// Instance id used for paths and socket names.
    #[arg(long, default_value = "0")]
    pub id: String,

    #[arg(long, default_value_t = 1280)]
    pub width: u32,

    #[arg(long, default_value_t = 720)]
    pub height: u32,

    #[arg(long, default_value_t = 60)]
    pub fps: u32,

    /// Number of export slots (DMA-BUF buffers) to keep in flight.
    ///
    /// Too few slots can deadlock if the consumer's release-fence signaling lags behind
    /// (e.g. due to GPU load). Higher values trade VRAM for robustness.
    #[arg(long, default_value_t = 8)]
    pub slots: usize,

    /// Runtime directory for this instance (frame.sock, compositor.env, etc).
    #[arg(long)]
    pub run_dir: Option<String>,

    /// Frame export socket path.
    #[arg(long)]
    pub frame_sock: Option<String>,

    /// Wayland socket name (reserved; full Smithay compositor integration is TODO).
    #[arg(long)]
    pub wayland_display: Option<String>,

    /// DRM FourCC to export (currently only ARGB8888 is supported).
    #[arg(long, default_value = "ARGB8888")]
    pub format: String,

    /// Select Vulkan physical device by index (0-based, enumeration order).
    #[arg(long)]
    pub vk_device_index: Option<u32>,

    /// Select Vulkan physical device by name substring (case-insensitive).
    #[arg(long)]
    pub vk_device_name_contains: Option<String>,

    /// Enable XWayland supervision (requires an externally managed XWayland command for now).
    ///
    /// Full Smithay-native XWayland integration (rootless XWayland + WM) is TODO.
    #[arg(long, default_value_t = false)]
    pub xwayland: bool,

    /// Command to launch XWayland (space-separated). Only used with --xwayland.
    ///
    /// Example:
    ///   --xwayland-cmd "Xwayland :102 -rootless -terminate"
    #[arg(long)]
    pub xwayland_cmd: Option<String>,

    /// DISPLAY string to heartbeat (e.g. ":102"). Only used with --xwayland.
    #[arg(long)]
    pub xwayland_display: Option<String>,

    /// Heartbeat period for XWayland (ms).
    #[arg(long, default_value_t = 500)]
    pub xwayland_heartbeat_ms: u64,

    /// Consecutive heartbeat failures before restart.
    #[arg(long, default_value_t = 3)]
    pub xwayland_fail_threshold: u32,
}

impl Args {
    pub fn resolved_run_dir(&self) -> String {
        if let Some(d) = &self.run_dir {
            return d.clone();
        }
        if let Ok(xdg) = std::env::var("XDG_RUNTIME_DIR") {
            if !xdg.trim().is_empty() {
                return format!("{}/metabonk/{}", xdg.trim_end_matches('/'), self.id);
            }
        }
        format!("/tmp/metabonk/{}", self.id)
    }

    pub fn resolved_frame_sock(&self) -> String {
        if let Some(p) = &self.frame_sock {
            return p.clone();
        }
        format!("{}/frame.sock", self.resolved_run_dir())
    }

    pub fn resolved_wayland_display(&self) -> String {
        if let Some(n) = &self.wayland_display {
            return n.clone();
        }
        format!("metabonk-wl-{}", self.id)
    }
}
