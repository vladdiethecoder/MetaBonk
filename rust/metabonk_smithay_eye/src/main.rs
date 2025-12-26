mod args;
mod compositor;
mod frame_socket;
mod protocols;
mod vulkan_producer;
mod xwayland_watchdog;

use std::{
    fs,
    os::fd::IntoRawFd,
    sync::{Arc, Mutex},
    thread,
    time::{Duration, Instant},
};

use anyhow::Context;
use clap::Parser;
use metabonk_frame_abi::{FrameV1, MsgType, PlaneV1, ResetV1};
use tracing::{debug, info, warn};

use crate::{
    args::Args,
    compositor::{spawn_wayland_xwayland_thread, LatestDmabuf},
    frame_socket::FrameServer,
    vulkan_producer::{VkSelect, VulkanProducer},
};

fn fd_valid(fd: i32) -> bool {
    if fd < 0 {
        return false;
    }
    let ret = unsafe { libc::fcntl(fd, libc::F_GETFD) };
    ret != -1
}

fn main() -> anyhow::Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env().unwrap_or_else(|_| "info".into()),
        )
        .init();

    let args = Args::parse();
    let run_dir = args.resolved_run_dir();
    fs::create_dir_all(&run_dir).with_context(|| format!("create_dir_all({run_dir})"))?;

    let sock_path = args.resolved_frame_sock();
    let mut server = FrameServer::bind(&sock_path)?;

    let latest_dmabuf: Arc<Mutex<Option<LatestDmabuf>>> = Arc::new(Mutex::new(None));
    let export_period: Arc<Mutex<Duration>> = Arc::new(Mutex::new(Duration::from_secs_f64(
        1.0 / f64::from(args.fps.max(1)),
    )));
    let export_size: Arc<Mutex<(u32, u32)>> = Arc::new(Mutex::new((args.width, args.height)));

    let select = VkSelect {
        device_index: args.vk_device_index,
        device_name_contains: args.vk_device_name_contains.clone(),
    };
    let mut producer = VulkanProducer::new(args.width, args.height, &args.format, args.slots.max(2), select)?;

    // If XWayland is enabled, start a per-instance Wayland socket + XWayland + XWM in a background thread.
    // The thread writes compositor.env only once XWayland is ready (dynamic DISPLAY handshake).
    if args.xwayland {
        let main_dev: libc::dev_t = if let Ok(node) = std::env::var("METABONK_DRM_RENDER_NODE") {
            let md = std::fs::metadata(&node).with_context(|| format!("metadata({node})"))?;
            use std::os::unix::fs::MetadataExt;
            md.rdev() as libc::dev_t
        } else if let Some(dev_id) = producer.drm_main_device_dev_id() {
            dev_id
        } else {
            anyhow::bail!(
                "unable to resolve DRM device id for linux-dmabuf feedback (set METABONK_DRM_RENDER_NODE explicitly)"
            );
        };

        // When running as a compositor, keep the Wayland socket isolated per instance by rebinding
        // XDG_RUNTIME_DIR to the instance run_dir for this process. Do this *after* resolving the
        // frame socket path so we don't accidentally nest run_dir paths.
        std::env::set_var("XDG_RUNTIME_DIR", &run_dir);
        if args.xwayland_cmd.is_some() || args.xwayland_display.is_some() {
            warn!("--xwayland-cmd/--xwayland-display are ignored (Smithay-managed XWayland is used).");
        }
        let wayland_name = args.resolved_wayland_display();
        let _compositor = spawn_wayland_xwayland_thread(
            std::path::PathBuf::from(&run_dir),
            wayland_name,
            sock_path.clone(),
            args.width,
            args.height,
            main_dev,
            export_period.clone(),
            export_size.clone(),
            latest_dmabuf.clone(),
        );
    } else {
        // Legacy mode: no Wayland compositor. Still emit the frame socket location for the worker.
        write_env_file(&run_dir, &args, &sock_path)?;
    }
    info!("frame socket: {sock_path}");

    let mut phase: u8 = 0;
    let mut saw_xwayland_dmabuf = false;
    let start = std::time::Instant::now();
    let mut last_ok_frame_at = Instant::now();
    let mut consecutive_failures: u32 = 0;
    let mut prev_connected: bool = false;
    let mut last_reset_sent_at: Option<Instant> = None;
    let dmabuf_wait_s: f64 = std::env::var("METABONK_XWAYLAND_DMABUF_WAIT_S")
        .ok()
        .and_then(|s| s.parse::<f64>().ok())
        .unwrap_or(60.0);
    if args.xwayland {
        info!("XWayland DMA-BUF wait timeout: {dmabuf_wait_s}s (env METABONK_XWAYLAND_DMABUF_WAIT_S)");
    }
    let passthrough: bool = std::env::var("METABONK_SYNTHETIC_EYE_PASSTHROUGH")
        .ok()
        .map(|v| matches!(v.trim().to_ascii_lowercase().as_str(), "1" | "true" | "yes" | "on"))
        .unwrap_or(false);
    if args.xwayland && passthrough {
        info!("synthetic_eye passthrough enabled (METABONK_SYNTHETIC_EYE_PASSTHROUGH=1): forwarding XWayland DMA-BUF without Vulkan import");
    } else if args.xwayland {
        info!("synthetic_eye passthrough disabled (METABONK_SYNTHETIC_EYE_PASSTHROUGH=0): using Vulkan export path");
    }
    loop {
        let t0 = std::time::Instant::now();
        let connected_now = server.is_connected();
        if prev_connected && !connected_now {
            // Worker disconnected (or sendmsg failed). Drop any outstanding consumer-release waits so
            // we don't wedge the GPU queue on semaphores that will never signal.
            warn!("worker disconnected; clearing consumer sync chain");
            producer.on_consumer_disconnect();
        }
        prev_connected = connected_now;

        let (dmabuf, acquire_fd) = {
            let mut guard = latest_dmabuf.lock().unwrap();
            if let Some(latest) = guard.as_mut() {
                (Some(latest.dmabuf.clone()), latest.acquire_fence.take())
            } else {
                (None, None)
            }
        };
        if !saw_xwayland_dmabuf && dmabuf.is_some() {
            saw_xwayland_dmabuf = true;
            info!("XWayland DMA-BUF source detected; exporting composed frames");
        }
        if args.xwayland && !saw_xwayland_dmabuf && dmabuf_wait_s > 0.0 && start.elapsed() > Duration::from_secs_f64(dmabuf_wait_s) {
            anyhow::bail!(
                "no XWayland DMA-BUF source detected after {dmabuf_wait_s}s (refusing to export test-pattern frames in GPU-only mode)"
            );
        }

        // GPU-only compositor mode: never export test-pattern frames once XWayland is enabled.
        // If the XWayland DMA-BUF source disappears (e.g. output reset), signal RESET and wait for
        // the source to reappear, rather than sending synthetic frames.
        if args.xwayland && dmabuf.is_none() {
            // Before the first DMA-BUF appears, just wait (up to dmabuf_wait_s).
            // After we have seen DMA-BUF at least once, treat this as a transient output reset and
            // force the worker to drop the rollout segment.
            if saw_xwayland_dmabuf && server.is_connected() {
                let now = Instant::now();
                let should_send = last_reset_sent_at
                    .map(|t| now.duration_since(t) > Duration::from_secs(1))
                    .unwrap_or(true);
                if should_send {
                    let reset = ResetV1 { reason: 2 }.encode_payload();
                    let _ = server.send_message(MsgType::Reset, &reset, &[]);
                    last_reset_sent_at = Some(now);
                }
            }
            producer.on_consumer_disconnect();
            producer.drop_cached_source();
            consecutive_failures = consecutive_failures.saturating_add(1);
            thread::sleep(Duration::from_millis(50));
            phase = phase.wrapping_add(1);
            continue;
        }

        let render_res = if let Some(d) = dmabuf.as_ref() {
            let wait_fd = acquire_fd.map(|fd| fd.into_raw_fd());
            if args.xwayland && passthrough {
                let (w, h) = export_size.lock().map(|s| *s).unwrap_or((args.width, args.height));
                producer.render_passthrough_dmabuf(d, Some((w, h)), wait_fd)
            } else {
                match producer.render_from_dmabuf(d, wait_fd) {
                    Ok(v) => Ok(v),
                    Err(e) => {
                        if args.xwayland && passthrough {
                            // Optional fallback within GPU-only constraints.
                            warn!("render_from_dmabuf failed; falling back to passthrough: {e}");
                            let (w, h) = export_size.lock().map(|s| *s).unwrap_or((args.width, args.height));
                            producer.render_passthrough_dmabuf(d, Some((w, h)), None)
                        } else {
                            warn!("render_from_dmabuf failed with passthrough disabled: {e}");
                            Err(e)
                        }
                    }
                }
            }
        } else {
            producer.render_next(phase)
        };
        match render_res {
            Ok(rf) => {
                // Prefer the actual exported buffer dimensions (rf.width/rf.height). export_size is a
                // requested geometry target and may differ if a client ignores resize requests.
                let mut out_w = rf.width;
                let mut out_h = rf.height;
                if out_w == 0 || out_h == 0 {
                    let (w, h) = export_size.lock().map(|s| *s).unwrap_or((0, 0));
                    if w != 0 && h != 0 {
                        out_w = w;
                        out_h = h;
                    }
                }
                if out_w == 0 || out_h == 0 {
                    warn!(
                        out_w,
                        out_h,
                        rf_w = rf.width,
                        rf_h = rf.height,
                        "export frame has zero dimensions"
                    );
                }
                let payload = FrameV1 {
                    frame_id: rf.frame_id,
                    width: out_w,
                    height: out_h,
                    drm_fourcc: rf.drm_fourcc,
                    modifier: rf.modifier,
                    dmabuf_fd_count: 1,
                    planes: vec![PlaneV1 {
                        fd_index: 0,
                        stride: rf.stride,
                        offset: rf.offset,
                        size_bytes: rf.size_bytes,
                    }],
                };
                let payload_bytes = payload.encode_payload();
                let mut fds: Vec<i32> = vec![rf.dmabuf_fd as i32];
                let mut acquire_fd = rf.acquire_fence_fd;
                let mut release_fd = rf.release_fence_fd;
                let acquire_ok = fd_valid(acquire_fd);
                let release_ok = fd_valid(release_fd);
                if acquire_ok && release_ok {
                    fds.push(acquire_fd as i32);
                    fds.push(release_fd as i32);
                } else {
                    debug!(
                        "dropping invalid fence fds: acquire_fd={} ok={} release_fd={} ok={}",
                        acquire_fd,
                        acquire_ok,
                        release_fd,
                        release_ok
                    );
                    acquire_fd = -1;
                    release_fd = -1;
                    let _ = nix::unistd::close(rf.acquire_fence_fd);
                    let _ = nix::unistd::close(rf.release_fence_fd);
                }

                if let Err(e) = server.send_message(MsgType::Frame, &payload_bytes, &fds) {
                    warn!("frame send failed: {e}");
                    // On disconnect we can't reliably send RESET; instead clear sync chain
                    // so the compositor can resume cleanly once the worker reconnects.
                    producer.on_consumer_disconnect();
                    producer.drop_cached_source();
                    rf.close_fds();
                    thread::sleep(Duration::from_millis(200));
                    consecutive_failures = consecutive_failures.saturating_add(1);
                } else {
                    rf.close_fds();
                    last_ok_frame_at = Instant::now();
                    consecutive_failures = 0;
                }
            }
            Err(e) => {
                warn!("render failed: {e}");
                // If rendering fails, force the worker to drop buffered rollout state.
                // This prevents training on frozen/invalid frames after a compositor hiccup.
                if server.is_connected() {
                    let now = Instant::now();
                    let should_send = last_reset_sent_at
                        .map(|t| now.duration_since(t) > Duration::from_secs(1))
                        .unwrap_or(true);
                    if should_send {
                        let reset = ResetV1 { reason: 2 }.encode_payload();
                        let _ = server.send_message(MsgType::Reset, &reset, &[]);
                        last_reset_sent_at = Some(now);
                    }
                } else {
                    producer.on_consumer_disconnect();
                }
                producer.on_consumer_disconnect();
                producer.drop_cached_source();
                thread::sleep(Duration::from_millis(200));
                consecutive_failures = consecutive_failures.saturating_add(1);
            }
        }

        // Watchdog: if we haven't successfully delivered frames recently, exit so the launcher can restart us.
        // Keep this conservative: transient backpressure from the CUDA consumer should not trigger restarts.
        if last_ok_frame_at.elapsed() > Duration::from_secs(60) || consecutive_failures >= 300 {
            anyhow::bail!(
                "synthetic eye stalled: last_ok_frame_age_s={} consecutive_failures={}",
                last_ok_frame_at.elapsed().as_secs_f64(),
                consecutive_failures
            );
        }
        phase = phase.wrapping_add(1);
        let elapsed = t0.elapsed();
        let period = export_period.lock().map(|p| *p).unwrap_or(Duration::from_millis(16));
        if elapsed < period {
            thread::sleep(period - elapsed);
        }
    }
}

fn write_env_file(run_dir: &str, args: &Args, frame_sock: &str) -> anyhow::Result<()> {
    let path = std::path::Path::new(run_dir).join("compositor.env");
    let mut lines = Vec::new();
    lines.push(format!("METABONK_INSTANCE_ID={}", args.id));
    lines.push(format!("WAYLAND_DISPLAY={}", args.resolved_wayland_display()));
    lines.push(format!("METABONK_FRAME_SOCK={frame_sock}"));
    fs::write(&path, lines.join("\n") + "\n").with_context(|| format!("write {path:?}"))?;
    Ok(())
}
