use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    collections::HashMap,
    fs,
    os::fd::{AsRawFd, FromRawFd, OwnedFd, RawFd},
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::Duration,
    time::Instant,
};

use anyhow::Context;
use nix::ioctl_readwrite;
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use smithay::{
    delegate_compositor, delegate_seat, delegate_xdg_shell, delegate_xwayland_shell,
    backend::allocator::dmabuf::Dmabuf,
    backend::allocator::dmabuf::DmabufFlags,
    backend::allocator::Buffer as _,
    input::{Seat, SeatHandler, SeatState, pointer::CursorImageStatus},
    output::{Mode, Output, PhysicalProperties, Scale, Subpixel},
    reexports::{
        calloop::EventLoop,
        wayland_server::{
            backend::{ClientData, ClientId, DisconnectReason, ObjectId},
            protocol::wl_buffer,
            protocol::wl_callback,
            protocol::wl_surface::WlSurface,
            Resource,
            Client, Display, DisplayHandle,
        },
    },
    utils::{Logical, Rectangle},
    utils::{Clock, Monotonic},
    utils::Transform,
    wayland::{
        buffer::BufferHandler,
        compositor::{
            with_states, CompositorClientState, CompositorHandler, CompositorState, SurfaceAttributes,
        },
        dmabuf::{
            get_dmabuf,
            DmabufFeedbackBuilder,
            DmabufGlobal,
            DmabufHandler,
            DmabufState,
            ImportNotifier,
        },
        drm_syncobj::{supports_syncobj_eventfd, DrmSyncobjCachedState, DrmSyncobjHandler, DrmSyncobjState},
        output::OutputHandler,
        shell::xdg::{PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState, XdgToplevelSurfaceData},
        shm::{self, ShmHandler, ShmState},
        socket::ListeningSocketSource,
        xwayland_shell::{XWaylandShellHandler, XWaylandShellState},
    },
    xwayland::{
        xwm::{Reorder, ResizeEdge, WmWindowProperty, XwmId},
        X11Wm, X11Surface, XWayland, XWaylandEvent, XwmHandler,
    },
};
use smithay::reexports::wayland_server::protocol::wl_shm;
use smithay::backend::input::KeyState;
use smithay::input::keyboard::{KeyboardTarget, KeysymHandle, ModifiersState};
use smithay::utils::{IsAlive, Serial};
use smithay::wayland::seat::WaylandFocus;
use std::borrow::Cow;
use tracing::{info, warn};
use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::backend::renderer::utils::RendererSurfaceStateUserData;
use smithay::backend::renderer::utils::Buffer as SurfaceBuffer;
use crate::protocols::ext_metabonk_control_v1::server::zmetabonk_agent_v1::{self, ZmetabonkAgentV1};
use crate::protocols::ext_metabonk_control_v1::server::zmetabonk_orchestrator_v1::{self, ZmetabonkOrchestratorV1};
use crate::protocols::wl_drm::server::wl_drm::{self, WlDrm};

#[derive(Clone, Debug, PartialEq)]
pub enum KeyboardFocusTarget {
    Wayland(WlSurface),
    X11(X11Surface),
}

impl KeyboardFocusTarget {
    fn from_wl_surface(surface: &WlSurface, x11: Option<&X11Surface>) -> Self {
        if let Some(win) = x11 {
            KeyboardFocusTarget::X11(win.clone())
        } else {
            KeyboardFocusTarget::Wayland(surface.clone())
        }
    }
}

impl IsAlive for KeyboardFocusTarget {
    fn alive(&self) -> bool {
        match self {
            KeyboardFocusTarget::Wayland(surface) => surface.alive(),
            KeyboardFocusTarget::X11(window) => window.alive(),
        }
    }
}

impl WaylandFocus for KeyboardFocusTarget {
    fn wl_surface(&self) -> Option<Cow<'_, WlSurface>> {
        match self {
            KeyboardFocusTarget::Wayland(surface) => Some(Cow::Borrowed(surface)),
            KeyboardFocusTarget::X11(window) => window.wl_surface().map(Cow::Owned),
        }
    }
}

impl KeyboardTarget<EyeCompositor> for KeyboardFocusTarget {
    fn enter(
        &self,
        seat: &Seat<EyeCompositor>,
        data: &mut EyeCompositor,
        keys: Vec<KeysymHandle<'_>>,
        serial: Serial,
    ) {
        match self {
            KeyboardFocusTarget::Wayland(surface) => KeyboardTarget::enter(surface, seat, data, keys, serial),
            KeyboardFocusTarget::X11(window) => KeyboardTarget::enter(window, seat, data, keys, serial),
        }
    }

    fn leave(&self, seat: &Seat<EyeCompositor>, data: &mut EyeCompositor, serial: Serial) {
        match self {
            KeyboardFocusTarget::Wayland(surface) => KeyboardTarget::leave(surface, seat, data, serial),
            KeyboardFocusTarget::X11(window) => KeyboardTarget::leave(window, seat, data, serial),
        }
    }

    fn key(
        &self,
        seat: &Seat<EyeCompositor>,
        data: &mut EyeCompositor,
        key: KeysymHandle<'_>,
        state: KeyState,
        serial: Serial,
        time: u32,
    ) {
        match self {
            KeyboardFocusTarget::Wayland(surface) => KeyboardTarget::key(surface, seat, data, key, state, serial, time),
            KeyboardFocusTarget::X11(window) => KeyboardTarget::key(window, seat, data, key, state, serial, time),
        }
    }

    fn modifiers(
        &self,
        seat: &Seat<EyeCompositor>,
        data: &mut EyeCompositor,
        modifiers: ModifiersState,
        serial: Serial,
    ) {
        match self {
            KeyboardFocusTarget::Wayland(surface) => KeyboardTarget::modifiers(surface, seat, data, modifiers, serial),
            KeyboardFocusTarget::X11(window) => KeyboardTarget::modifiers(window, seat, data, modifiers, serial),
        }
    }
}

const DMA_BUF_SYNC_READ: u32 = 1 << 0;
#[allow(dead_code)]
const DMA_BUF_SYNC_WRITE: u32 = 1 << 1;

#[repr(C)]
#[derive(Debug, Copy, Clone)]
struct DmaBufExportSyncFile {
    flags: u32,
    fd: i32,
}

ioctl_readwrite!(dma_buf_export_sync_file, b'b', 2, DmaBufExportSyncFile);

pub(crate) fn export_dmabuf_read_sync_file(dmabuf: &Dmabuf) -> anyhow::Result<Option<OwnedFd>> {
    let Some(handle) = dmabuf.handles().next() else {
        return Ok(None);
    };
    let dmabuf_fd: RawFd = handle.as_raw_fd();
    let mut arg = DmaBufExportSyncFile {
        flags: DMA_BUF_SYNC_READ,
        fd: -1,
    };
    // SAFETY: ioctl operates on the dmabuf fd and writes into `arg`.
    unsafe { dma_buf_export_sync_file(dmabuf_fd, &mut arg) }.context("DMA_BUF_IOCTL_EXPORT_SYNC_FILE")?;
    if arg.fd < 0 {
        return Ok(None);
    }
    // SAFETY: kernel returned a new fd for us to own.
    Ok(Some(unsafe { OwnedFd::from_raw_fd(arg.fd) }))
}

#[derive(Default)]
struct ClientState {
    compositor_state: CompositorClientState,
    pid: u32,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
}

#[derive(Clone, Copy, Debug)]
struct XWaylandPid(u32);

#[derive(Clone, Debug)]
struct WlDrmGlobalData {
    device_path: String,
    main_dev: u64,
    formats: Vec<u32>,
    capabilities: u32,
}

#[derive(Clone, Debug)]
struct WlDrmData {
    device_path: String,
    main_dev: u64,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AgentTier {
    Featured,
    Observation,
    Background,
    Suspended,
}

impl AgentTier {
    fn target_interval(&self) -> Duration {
        match self {
            AgentTier::Featured => Duration::from_millis(16),
            AgentTier::Observation => Duration::from_millis(33),
            // Training correctness: do not dilate time for background agents. Keep 60Hz pacing and
            // use resolution (configure) as the primary density lever.
            AgentTier::Background => Duration::from_millis(16),
            AgentTier::Suspended => Duration::from_secs(1),
        }
    }

    fn default_geometry(&self, fallback: (i32, i32)) -> (i32, i32) {
        match self {
            AgentTier::Featured => fallback,
            AgentTier::Observation => (1280, 720),
            AgentTier::Background => (640, 360),
            AgentTier::Suspended => (320, 180),
        }
    }
}

#[derive(Debug)]
pub struct AgentState {
    pub pid: u32,
    pub tier: AgentTier,
    pub last_frame_time: Instant,
    pub target_interval: Duration,
    pub geometry: (i32, i32),
    pub surface: Option<WlSurface>,
    pub app_id: String,
    pub toplevel: Option<ToplevelSurface>,
}

impl AgentState {
    fn new(pid: u32, fallback_geometry: (i32, i32)) -> Self {
        let tier = AgentTier::Observation;
        Self {
            pid,
            tier,
            last_frame_time: Instant::now() - tier.target_interval(),
            target_interval: tier.target_interval(),
            geometry: tier.default_geometry(fallback_geometry),
            surface: None,
            app_id: "unknown_app".into(),
            toplevel: None,
        }
    }
}

#[derive(Debug)]
struct PendingFrameCallback {
    due: Instant,
    seq: u64,
    pid: u32,
    callback: wl_callback::WlCallback,
}

impl PartialEq for PendingFrameCallback {
    fn eq(&self, other: &Self) -> bool {
        self.due == other.due && self.seq == other.seq
    }
}

impl Eq for PendingFrameCallback {}

impl PartialOrd for PendingFrameCallback {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for PendingFrameCallback {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        // Reverse ordering for min-heap behavior via Reverse<PendingFrameCallback>.
        self.due
            .cmp(&other.due)
            .then_with(|| self.seq.cmp(&other.seq))
    }
}

#[derive(Clone, Copy, Debug)]
struct AgentHandleData {
    pid: u32,
}

#[derive(Debug)]
pub struct LatestDmabuf {
    pub dmabuf: Dmabuf,
    pub acquire_fence: Option<OwnedFd>,
    pub buffer: SurfaceBuffer,
}

pub struct EyeCompositor {
    compositor_state: CompositorState,
    xwayland_shell_state: XWaylandShellState,
    xdg_shell_state: XdgShellState,
    seat_state: SeatState<Self>,
    #[allow(dead_code)]
    seat: Seat<Self>,
    shm_state: ShmState,
    dmabuf_state: DmabufState,
    drm_syncobj_state: Option<DrmSyncobjState>,
    #[allow(dead_code)]
    dmabuf_global: DmabufGlobal,
    #[allow(dead_code)]
    metabonk_control_global: smithay::reexports::wayland_server::backend::GlobalId,
    #[allow(dead_code)]
    wl_drm_global: Option<smithay::reexports::wayland_server::backend::GlobalId>,
    metabonk_orchestrators: Vec<ZmetabonkOrchestratorV1>,
    metabonk_agent_handles: HashMap<u32, Vec<ZmetabonkAgentV1>>,
    agents: HashMap<u32, AgentState>,
    featured_pid: Option<u32>,
    pending_frame_callbacks: BinaryHeap<Reverse<PendingFrameCallback>>,
    pending_frame_seq: u64,
    clock: Clock<Monotonic>,
    xwms: HashMap<XwmId, X11Wm>,
    output_w: i32,
    output_h: i32,
    output: Output,
    x11_windows: Vec<X11Surface>,
    export_period: Arc<Mutex<Duration>>,
    export_size: Arc<Mutex<(u32, u32)>>,
    latest_dmabuf: Arc<Mutex<Option<LatestDmabuf>>>,
    primary_surface: Option<ObjectId>,
    primary_surface_size: Option<(u32, u32)>,
    primary_surface_score: u64,
    forced_wl_surface: Option<u32>,
    force_focus: bool,
    last_forced_focus_at: Instant,
    wl_surface_to_x11: HashMap<u32, X11Surface>,
    xwayland_ready_at: Option<Instant>,
    last_no_buffer_log_at: Instant,
    last_non_dmabuf_log_at: Instant,
    last_commit_log_at: Instant,
    saw_any_buffer: bool,
    saw_any_dmabuf: bool,
    commit_calls: u64,
}

impl EyeCompositor {
    fn new(
        dh: &DisplayHandle,
        output_w: i32,
        output_h: i32,
        output: Output,
        export_period: Arc<Mutex<Duration>>,
        export_size: Arc<Mutex<(u32, u32)>>,
        latest_dmabuf: Arc<Mutex<Option<LatestDmabuf>>>,
        dmabuf_state: DmabufState,
        drm_syncobj_state: Option<DrmSyncobjState>,
        dmabuf_global: DmabufGlobal,
        wl_drm_global_data: Option<WlDrmGlobalData>,
    ) -> Self {
        let compositor_state = CompositorState::new::<Self>(dh);
        let xwayland_shell_state = XWaylandShellState::new::<Self>(dh);
        let xdg_shell_state = XdgShellState::new::<Self>(dh);
        let mut seat_state = SeatState::new();
        let mut seat = seat_state.new_wl_seat(dh, "seat-0");
        let _ = seat.add_pointer();
        let _ = seat.add_keyboard(smithay::input::keyboard::XkbConfig::default(), 200, 25);
        let force_focus = std::env::var("METABONK_EYE_FORCE_FOCUS")
            .ok()
            .map(|v| {
                let v = v.trim().to_ascii_lowercase();
                v == "1" || v == "true" || v == "yes" || v == "on"
            })
            .unwrap_or(false);
        if force_focus {
            info!("force focus enabled: keyboard focus will follow the primary capture surface");
        }
        // wl_shm is required by many clients (including XWayland) even if we refuse
        // SHM buffers for the vision path. We expose it for compatibility and enforce
        // GPU-only capture by ignoring SHM commits in `commit()`.
        let shm_state = ShmState::new::<Self>(dh, vec![
            wl_shm::Format::Argb8888,
            wl_shm::Format::Xrgb8888,
        ]);
        let metabonk_control_global = dh.create_global::<Self, ZmetabonkOrchestratorV1, _>(1, ());
        let wl_drm_global = wl_drm_global_data.map(|data| dh.create_global::<Self, WlDrm, _>(2, data));
        Self {
            compositor_state,
            xwayland_shell_state,
            xdg_shell_state,
            seat_state,
            seat,
            shm_state,
            dmabuf_state,
            drm_syncobj_state,
            dmabuf_global,
            metabonk_control_global,
            wl_drm_global,
            metabonk_orchestrators: Vec::new(),
            metabonk_agent_handles: HashMap::new(),
            agents: HashMap::new(),
            featured_pid: None,
            pending_frame_callbacks: BinaryHeap::new(),
            pending_frame_seq: 0,
            clock: Clock::<Monotonic>::new(),
            xwms: HashMap::new(),
            output_w,
            output_h,
            output,
            x11_windows: Vec::new(),
            export_period,
            export_size,
            latest_dmabuf,
            primary_surface: None,
            primary_surface_size: None,
            primary_surface_score: 0,
            forced_wl_surface: std::env::var("METABONK_EYE_FORCE_WL_SURFACE")
                .ok()
                .and_then(|v| v.parse::<u32>().ok()),
            force_focus,
            last_forced_focus_at: Instant::now().checked_sub(Duration::from_secs(3600)).unwrap_or_else(Instant::now),
            wl_surface_to_x11: HashMap::new(),
            xwayland_ready_at: None,
            last_no_buffer_log_at: Instant::now(),
            last_non_dmabuf_log_at: Instant::now(),
            last_commit_log_at: Instant::now(),
            saw_any_buffer: false,
            saw_any_dmabuf: false,
            commit_calls: 0,
        }
    }

    fn fallback_geometry(&self) -> (i32, i32) {
        (self.output_w, self.output_h)
    }

    fn ensure_agent(&mut self, pid: u32) -> &mut AgentState {
        let fallback = self.fallback_geometry();
        self.agents
            .entry(pid)
            .or_insert_with(|| AgentState::new(pid, fallback))
    }

    fn x11_surface_score(&self, wl_surface: u32) -> (u64, Option<(u32, u32, u32, String, String)>) {
        let Some(win) = self.wl_surface_to_x11.get(&wl_surface) else {
            return (0, None);
        };
        if !win.alive() {
            return (0, None);
        }
        if win.is_override_redirect() {
            return (0, Some((win.window_id(), win.pid().unwrap_or(0), 0, win.title(), win.class())));
        }
        let type_score: u64 = match win.window_type() {
            Some(smithay::xwayland::xwm::WmWindowType::Normal) => 100,
            Some(smithay::xwayland::xwm::WmWindowType::Dialog) => 80,
            Some(smithay::xwayland::xwm::WmWindowType::Splash) => 60,
            Some(smithay::xwayland::xwm::WmWindowType::Utility) => 40,
            Some(smithay::xwayland::xwm::WmWindowType::Toolbar) => 20,
            Some(smithay::xwayland::xwm::WmWindowType::Menu) => 10,
            Some(smithay::xwayland::xwm::WmWindowType::PopupMenu) => 10,
            Some(smithay::xwayland::xwm::WmWindowType::DropdownMenu) => 10,
            Some(smithay::xwayland::xwm::WmWindowType::Tooltip) => 1,
            Some(smithay::xwayland::xwm::WmWindowType::Notification) => 1,
            None => 50,
        };
        let has_pid = win.pid().is_some();
        let title = win.title();
        let class = win.class();
        let has_title = !title.trim().is_empty();
        let has_class = !class.trim().is_empty();
        let bonus = (has_pid as u64) * 50 + (has_title as u64) * 10 + (has_class as u64) * 10;
        let score = type_score.saturating_mul(1_000).saturating_add(bonus);
        (
            score,
            Some((win.window_id(), win.pid().unwrap_or(0), type_score as u32, title, class)),
        )
    }

    fn pid_for_client(client: &Client) -> u32 {
        if let Some(cs) = client.get_data::<ClientState>() {
            if cs.pid != 0 {
                return cs.pid;
            }
        }
        if let Some(x) = client.get_data::<smithay::xwayland::XWaylandClientData>() {
            if let Some(pid) = x.user_data().get::<XWaylandPid>().map(|p| p.0) {
                return pid;
            }
        }
        0
    }

    fn broadcast_agent_detected(&mut self, pid: u32) {
        let app_id = self
            .agents
            .get(&pid)
            .map(|a| a.app_id.clone())
            .unwrap_or_else(|| "unknown_app".into());
        self.metabonk_orchestrators.retain(|o| o.is_alive());
        for o in &self.metabonk_orchestrators {
            o.agent_detected(pid, app_id.clone());
        }
    }

    fn apply_agent_geometry(&mut self, pid: u32) {
        let Some(agent) = self.agents.get_mut(&pid) else {
            return;
        };
        let Some(toplevel) = agent.toplevel.as_ref() else {
            return;
        };
        if !toplevel.alive() {
            agent.toplevel = None;
            return;
        }

        let (w, h) = agent.geometry;
        toplevel.with_pending_state(|state| {
            state.size = Some((w, h).into());
        });
        let _ = toplevel.send_pending_configure();
    }

    fn apply_xwayland_geometry(&mut self, size: (i32, i32)) {
        let (w, h) = size;
        if w <= 0 || h <= 0 {
            return;
        }
        if self.output_w == w && self.output_h == h {
            return;
        }
        self.output_w = w;
        self.output_h = h;
        let mode = Mode {
            size: (w, h).into(),
            refresh: 60_000,
        };
        self.output.change_current_state(
            Some(mode),
            Some(Transform::Normal),
            Some(Scale::Integer(1)),
            Some((0, 0).into()),
        );
        self.output.set_preferred(mode);
        if let Ok(mut g) = self.export_size.lock() {
            *g = (w as u32, h as u32);
        }

        // Force all mapped X11 windows fullscreen to the new output size.
        self.x11_windows.retain(|w| w.alive());
        let rect = Rectangle::<i32, Logical>::from_loc_and_size((0, 0), (w, h));
        for win in &self.x11_windows {
            let _ = win.configure(Some(rect));
        }
    }

    fn set_agent_tier(&mut self, pid: u32, tier: AgentTier) {
        let fallback = self.fallback_geometry();
        let (target_interval, geometry) = {
            let agent = self.ensure_agent(pid);
            agent.tier = tier;
            agent.target_interval = tier.target_interval();
            agent.geometry = tier.default_geometry(fallback);
            (agent.target_interval, agent.geometry)
        };
        if tier == AgentTier::Featured {
            self.featured_pid = Some(pid);
        }
        if let Ok(mut g) = self.export_period.lock() {
            *g = target_interval;
        }
        self.apply_agent_geometry(pid);
        // If there is no XDG toplevel (XWayland), treat geometry as the output size.
        self.apply_xwayland_geometry(geometry);
    }

    fn force_agent_geometry(&mut self, pid: u32, width: u32, height: u32) {
        let geometry = {
            let agent = self.ensure_agent(pid);
            agent.geometry = (width as i32, height as i32);
            agent.geometry
        };
        self.apply_agent_geometry(pid);
        self.apply_xwayland_geometry(geometry);
    }

    fn queue_frame_callbacks_for_surface(&mut self, surface: &WlSurface, pid: u32) {
        let mut callbacks: Vec<wl_callback::WlCallback> = Vec::new();
        with_states(surface, |states| {
            callbacks.extend(
                states
                    .cached_state
                    .get::<SurfaceAttributes>()
                    .current()
                    .frame_callbacks
                    .drain(..),
            );
        });

        if callbacks.is_empty() {
            return;
        }

        let agent = self.ensure_agent(pid);
        let now = Instant::now();
        let elapsed = now.duration_since(agent.last_frame_time);
        let target = agent.target_interval;
        let due = if agent.tier == AgentTier::Featured || elapsed >= target {
            now
        } else {
            now + (target - elapsed)
        };

        for callback in callbacks {
            self.pending_frame_seq = self.pending_frame_seq.wrapping_add(1);
            self.pending_frame_callbacks.push(Reverse(PendingFrameCallback {
                due,
                seq: self.pending_frame_seq,
                pid,
                callback,
            }));
        }
    }

    fn drain_due_frame_callbacks(&mut self) {
        let now = Instant::now();
        let time = self.clock.now();
        let time_dur: Duration = time.into();
        let time_ms = time_dur.as_millis() as u32;

        while let Some(Reverse(head)) = self.pending_frame_callbacks.peek() {
            if head.due > now {
                break;
            }

            let Reverse(entry) = self.pending_frame_callbacks.pop().unwrap();
            entry.callback.done(time_ms);

            if let Some(agent) = self.agents.get_mut(&entry.pid) {
                agent.last_frame_time = now;
                // Minimal telemetry: report effective callback FPS. GPU time is unknown here.
                let denom_ms = agent.target_interval.as_millis().max(1) as u64;
                let fps = (1000 / denom_ms).max(1) as u32;
                if let Some(handles) = self.metabonk_agent_handles.get(&entry.pid) {
                    for h in handles.iter().filter(|h| h.is_alive()) {
                        h.telemetry(fps, 0);
                    }
                }
            }
        }
    }
}

impl BufferHandler for EyeCompositor {
    fn buffer_destroyed(&mut self, _buffer: &wl_buffer::WlBuffer) {
        // No-op: we only keep the latest Dmabuf handle (Arc) and let it drop naturally.
    }
}

impl DmabufHandler for EyeCompositor {
    fn dmabuf_state(&mut self) -> &mut DmabufState {
        &mut self.dmabuf_state
    }

    fn dmabuf_imported(&mut self, _global: &DmabufGlobal, _dmabuf: Dmabuf, notifier: ImportNotifier) {
        let log_imports = std::env::var("METABONK_EYE_LOG_DMABUF_IMPORTS")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        if log_imports {
            let fmt = _dmabuf.format().code;
            let modifier: u64 = _dmabuf.format().modifier.into();
            let stride0 = _dmabuf.strides().next().unwrap_or(0);
            let offset0 = _dmabuf.offsets().next().unwrap_or(0);
            let planes = _dmabuf.num_planes();
            info!(
                ?fmt,
                modifier = format_args!("0x{modifier:016x}"),
                planes,
                stride0,
                offset0,
                w = _dmabuf.width(),
                h = _dmabuf.height(),
                "linux-dmabuf imported"
            );
        }
        // Accept all dmabufs from clients. We enforce "GPU-only" at the consumer by
        // ignoring SHM buffers and requiring a dmabuf-backed source to appear.
        let _ = notifier.successful::<Self>();
    }
}

impl DrmSyncobjHandler for EyeCompositor {
    fn drm_syncobj_state(&mut self) -> Option<&mut DrmSyncobjState> {
        self.drm_syncobj_state.as_mut()
    }
}

impl CompositorHandler for EyeCompositor {
    fn compositor_state(&mut self) -> &mut CompositorState {
        &mut self.compositor_state
    }

    fn client_compositor_state<'a>(&self, client: &'a Client) -> &'a CompositorClientState {
        // XWayland inserts its own XWaylandClientData containing a compositor_state; normal clients use ClientState.
        if let Some(x) = client.get_data::<smithay::xwayland::XWaylandClientData>() {
            return &x.compositor_state;
        }
        &client.get_data::<ClientState>().unwrap().compositor_state
    }

    fn commit(&mut self, surface: &WlSurface) {
        self.commit_calls = self.commit_calls.saturating_add(1);
        let now = Instant::now();
        if now.duration_since(self.last_commit_log_at) > Duration::from_secs(5) {
            self.last_commit_log_at = now;
            info!(commit_calls = self.commit_calls, "wl_surface commits observed");
        }

        // This compositor has a single output; treat every client surface as entering it.
        // Some clients (notably XWayland/glamor) expect wl_surface.enter events to decide
        // how/where to present.
        self.output.enter(surface);

        // Frame callback throttling: drain callbacks requested for this commit and schedule them based on agent tier.
        if let Some(client) = surface.client() {
            let pid = Self::pid_for_client(&client);
            if pid != 0 {
                self.queue_frame_callbacks_for_surface(surface, pid);
            }
        }

        // Let Smithay handle buffer management for this surface first.
        // This populates RendererSurfaceState with the latest committed wl_buffer.
        on_commit_buffer_handler::<Self>(surface);

        // Capture from a single primary surface to avoid thrashing between multiple clients/surfaces
        // (cursor, popups, etc.). We select the primary only once we observe a DMA-BUF-backed commit
        // (GPU-only), which avoids accidentally pinning a SHM-only surface as "primary".
        with_states(surface, |data| {
            let rs = match data.data_map.get::<RendererSurfaceStateUserData>() {
                Some(s) => s,
                None => return,
            };
            let buf = match rs.lock().ok().and_then(|s| s.buffer().cloned()) {
                Some(b) => b,
                None => {
                    if now.duration_since(self.last_no_buffer_log_at) > Duration::from_secs(2) {
                        self.last_no_buffer_log_at = now;
                        let pid = surface
                            .client()
                            .map(|c| Self::pid_for_client(&c))
                            .unwrap_or(0);
                        info!(pid, "wl_surface commit without buffer (no attach)");
                    }
                    return;
                }
            };
            if !self.saw_any_buffer {
                self.saw_any_buffer = true;
                info!("first wl_buffer observed from a client surface");
            }

            if let Ok(dmabuf) = get_dmabuf(&*buf) {
                if !self.saw_any_dmabuf {
                    self.saw_any_dmabuf = true;
                    info!("first DMA-BUF wl_buffer observed from a client surface");
                }

                let sz = dmabuf.size();
                let pid = surface
                    .client()
                    .map(|c| Self::pid_for_client(&c))
                    .unwrap_or(0);

                if now.duration_since(self.last_commit_log_at) > Duration::from_secs(2) {
                    self.last_commit_log_at = now;
                    let stride0 = dmabuf.strides().next().unwrap_or(0);
                    let offset0 = dmabuf.offsets().next().unwrap_or(0);
                    let modifier_u64: u64 = dmabuf.format().modifier.into();
                    let fmt = dmabuf.format().code;
                    let wl_surface = surface.id().protocol_id();
                    info!(
                        pid,
                        wl_surface,
                        w = sz.w,
                        h = sz.h,
                        stride0,
                        offset0,
                        modifier = format_args!("0x{modifier_u64:016x}"),
                        format = format_args!("{fmt:?}"),
                        "dmabuf commit"
                    );
                }

                // Record the latest surface seen for this pid (for featured selection).
                if let Some(client) = surface.client() {
                    let pid = Self::pid_for_client(&client);
                    if pid != 0 {
                        let agent = self.ensure_agent(pid);
                        agent.surface = Some(surface.clone());
                    }
                }

                // If a PID is explicitly featured, ignore DMA-BUF commits from other clients.
                if let Some(featured) = self.featured_pid {
                    if pid != featured {
                        return;
                    }
                }

                // Games (and XWayland) often create tiny helper surfaces before the real viewport.
                // Ignore them so we don't latch onto a 1x1/tooltip/cursor buffer.
                if sz.w < 64 || sz.h < 64 {
                    return;
                }

                // Select capture surface heuristically:
                // - Prefer the largest DMA-BUF surface (area).
                // - If area ties, prefer the most recently created surface (higher protocol ID).
                // This avoids latching onto early splash/launcher windows and prevents thrashing
                // when multiple full-size surfaces commit alternately (e.g., overlays).
                let candidate_id = surface.id();
                let candidate_wl_surface = candidate_id.protocol_id();
                if let Some(forced) = self.forced_wl_surface {
                    if candidate_wl_surface != forced {
                        return;
                    }
                }
                let (cw, ch) = match (u32::try_from(sz.w), u32::try_from(sz.h)) {
                    (Ok(w), Ok(h)) => (w, h),
                    _ => return,
                };
                let candidate_area = u64::from(cw).saturating_mul(u64::from(ch));
                let (x11_score, x11_meta) = self.x11_surface_score(candidate_wl_surface);
                let candidate_score = candidate_area
                    .saturating_mul(10_000)
                    .saturating_add(x11_score.saturating_mul(100))
                    .saturating_add(u64::from(candidate_wl_surface));
                let should_select = self.primary_surface.is_none() || candidate_score > self.primary_surface_score;
                if should_select {
                    let changed = self.primary_surface.as_ref() != Some(&candidate_id);
                    if changed {
                        if let Some((x11_window, x11_pid, x11_type_score, title, class)) = x11_meta {
                            info!(
                                pid,
                                surface_id = ?candidate_id,
                                wl_surface = candidate_wl_surface,
                                w = sz.w,
                                h = sz.h,
                                x11_window,
                                x11_pid,
                                x11_type_score,
                                title = title.as_str(),
                                class = class.as_str(),
                                "switching primary surface capture"
                            );
                        } else {
                            info!(
                                pid,
                                surface_id = ?candidate_id,
                                wl_surface = candidate_wl_surface,
                                w = sz.w,
                                h = sz.h,
                                "switching primary surface capture"
                            );
                        }
                    }
                    if self.force_focus {
                        // Games can throttle/freeze rendering when unfocused. Continuously asserting focus
                        // keeps the capture surface "active" even as new helper windows appear.
                        let should_refocus = changed || now.duration_since(self.last_forced_focus_at) > Duration::from_secs(1);
                        if should_refocus {
                            if let Some(kbd) = self.seat.get_keyboard() {
                                let focus = KeyboardFocusTarget::from_wl_surface(
                                    surface,
                                    self.wl_surface_to_x11.get(&candidate_wl_surface),
                                );
                                kbd.set_focus(
                                    self,
                                    Some(focus),
                                    smithay::utils::SERIAL_COUNTER.next_serial(),
                                );
                                info!(
                                    pid,
                                    wl_surface = candidate_wl_surface,
                                    "forcing keyboard focus to capture surface"
                                );
                                self.last_forced_focus_at = now;
                            }
                        }
                    }
                    self.primary_surface = Some(candidate_id);
                    self.primary_surface_size = Some((cw, ch));
                    self.primary_surface_score = candidate_score;
                }
                if let Some(primary) = self.primary_surface.as_ref() {
                    if &surface.id() != primary {
                        return;
                    }
                }

                // Keep asserting focus on the selected capture surface. This is important for
                // many games that throttle/freeze rendering when unfocused, and can lose focus
                // due to transient/helper windows inside XWayland.
                if self.force_focus {
                    let now = Instant::now();
                    if now.duration_since(self.last_forced_focus_at) > Duration::from_secs(1) {
                        if let Some(kbd) = self.seat.get_keyboard() {
                            let wl_surface = surface.id().protocol_id();
                            let focus = KeyboardFocusTarget::from_wl_surface(
                                surface,
                                self.wl_surface_to_x11.get(&wl_surface),
                            );
                            kbd.set_focus(self, Some(focus), smithay::utils::SERIAL_COUNTER.next_serial());
                            info!(pid, wl_surface, "reasserting keyboard focus to capture surface");
                            self.last_forced_focus_at = now;
                        }
                    }
                }
                let mut acquire_fd: Option<OwnedFd> = None;
                if self.drm_syncobj_state.is_some() {
                    let mut guard = data.cached_state.get::<DrmSyncobjCachedState>();
                    let sync_state = guard.current();
                    let has_acquire = sync_state.acquire_point.is_some();
                    let has_release = sync_state.release_point.is_some();
                    info!(
                        pid,
                        surface = ?surface.id(),
                        acquire = has_acquire,
                        release = has_release,
                        "drm syncobj points on commit"
                    );
                    if !has_acquire {
                        warn!(pid, surface = ?surface.id(), "drm syncobj commit missing acquire point");
                    }
                    if !has_release {
                        warn!(pid, surface = ?surface.id(), "drm syncobj commit missing release point");
                    }
                    if let Some(point) = sync_state.acquire_point.as_ref() {
                        match point.export_sync_file() {
                            Ok(fd) => {
                                acquire_fd = Some(fd);
                            }
                            Err(err) => {
                                warn!("failed to export acquire sync_file fd: {err}");
                            }
                        }
                    }
                } else {
                    let pid = surface
                        .client()
                        .map(|c| Self::pid_for_client(&c))
                        .unwrap_or(0);
                    warn!(pid, surface = ?surface.id(), "drm syncobj global disabled; no explicit sync points");
                }

                if acquire_fd.is_none() {
                    let pid = surface
                        .client()
                        .map(|c| Self::pid_for_client(&c))
                        .unwrap_or(0);
                    match export_dmabuf_read_sync_file(&dmabuf) {
                        Ok(Some(fd)) => {
                            info!(
                                pid,
                                surface = ?surface.id(),
                                fence_fd = fd.as_raw_fd(),
                                "exported implicit dmabuf sync_file fence (read)"
                            );
                            acquire_fd = Some(fd);
                        }
                        Ok(None) => {
                            info!(pid, surface = ?surface.id(), "no implicit dmabuf sync_file fence exported");
                        }
                        Err(err) => {
                            warn!(pid, surface = ?surface.id(), "failed to export implicit dmabuf sync_file fence: {err}");
                        }
                    }
                }

                let mut g = self.latest_dmabuf.lock().unwrap();
                *g = Some(LatestDmabuf {
                    dmabuf: dmabuf.clone(),
                    acquire_fence: acquire_fd,
                    buffer: buf.clone(),
                });
            } else {
                // Ignore SHM buffers to preserve the GPU-only contract. Do NOT clear latest_dmabuf here:
                // some clients briefly commit SHM surfaces (cursor/overlays) even while the main window
                // continues to present DMA-BUF frames.
                if now.duration_since(self.last_non_dmabuf_log_at) > Duration::from_secs(2) {
                    self.last_non_dmabuf_log_at = now;
                    let pid = surface
                        .client()
                        .map(|c| Self::pid_for_client(&c))
                        .unwrap_or(0);
                    let mut logged = false;
                    if let Ok(_) = shm::with_buffer_contents(&*buf, |_, len, data| {
                        info!(
                            pid,
                            w = data.width,
                            h = data.height,
                            stride = data.stride,
                            bytes = len,
                            format = format_args!("{:?}", data.format),
                            "shm commit"
                        );
                        logged = true;
                    }) {
                        // logged flag updated inside closure
                    }
                    if !logged {
                        warn!(pid, "non-dmabuf wl_buffer commit (unknown buffer type)");
                    }
                }
            }
        });
    }
}

impl XWaylandShellHandler for EyeCompositor {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.xwayland_shell_state
    }

    fn surface_associated(&mut self, _xwm_id: XwmId, surface: WlSurface, window: X11Surface) {
        let wl_surface = surface.id().protocol_id();
        let window_id = window.window_id();
        let x11_pid = window.pid().unwrap_or(0);
        let title = window.title();
        let class = window.class();
        let window_type = window.window_type();
        let override_redirect = window.is_override_redirect();
        info!(
            wl_surface,
            window_id,
            x11_pid,
            ?window_type,
            override_redirect,
            title = title.as_str(),
            class = class.as_str(),
            "xwayland window associated with wl_surface"
        );
        self.wl_surface_to_x11.insert(wl_surface, window);
    }
}

impl XwmHandler for EyeCompositor {
    fn xwm_state(&mut self, xwm: XwmId) -> &mut X11Wm {
        self.xwms.get_mut(&xwm).expect("unknown XWM id")
    }

    fn new_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn new_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn map_window_request(&mut self, xwm: XwmId, window: X11Surface) {
        // Minimal "WM": allow windows to appear by mapping them and forcing them fullscreen.
        let _ = window.set_mapped(true);
        let rect = Rectangle::<i32, Logical>::from_loc_and_size((0, 0), (self.output_w, self.output_h));
        let _ = window.configure(Some(rect));
        self.x11_windows.push(window.clone());
        // Keep stacking order sane.
        let _ = self.xwm_state(xwm).raise_window(&window);
    }

    fn mapped_override_redirect_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn unmapped_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn destroyed_window(&mut self, _xwm: XwmId, _window: X11Surface) {}

    fn configure_request(
        &mut self,
        _xwm: XwmId,
        window: X11Surface,
        _x: Option<i32>,
        _y: Option<i32>,
        _w: Option<u32>,
        _h: Option<u32>,
        _reorder: Option<Reorder>,
    ) {
        let rect = Rectangle::<i32, Logical>::from_loc_and_size((0, 0), (self.output_w, self.output_h));
        let _ = window.configure(Some(rect));
    }

    fn configure_notify(
        &mut self,
        _xwm: XwmId,
        _window: X11Surface,
        _geometry: Rectangle<i32, Logical>,
        _above: Option<smithay::xwayland::xwm::X11Window>,
    ) {
    }

    fn property_notify(&mut self, _xwm: XwmId, _window: X11Surface, _property: WmWindowProperty) {}

    fn resize_request(&mut self, _xwm: XwmId, _window: X11Surface, _button: u32, _resize_edge: ResizeEdge) {}

    fn move_request(&mut self, _xwm: XwmId, _window: X11Surface, _button: u32) {}
}

impl ShmHandler for EyeCompositor {
    fn shm_state(&self) -> &ShmState {
        &self.shm_state
    }
}

impl SeatHandler for EyeCompositor {
    type KeyboardFocus = KeyboardFocusTarget;
    type PointerFocus = WlSurface;
    type TouchFocus = WlSurface;

    fn seat_state(&mut self) -> &mut SeatState<Self> {
        &mut self.seat_state
    }

    fn focus_changed(&mut self, _seat: &Seat<Self>, _focused: Option<&Self::KeyboardFocus>) {}

    fn cursor_image(&mut self, _seat: &Seat<Self>, _image: CursorImageStatus) {}
}

impl XdgShellHandler for EyeCompositor {
    fn xdg_shell_state(&mut self) -> &mut XdgShellState {
        &mut self.xdg_shell_state
    }

    fn new_toplevel(&mut self, surface: ToplevelSurface) {
        // Minimal toplevel handler: immediately configure the surface to our preferred geometry.
        let pid = surface
            .wl_surface()
            .client()
            .map(|c| Self::pid_for_client(&c))
            .unwrap_or(0);
        if pid != 0 {
            let agent = self.ensure_agent(pid);
            agent.toplevel = Some(surface.clone());
            self.apply_agent_geometry(pid);
        }
        let _ = surface.send_pending_configure();
    }

    fn new_popup(&mut self, _surface: PopupSurface, _positioner: PositionerState) {}

    fn grab(
        &mut self,
        _surface: PopupSurface,
        _seat: smithay::reexports::wayland_server::protocol::wl_seat::WlSeat,
        _serial: smithay::utils::Serial,
    ) {
    }

    fn reposition_request(&mut self, surface: PopupSurface, _positioner: PositionerState, token: u32) {
        surface.send_repositioned(token);
    }

    fn app_id_changed(&mut self, surface: ToplevelSurface) {
        let pid = surface
            .wl_surface()
            .client()
            .map(|c| Self::pid_for_client(&c))
            .unwrap_or(0);
        if pid == 0 {
            return;
        }
        let app_id = with_states(surface.wl_surface(), |states| {
            states
                .data_map
                .get::<XdgToplevelSurfaceData>()
                .and_then(|m| m.lock().ok().and_then(|g| g.app_id.clone()))
        })
        .unwrap_or_else(|| "unknown_app".into());

        let agent = self.ensure_agent(pid);
        agent.app_id = app_id;
        self.broadcast_agent_detected(pid);
    }
}

impl OutputHandler for EyeCompositor {}

delegate_compositor!(EyeCompositor);
delegate_seat!(EyeCompositor);
smithay::delegate_shm!(EyeCompositor);
smithay::delegate_dmabuf!(EyeCompositor);
smithay::delegate_drm_syncobj!(EyeCompositor);
smithay::delegate_output!(EyeCompositor);
delegate_xwayland_shell!(EyeCompositor);
delegate_xdg_shell!(EyeCompositor);

impl smithay::reexports::wayland_server::GlobalDispatch<WlDrm, WlDrmGlobalData, EyeCompositor> for EyeCompositor {
    fn bind(
        _state: &mut EyeCompositor,
        _handle: &DisplayHandle,
        _client: &Client,
        resource: smithay::reexports::wayland_server::New<WlDrm>,
        global_data: &WlDrmGlobalData,
        data_init: &mut smithay::reexports::wayland_server::DataInit<'_, EyeCompositor>,
    ) {
        let res = data_init.init(
            resource,
            WlDrmData {
                device_path: global_data.device_path.clone(),
                main_dev: global_data.main_dev,
            },
        );
        res.device(global_data.device_path.clone());
        for fmt in &global_data.formats {
            res.format(*fmt);
        }
        res.capabilities(global_data.capabilities);
    }
}

impl smithay::reexports::wayland_server::Dispatch<WlDrm, WlDrmData, EyeCompositor> for EyeCompositor {
    fn request(
        _state: &mut EyeCompositor,
        _client: &Client,
        resource: &WlDrm,
        request: wl_drm::Request,
        data: &WlDrmData,
        _dh: &DisplayHandle,
        data_init: &mut smithay::reexports::wayland_server::DataInit<'_, EyeCompositor>,
    ) {
        let disable_prime = std::env::var("METABONK_WL_DRM_DISABLE_PRIME")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        match request {
            wl_drm::Request::Authenticate { .. } => {
                resource.authenticated();
            }
            wl_drm::Request::CreateBuffer { .. } | wl_drm::Request::CreatePlanarBuffer { .. } => {
                resource.post_error(
                    wl_drm::Error::InvalidName,
                    "wl_drm name-based buffers are unsupported; use prime buffers or linux-dmabuf",
                );
            }
            wl_drm::Request::CreatePrimeBuffer {
                id,
                name,
                width,
                height,
                format,
                offset0,
                stride0,
                offset1,
                stride1,
                offset2,
                stride2,
            } => {
                use std::os::fd::{AsRawFd, FromRawFd, OwnedFd};
                use smithay::backend::allocator::{Fourcc, Modifier};
                use smithay::backend::drm::DrmNode;

                if disable_prime {
                    resource.post_error(
                        wl_drm::Error::InvalidName,
                        "wl_drm PRIME buffers disabled (METABONK_WL_DRM_DISABLE_PRIME=1); use linux-dmabuf",
                    );
                    return;
                }

                if width <= 0 || height <= 0 {
                    resource.post_error(wl_drm::Error::InvalidName, "invalid buffer dimensions");
                    return;
                }

                let fourcc = match Fourcc::try_from(format) {
                    Ok(f) => f,
                    Err(_) => {
                        resource.post_error(wl_drm::Error::InvalidFormat, "unrecognized drm fourcc");
                        return;
                    }
                };

                // Diagnostic: log PRIME buffer parameters so we can distinguish wl_drm PRIME vs linux-dmabuf flows.
                if std::env::var("METABONK_EYE_LOG_DMABUF_IMPORTS").ok().map(|v| v.trim() == "1").unwrap_or(false) {
                    info!(
                        ?fourcc,
                        w = width,
                        h = height,
                        stride0,
                        offset0,
                        stride1,
                        offset1,
                        stride2,
                        offset2,
                        "wl_drm CreatePrimeBuffer"
                    );
                }

                let raw_fd = name.as_raw_fd();
                let mut builder = Dmabuf::builder((width, height), fourcc, Modifier::Linear, DmabufFlags::empty());
                if let Ok(node) = DrmNode::from_dev_id(data.main_dev as _) {
                    builder.set_node(node);
                }

                let plane0_offset = match u32::try_from(offset0) {
                    Ok(v) => v,
                    Err(_) => {
                        resource.post_error(wl_drm::Error::InvalidName, "invalid plane0 offset");
                        return;
                    }
                };
                let plane0_stride = match u32::try_from(stride0) {
                    Ok(v) => v,
                    Err(_) => {
                        resource.post_error(wl_drm::Error::InvalidName, "invalid plane0 stride");
                        return;
                    }
                };
                if plane0_stride == 0 {
                    resource.post_error(wl_drm::Error::InvalidName, "invalid plane0 stride");
                    return;
                }

                if !builder.add_plane(name, 0, plane0_offset, plane0_stride) {
                    resource.post_error(wl_drm::Error::InvalidName, "too many dmabuf planes");
                    return;
                }

                let extra_planes = [(1u32, offset1, stride1), (2u32, offset2, stride2)];
                for (idx, offset, stride) in extra_planes {
                    if stride == 0 {
                        continue;
                    }
                    let off = match u32::try_from(offset) {
                        Ok(v) => v,
                        Err(_) => {
                            resource.post_error(wl_drm::Error::InvalidName, "invalid plane offset");
                            return;
                        }
                    };
                    let strd = match u32::try_from(stride) {
                        Ok(v) => v,
                        Err(_) => {
                            resource.post_error(wl_drm::Error::InvalidName, "invalid plane stride");
                            return;
                        }
                    };
                    let dup_fd = unsafe { libc::dup(raw_fd) };
                    if dup_fd < 0 {
                        resource.post_error(wl_drm::Error::InvalidName, "dup dmabuf fd failed");
                        return;
                    }
                    let owned = unsafe { OwnedFd::from_raw_fd(dup_fd) };
                    if !builder.add_plane(owned, idx, off, strd) {
                        resource.post_error(wl_drm::Error::InvalidName, "too many dmabuf planes");
                        return;
                    }
                }

                let Some(dmabuf) = builder.build() else {
                    resource.post_error(wl_drm::Error::InvalidName, "failed to build dmabuf");
                    return;
                };

                let _buf = data_init.init(id, dmabuf);
            }
        }
    }
}

impl smithay::reexports::wayland_server::GlobalDispatch<ZmetabonkOrchestratorV1, (), EyeCompositor> for EyeCompositor {
    fn bind(
        state: &mut EyeCompositor,
        _handle: &DisplayHandle,
        client: &Client,
        resource: smithay::reexports::wayland_server::New<ZmetabonkOrchestratorV1>,
        _global_data: &(),
        data_init: &mut smithay::reexports::wayland_server::DataInit<'_, EyeCompositor>,
    ) {
        let creds = client.get_credentials(_handle).ok();
        if let Some(creds) = creds {
            info!(pid = creds.pid, uid = creds.uid, gid = creds.gid, "metabonk orchestrator bound");
        }

        let res = data_init.init(resource, ());
        state.metabonk_orchestrators.push(res);
        // Avoid missed events: on bind, re-emit the current agent set.
        for pid in state.agents.keys().copied().collect::<Vec<_>>() {
            state.broadcast_agent_detected(pid);
        }
    }
}

impl smithay::reexports::wayland_server::Dispatch<ZmetabonkOrchestratorV1, (), EyeCompositor> for EyeCompositor {
    fn request(
        state: &mut EyeCompositor,
        _client: &Client,
        _orchestrator: &ZmetabonkOrchestratorV1,
        request: zmetabonk_orchestrator_v1::Request,
        _data: &(),
        _dh: &DisplayHandle,
        data_init: &mut smithay::reexports::wayland_server::DataInit<'_, EyeCompositor>,
    ) {
        match request {
            zmetabonk_orchestrator_v1::Request::GetAgentHandle { id, pid } => {
                let pid = pid as u32;
                let _agent = state.ensure_agent(pid);
                let agent_res: ZmetabonkAgentV1 = data_init.init(id, AgentHandleData { pid });
                state
                    .metabonk_agent_handles
                    .entry(pid)
                    .or_default()
                    .push(agent_res);
            }
        }
    }
}

impl smithay::reexports::wayland_server::Dispatch<ZmetabonkAgentV1, AgentHandleData, EyeCompositor> for EyeCompositor {
    fn request(
        state: &mut EyeCompositor,
        _client: &Client,
        _resource: &ZmetabonkAgentV1,
        request: zmetabonk_agent_v1::Request,
        data: &AgentHandleData,
        _dh: &DisplayHandle,
        _data_init: &mut smithay::reexports::wayland_server::DataInit<'_, EyeCompositor>,
    ) {
        match request {
            zmetabonk_agent_v1::Request::SetTier { tier } => {
                let tier = match tier {
                    smithay::reexports::wayland_server::WEnum::Value(zmetabonk_agent_v1::Tier::Featured) => {
                        AgentTier::Featured
                    }
                    smithay::reexports::wayland_server::WEnum::Value(zmetabonk_agent_v1::Tier::Observation) => {
                        AgentTier::Observation
                    }
                    smithay::reexports::wayland_server::WEnum::Value(zmetabonk_agent_v1::Tier::Background) => {
                        AgentTier::Background
                    }
                    smithay::reexports::wayland_server::WEnum::Value(zmetabonk_agent_v1::Tier::Suspended) => {
                        AgentTier::Suspended
                    }
                    smithay::reexports::wayland_server::WEnum::Unknown(_) => return,
                };
                state.set_agent_tier(data.pid, tier);
            }
            zmetabonk_agent_v1::Request::ForceGeometry { width, height } => {
                state.force_agent_geometry(data.pid, width, height);
            }
        }
    }
}

fn resolve_render_node_path(main_dev: libc::dev_t) -> anyhow::Result<PathBuf> {
    use std::os::unix::fs::MetadataExt;

    if let Ok(p) = std::env::var("METABONK_DRM_RENDER_NODE") {
        return Ok(PathBuf::from(p));
    }

    let entries = std::fs::read_dir("/dev/dri").context("read_dir(/dev/dri)")?;
    for ent in entries.flatten() {
        let name = ent.file_name().to_string_lossy().to_string();
        if !name.starts_with("renderD") {
            continue;
        }
        let Ok(md) = ent.metadata() else { continue };
        if md.rdev() as libc::dev_t == main_dev {
            return Ok(ent.path());
        }
    }

    anyhow::bail!(
        "could not resolve DRM render node for main_dev=0x{:x} (set METABONK_DRM_RENDER_NODE)",
        main_dev as u64
    )
}

fn resolve_card_node_path(main_dev: libc::dev_t) -> anyhow::Result<PathBuf> {
    if let Ok(p) = std::env::var("METABONK_DRM_CARD_NODE") {
        return Ok(PathBuf::from(p));
    }

    let render_node = resolve_render_node_path(main_dev)?;
    let base = render_node
        .file_name()
        .map(|s| s.to_string_lossy().to_string())
        .unwrap_or_else(|| "renderD?".into());
    let sys = std::path::Path::new("/sys/class/drm").join(&base).join("device/drm");
    let mut found: Option<PathBuf> = None;
    if let Ok(entries) = std::fs::read_dir(&sys) {
        for ent in entries.flatten() {
            let name = ent.file_name().to_string_lossy().to_string();
            if name.starts_with("card") {
                found = Some(PathBuf::from("/dev/dri").join(name));
                break;
            }
        }
    }

    Ok(found.unwrap_or_else(|| PathBuf::from("/dev/dri/card0")))
}

fn write_compositor_env(
    run_dir: &Path,
    wayland_display: &str,
    frame_sock: &str,
    display: &str,
) -> anyhow::Result<()> {
    let path = run_dir.join("compositor.env");
    let mut lines = Vec::new();
    lines.push(format!("XDG_RUNTIME_DIR={}", run_dir.display()));
    lines.push(format!("WAYLAND_DISPLAY={wayland_display}"));
    lines.push(format!("DISPLAY={display}"));
    lines.push(format!("METABONK_FRAME_SOCK={frame_sock}"));
    // Atomic-ish update: write then rename.
    let tmp = run_dir.join("compositor.env.tmp");
    fs::write(&tmp, lines.join("\n") + "\n").with_context(|| format!("write {tmp:?}"))?;
    fs::rename(&tmp, &path).with_context(|| format!("rename {tmp:?} -> {path:?}"))?;
    Ok(())
}

pub fn spawn_wayland_xwayland_thread(
    run_dir: PathBuf,
    wayland_display: String,
    frame_sock: String,
    output_w: u32,
    output_h: u32,
    main_dev: libc::dev_t,
    export_period: Arc<Mutex<Duration>>,
    export_size: Arc<Mutex<(u32, u32)>>,
    latest_dmabuf: Arc<Mutex<Option<LatestDmabuf>>>,
) -> std::thread::JoinHandle<anyhow::Result<()>> {
    std::thread::spawn(move || {
        // Assumes the parent has set XDG_RUNTIME_DIR to `run_dir` for this process.
        fs::create_dir_all(&run_dir).with_context(|| format!("create_dir_all({:?})", run_dir))?;

        let mut display: Display<EyeCompositor> = Display::new()?;
        let dh = display.handle();

        let mut event_loop: EventLoop<EyeCompositor> = EventLoop::try_new()?;
        let handle = event_loop.handle();

        // Advertise linux-dmabuf so XWayland clients can submit GPU buffers (no SHM fallback).
        let mut dmabuf_state = DmabufState::new();
        use smithay::backend::allocator::{Format, Fourcc, Modifier};
        let only_linear = std::env::var("METABONK_DMABUF_ONLY_LINEAR")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        if only_linear {
            warn!("dmabuf: METABONK_DMABUF_ONLY_LINEAR=1 set; advertising only linear buffers");
        }
        let mut preferred_modifiers: Vec<Modifier> = vec![
            Modifier::Nvidia_16bx2_block_one_gob,
            Modifier::Nvidia_16bx2_block_two_gob,
            Modifier::Nvidia_16bx2_block_four_gob,
            Modifier::Nvidia_16bx2_block_eight_gob,
            Modifier::Nvidia_16bx2_block_sixteen_gob,
            Modifier::Nvidia_16bx2_block_thirtytwo_gob,
        ];
        let preferred_codes: [Fourcc; 4] = [
            Fourcc::Argb8888,
            Fourcc::Xrgb8888,
            Fourcc::Abgr8888,
            Fourcc::Xbgr8888,
        ];
        let mut preferred_formats: Vec<Format> = Vec::new();
        if only_linear {
            preferred_modifiers.clear();
        }
        for modifier in preferred_modifiers {
            for code in preferred_codes {
                preferred_formats.push(Format { code, modifier });
            }
        }
        let allow_linear = std::env::var("METABONK_DMABUF_ALLOW_LINEAR")
            .map(|v| v == "1")
            .unwrap_or(false)
            || only_linear;
        if !allow_linear {
            info!("dmabuf: linear modifier disabled (set METABONK_DMABUF_ALLOW_LINEAR=1 to enable)");
        }
        let mut formats = preferred_formats.clone();
        let linear_formats: Vec<Format> = vec![
            Format { code: Fourcc::Argb8888, modifier: Modifier::Linear },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Linear },
            Format { code: Fourcc::Abgr8888, modifier: Modifier::Linear },
            Format { code: Fourcc::Xbgr8888, modifier: Modifier::Linear },
        ];
        if allow_linear {
            formats.extend(linear_formats.iter().copied());
        }
        info!(
            main_dev = format_args!("0x{:x}", main_dev as u64),
            formats_len = formats.len(),
            preferred_len = preferred_formats.len(),
            allow_linear,
            "dmabuf: configured feedback formats"
        );
        let mut builder = DmabufFeedbackBuilder::new(main_dev, formats);
        // Debug lever: when linear is enabled, offer a dedicated "linear first" tranche so that
        // clients which don't gracefully retry other modifiers have a known-good escape hatch.
        if allow_linear {
            builder = builder.add_preference_tranche(main_dev, None, linear_formats);
        }
        if !preferred_formats.is_empty() {
            builder = builder.add_preference_tranche(main_dev, None, preferred_formats);
        }
        let default_feedback = builder.build().context("build dmabuf feedback")?;
        let dmabuf_global = dmabuf_state.create_global_with_default_feedback::<EyeCompositor>(&dh, &default_feedback);

        let disable_wl_drm = std::env::var("METABONK_DISABLE_WL_DRM")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        let wl_drm_global_data = if disable_wl_drm {
            warn!("wl_drm: disabled via METABONK_DISABLE_WL_DRM=1 (forcing XWayland to use linux-dmabuf + feedback)");
            None
        } else {
            // XWayland still expects a `wl_drm` global for device discovery and PRIME capability gating,
            // even if it later submits buffers via `zwp_linux_dmabuf_v1`. However, some stacks will use
            // `wl_drm` PRIME buffers (no modifiers) which can lead to misinterpreting tiled layouts.
            //
            // METABONK_DISABLE_WL_DRM=1 is a diagnostic/escape hatch to force linux-dmabuf usage.
            let data = {
                let render_node = resolve_render_node_path(main_dev)
                    .with_context(|| format!("resolve wl_drm device path for main_dev=0x{:x}", main_dev as u64))?;
                // wl_drm "device" should generally be a render node so clients can open it without
                // drm master / authentication. Exposing a primary node (card*) requires a real
                // drmAuthMagic implementation, which we intentionally do not provide in this
                // headless compositor.
                //
                // Allow overriding for experiments via METABONK_DRM_CARD_NODE, but default to the
                // render node resolved from linux-dmabuf feedback.
                let device_node = if let Ok(p) = std::env::var("METABONK_DRM_CARD_NODE") {
                    PathBuf::from(p)
                } else {
                    render_node.clone()
                };
                WlDrmGlobalData {
                    device_path: device_node.to_string_lossy().to_string(),
                    main_dev: main_dev as u64,
                    formats: vec![
                        Fourcc::Argb8888 as u32,
                        Fourcc::Xrgb8888 as u32,
                        Fourcc::Abgr8888 as u32,
                        Fourcc::Xbgr8888 as u32,
                    ],
                    capabilities: wl_drm::Capability::Prime as u32,
                }
            };
            Some(data)
        };

        // Advertise a headless wl_output. XWayland (and other clients) expect at least one output
        // to exist before committing buffers in rootless mode.
        let output = Output::new(
            "metabonk-output-0".into(),
            PhysicalProperties {
                size: (0, 0).into(),
                subpixel: Subpixel::Unknown,
                make: "MetaBonk".into(),
                model: "SmithayEye".into(),
            },
        );
        let _output_global = output.create_global::<EyeCompositor>(&dh);
        let mode = Mode {
            size: (output_w as i32, output_h as i32).into(),
            refresh: 60_000,
        };
        output.change_current_state(Some(mode), Some(Transform::Normal), Some(Scale::Integer(1)), Some((0, 0).into()));
        output.set_preferred(mode);
        if let Ok(mut g) = export_size.lock() {
            *g = (output_w, output_h);
        }

        let drm_syncobj_state: Option<DrmSyncobjState> = (|| {
            let disable = std::env::var("METABONK_DISABLE_DRM_SYNCOBJ")
                .ok()
                .map(|v| v.trim() == "1")
                .unwrap_or(false);
            if disable {
                warn!("linux-drm-syncobj-v1 disabled via METABONK_DISABLE_DRM_SYNCOBJ=1");
                return None;
            }

            use smithay::backend::drm::DrmDeviceFd;
            use smithay::utils::DeviceFd;
            use std::os::unix::fs::MetadataExt;

            // Prefer the render node matching linux-dmabuf feedback main device.
            let mut render_node: Option<std::path::PathBuf> = None;
            if let Ok(p) = std::env::var("METABONK_DRM_RENDER_NODE") {
                render_node = Some(std::path::PathBuf::from(p));
            } else if let Ok(entries) = std::fs::read_dir("/dev/dri") {
                for ent in entries.flatten() {
                    let name = ent.file_name();
                    let name = name.to_string_lossy();
                    if !name.starts_with("renderD") {
                        continue;
                    }
                    let Ok(md) = ent.metadata() else { continue };
                    if md.rdev() as libc::dev_t == main_dev {
                        render_node = Some(ent.path());
                        break;
                    }
                }
            }
            let render_node = match render_node {
                Some(p) => p,
                None => {
                    warn!("linux-drm-syncobj-v1 disabled: could not resolve render node (set METABONK_DRM_RENDER_NODE)");
                    return None;
                }
            };
            let drm_file = match std::fs::OpenOptions::new().read(true).write(true).open(&render_node) {
                Ok(f) => f,
                Err(err) => {
                    warn!(?err, render_node = %render_node.to_string_lossy(), "linux-drm-syncobj-v1 disabled: failed to open render node");
                    return None;
                }
            };
            let owned: OwnedFd = drm_file.into();
            let dev_fd = DeviceFd::from(owned);
            let drm_dev = DrmDeviceFd::new(dev_fd);
            if supports_syncobj_eventfd(&drm_dev) {
                info!("linux-drm-syncobj-v1 enabled for explicit sync");
                Some(DrmSyncobjState::new::<EyeCompositor>(&dh, drm_dev))
            } else {
                warn!("linux-drm-syncobj-v1 disabled: drmSyncobjEventfd not supported on this device");
                None
            }
        })();

        let mut state = EyeCompositor::new(
            &dh,
            output_w as i32,
            output_h as i32,
            output,
            export_period,
            export_size,
            latest_dmabuf,
            dmabuf_state,
            drm_syncobj_state,
            dmabuf_global,
            wl_drm_global_data,
        );

        // Listen for Wayland clients (required for XWayland to roundtrip).
        let socket_source = ListeningSocketSource::with_name(&wayland_display)?;
        let socket_name = socket_source.socket_name().to_string_lossy().to_string();
        info!(wayland_display = %socket_name, "Wayland socket ready");
        let mut dh_accept = dh.clone();
        handle.insert_source(socket_source, move |client_stream, _, _state| {
            let pid = getsockopt(&client_stream, PeerCredentials)
                .map(|c| c.pid() as u32)
                .unwrap_or(0);
            let client_state = ClientState {
                compositor_state: CompositorClientState::default(),
                pid,
            };
            if let Ok(id) = dh_accept.insert_client(client_stream, Arc::new(client_state)) {
                let _ = id;
            }
            if pid != 0 {
                _state.ensure_agent(pid);
                _state.broadcast_agent_detected(pid);
            }
        })?;

        // Spawn XWayland and attach an XWM.
        // Capture XWayland stdout/stderr for debugging (glamor/DRI3/DMABUF fallbacks).
        let xwayland_log_path = run_dir.join("xwayland.log");
        let xwayland_log = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(&xwayland_log_path)
            .with_context(|| format!("open({xwayland_log_path:?})"))?;
        let xwayland_log_err = xwayland_log
            .try_clone()
            .with_context(|| format!("clone({xwayland_log_path:?})"))?;
        let mut xwayland_env: Vec<(String, String)> = vec![
            // XWayland needs to connect to our per-instance Wayland socket. Since we isolate
            // each compositor by rebinding XDG_RUNTIME_DIR, we must also provide the matching
            // WAYLAND_DISPLAY name explicitly for the XWayland child process.
            ("XDG_RUNTIME_DIR".to_string(), run_dir.to_string_lossy().to_string()),
            ("WAYLAND_DISPLAY".to_string(), socket_name.clone()),
        ];
        let wayland_debug = std::env::var("METABONK_XWAYLAND_WAYLAND_DEBUG")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false);
        if wayland_debug {
            xwayland_env.push(("WAYLAND_DEBUG".to_string(), "client".to_string()));
        }
        if std::env::var("METABONK_XWAYLAND_NO_GLAMOR")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false)
        {
            xwayland_env.push(("XWAYLAND_NO_GLAMOR".to_string(), "1".to_string()));
        }
        if std::env::var("METABONK_XWAYLAND_NO_DRI3")
            .ok()
            .map(|v| v.trim() == "1")
            .unwrap_or(false)
        {
            xwayland_env.push(("XWAYLAND_NO_DRI3".to_string(), "1".to_string()));
        }
        let (xwayland_source, xwayland_client) = XWayland::spawn(
            &dh,
            None,
            xwayland_env.into_iter(),
            true,
            Stdio::from(xwayland_log),
            Stdio::from(xwayland_log_err),
            |_| {},
        )
        .context("spawn XWayland")?;

        let run_dir_env = run_dir.clone();
        let frame_sock_env = frame_sock.clone();
        let wayland_display_env = socket_name.clone();
        let dh_for_creds = dh.clone();

        let handle_wm = handle.clone();
        handle.insert_source(xwayland_source, move |event, _, state| {
            match event {
                XWaylandEvent::Ready { x11_socket, display_number } => {
                    if let Ok(creds) = xwayland_client.get_credentials(&dh_for_creds) {
                        let pid = creds.pid as u32;
                        if pid != 0 {
                            if let Some(cd) = xwayland_client.get_data::<smithay::xwayland::XWaylandClientData>() {
                                let _ = cd.user_data().insert_if_missing_threadsafe(|| XWaylandPid(pid));
                            }
                            state.ensure_agent(pid);
                            state.broadcast_agent_detected(pid);
                        }
                    }
                    let wm = match X11Wm::start_wm(handle_wm.clone(), x11_socket, xwayland_client.clone()) {
                        Ok(wm) => wm,
                        Err(e) => {
                            warn!("Failed to start X11 WM: {e}");
                            return;
                        }
                    };
                    let id = wm.id();
                    state.xwms.insert(id, wm);
                    let disp = format!(":{}", display_number);
                    state.xwayland_ready_at = Some(Instant::now());
                    info!(display = %disp, "XWayland ready");
                    if let Err(e) = write_compositor_env(
                        &run_dir_env,
                        &wayland_display_env,
                        &frame_sock_env,
                        &disp,
                    ) {
                        warn!("Failed to write compositor.env: {e}");
                    }
                }
                XWaylandEvent::Error => {
                    warn!("XWayland failed to start");
                }
            }
        })?;

        // Main loop: dispatch calloop sources + wayland clients.
        loop {
            event_loop.dispatch(Duration::from_millis(16), &mut state)?;
            display.dispatch_clients(&mut state)?;
            state.drain_due_frame_callbacks();
            display.flush_clients()?;
        }
    })
}
