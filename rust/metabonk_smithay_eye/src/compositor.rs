use std::{
    cmp::Reverse,
    collections::BinaryHeap,
    collections::HashMap,
    fs,
    path::{Path, PathBuf},
    process::Stdio,
    sync::{Arc, Mutex},
    time::Instant,
    time::Duration,
};

use anyhow::Context;
use nix::sys::socket::{getsockopt, sockopt::PeerCredentials};
use smithay::{
    delegate_compositor, delegate_xdg_shell, delegate_xwayland_shell,
    backend::allocator::dmabuf::Dmabuf,
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
        output::OutputHandler,
        shell::xdg::{PopupSurface, PositionerState, ToplevelSurface, XdgShellHandler, XdgShellState, XdgToplevelSurfaceData},
        shm::{ShmHandler, ShmState},
        socket::ListeningSocketSource,
        xwayland_shell::{XWaylandShellHandler, XWaylandShellState},
    },
    xwayland::{
        xwm::{Reorder, ResizeEdge, WmWindowProperty, XwmId},
        X11Wm, X11Surface, XWayland, XWaylandEvent, XwmHandler,
    },
};
use smithay::reexports::wayland_server::protocol::wl_shm;
use tracing::{info, warn};
use smithay::backend::renderer::utils::on_commit_buffer_handler;
use smithay::backend::renderer::utils::RendererSurfaceStateUserData;

use crate::protocols::ext_metabonk_control_v1::server::zmetabonk_agent_v1::{self, ZmetabonkAgentV1};
use crate::protocols::ext_metabonk_control_v1::server::zmetabonk_orchestrator_v1::{self, ZmetabonkOrchestratorV1};

#[derive(Default)]
struct ClientState {
    compositor_state: CompositorClientState,
    pid: u32,
}

impl ClientData for ClientState {
    fn initialized(&self, _client_id: ClientId) {}
    fn disconnected(&self, _client_id: ClientId, _reason: DisconnectReason) {}
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

pub struct EyeCompositor {
    compositor_state: CompositorState,
    xwayland_shell_state: XWaylandShellState,
    xdg_shell_state: XdgShellState,
    seat_state: SeatState<Self>,
    shm_state: ShmState,
    dmabuf_state: DmabufState,
    #[allow(dead_code)]
    dmabuf_global: DmabufGlobal,
    #[allow(dead_code)]
    metabonk_control_global: smithay::reexports::wayland_server::backend::GlobalId,
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
    latest_dmabuf: Arc<Mutex<Option<Dmabuf>>>,
    primary_surface: Option<ObjectId>,
    xwayland_ready_at: Option<Instant>,
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
        latest_dmabuf: Arc<Mutex<Option<Dmabuf>>>,
        dmabuf_state: DmabufState,
        dmabuf_global: DmabufGlobal,
    ) -> Self {
        let compositor_state = CompositorState::new::<Self>(dh);
        let xwayland_shell_state = XWaylandShellState::new::<Self>(dh);
        let xdg_shell_state = XdgShellState::new::<Self>(dh);
        let seat_state = SeatState::new();
        // wl_shm is required by many clients (including XWayland) even if we refuse
        // SHM buffers for the vision path. We expose it for compatibility and enforce
        // GPU-only capture by ignoring SHM commits in `commit()`.
        let shm_state = ShmState::new::<Self>(dh, vec![
            wl_shm::Format::Argb8888,
            wl_shm::Format::Xrgb8888,
        ]);
        let metabonk_control_global = dh.create_global::<Self, ZmetabonkOrchestratorV1, _>(1, ());
        Self {
            compositor_state,
            xwayland_shell_state,
            xdg_shell_state,
            seat_state,
            shm_state,
            dmabuf_state,
            dmabuf_global,
            metabonk_control_global,
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
            xwayland_ready_at: None,
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
        // Accept all dmabufs from clients. We enforce "GPU-only" at the consumer by
        // ignoring SHM buffers and requiring a dmabuf-backed source to appear.
        let _ = notifier.successful::<Self>();
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

        // Frame callback throttling: drain callbacks requested for this commit and schedule them based on agent tier.
        if let Some(client) = surface.client() {
            let pid = client.get_data::<ClientState>().map(|cs| cs.pid).unwrap_or(0);
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
                None => return,
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

                if now.duration_since(self.last_commit_log_at) > Duration::from_secs(2) {
                    self.last_commit_log_at = now;
                    let sz = dmabuf.size();
                    let stride0 = dmabuf.strides().next().unwrap_or(0);
                    let offset0 = dmabuf.offsets().next().unwrap_or(0);
                    let modifier_u64: u64 = dmabuf.format().modifier.into();
                    let fmt = dmabuf.format().code;
                    let pid = surface
                        .client()
                        .and_then(|c| c.get_data::<ClientState>().map(|cs| cs.pid))
                        .unwrap_or(0);
                    info!(
                        pid,
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
                    if let Some(cs) = client.get_data::<ClientState>() {
                        let agent = self.ensure_agent(cs.pid);
                        agent.surface = Some(surface.clone());
                    }
                }

                if self.primary_surface.is_none() && self.featured_pid.is_none() {
                    self.primary_surface = Some(surface.id());
                    info!("selected primary surface for capture (first DMA-BUF commit)");
                }
                if let Some(primary) = self.primary_surface.as_ref() {
                    if &surface.id() != primary {
                        return;
                    }
                }
                if let Some(featured) = self.featured_pid {
                    if let Some(client) = surface.client() {
                        if let Some(cs) = client.get_data::<ClientState>() {
                            if cs.pid != featured {
                                return;
                            }
                        }
                    }
                }
                let mut g = self.latest_dmabuf.lock().unwrap();
                *g = Some(dmabuf.clone());
            } else {
                // Ignore SHM buffers to preserve the GPU-only contract. Do NOT clear latest_dmabuf here:
                // some clients briefly commit SHM surfaces (cursor/overlays) even while the main window
                // continues to present DMA-BUF frames.
                if now.duration_since(self.last_non_dmabuf_log_at) > Duration::from_secs(2) {
                    self.last_non_dmabuf_log_at = now;
                    warn!("non-dmabuf wl_buffer commit (SHM?) - GPU-only path requires linux-dmabuf");
                }
            }
        });
    }
}

impl XWaylandShellHandler for EyeCompositor {
    fn xwayland_shell_state(&mut self) -> &mut XWaylandShellState {
        &mut self.xwayland_shell_state
    }

    fn surface_associated(&mut self, _xwm_id: XwmId, surface: WlSurface, _window: X11Surface) {
        // Do not pick a primary surface here. XWayland can associate multiple wl_surfaces, some of
        // which may be SHM-only (cursor/overlays). We choose the primary only once a DMA-BUF commit
        // is observed in `commit()`.
        let _ = surface;
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
    type KeyboardFocus = WlSurface;
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
            .and_then(|c| c.get_data::<ClientState>().map(|cs| cs.pid))
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
            .and_then(|c| c.get_data::<ClientState>().map(|cs| cs.pid))
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
smithay::delegate_shm!(EyeCompositor);
smithay::delegate_dmabuf!(EyeCompositor);
smithay::delegate_output!(EyeCompositor);
delegate_xwayland_shell!(EyeCompositor);
delegate_xdg_shell!(EyeCompositor);

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
    export_period: Arc<Mutex<Duration>>,
    export_size: Arc<Mutex<(u32, u32)>>,
    latest_dmabuf: Arc<Mutex<Option<Dmabuf>>>,
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
        let main_dev: libc::dev_t = {
            let node = std::env::var("METABONK_DRM_RENDER_NODE").unwrap_or_else(|_| "/dev/dri/renderD128".into());
            let md = std::fs::metadata(&node).with_context(|| format!("metadata({node})"))?;
            use std::os::unix::fs::MetadataExt;
            md.rdev() as libc::dev_t
        };
        use smithay::backend::allocator::{Format, Fourcc, Modifier};
        let formats: Vec<Format> = vec![
            Format { code: Fourcc::Argb8888, modifier: Modifier::Linear },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Linear },
            Format { code: Fourcc::Argb8888, modifier: Modifier::Nvidia_16bx2_block_one_gob },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Nvidia_16bx2_block_one_gob },
            Format { code: Fourcc::Argb8888, modifier: Modifier::Nvidia_16bx2_block_two_gob },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Nvidia_16bx2_block_two_gob },
            Format { code: Fourcc::Argb8888, modifier: Modifier::Nvidia_16bx2_block_four_gob },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Nvidia_16bx2_block_four_gob },
            Format { code: Fourcc::Argb8888, modifier: Modifier::Nvidia_16bx2_block_eight_gob },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Nvidia_16bx2_block_eight_gob },
            Format { code: Fourcc::Argb8888, modifier: Modifier::Nvidia_16bx2_block_sixteen_gob },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Nvidia_16bx2_block_sixteen_gob },
            Format { code: Fourcc::Argb8888, modifier: Modifier::Nvidia_16bx2_block_thirtytwo_gob },
            Format { code: Fourcc::Xrgb8888, modifier: Modifier::Nvidia_16bx2_block_thirtytwo_gob },
        ];
        let default_feedback = DmabufFeedbackBuilder::new(main_dev, formats)
            .build()
            .context("build dmabuf feedback")?;
        let dmabuf_global = dmabuf_state.create_global_with_default_feedback::<EyeCompositor>(&dh, &default_feedback);

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

        let mut state = EyeCompositor::new(
            &dh,
            output_w as i32,
            output_h as i32,
            output,
            export_period,
            export_size,
            latest_dmabuf,
            dmabuf_state,
            dmabuf_global,
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
        let (xwayland_source, xwayland_client) = XWayland::spawn(
            &dh,
            None,
            // XWayland needs to connect to our per-instance Wayland socket. Since we isolate
            // each compositor by rebinding XDG_RUNTIME_DIR, we must also provide the matching
            // WAYLAND_DISPLAY name explicitly for the XWayland child process.
            [
                ("XDG_RUNTIME_DIR".to_string(), run_dir.to_string_lossy().to_string()),
                ("WAYLAND_DISPLAY".to_string(), socket_name.clone()),
            ]
            .into_iter(),
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
