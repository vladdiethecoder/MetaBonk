# Compositor Paradigms: Smithay (Rust) vs Gamescope (Valve)

This document captures a comparative analysis of two compositor strategies for Linux gaming and
machine-learning capture pipelines:

- **Smithay**: build-your-own compositor framework in Rust (max control, easiest to add custom RL
  protocols / lock-step stepping).
- **Gamescope**: Valve’s micro-compositor optimized for Steam (best “it just works” compatibility and
  production gaming ergonomics).

MetaBonk currently supports both philosophies:

- **Research rig (default)**: `metabonk_smithay_eye` (“Synthetic Eye”) for DMA-BUF + fence export into
  the agent loop. See `docs/synthetic_eye.md` and `docs/synthetic_eye_abi.md`.
- **Production console / containment (optional)**: Gamescope + PipeWire style flows (useful for
  compatibility, HDR, upscaling, and mainstream Steam behavior).

Below is the original report text (lightly reformatted for Markdown).

---

## Architectural Paradigms for High-Performance Linux Gaming and Machine Learning Compositors: A Comparative Analysis of Rust Smithay and Valve Gamescope

### 1. Introduction: The Linux Display Server Transition and Gaming Requirements

The Linux graphics stack is currently navigating a pivotal transition from the legacy X Window
System (X11) to the modern Wayland protocol. This shift represents more than a mere software
update; it is a fundamental architectural realignment of how graphical content is rendered,
composited, and displayed. For systems engineers tasked with developing custom gaming environments
—ranging from embedded console operating systems to high-frequency reinforcement learning (RL)
research rigs—the choice of the underlying compositor architecture is the single most consequential
decision. The display server acts as the arbiter of reality for the running application; it controls
the frame timing, input latency, and the very visibility of the rendered pixels.

Historically, X11 served as a monolithic server that handled everything from input routing to
drawing primitives. While robust for network-transparent desktop computing of the 1990s, X11
introduces inherent architectural bottlenecks for modern gaming. Its design enforces a client-server
round-trip model that can introduce stochastic latency, and its lack of mandatory compositing often
leads to screen tearing or stuttering when the application's frame rate desynchronizes from the
display's vertical blanking interval (VBlank). Furthermore, X11’s permissive security model, where
any client can snoop on the input or output of another, poses security risks and complicates the
isolation required for sandboxed gaming environments.

Wayland addresses these deficiencies by collapsing the display server and the window manager into a
single entity: the compositor. In this model, the compositor is the kernel-level display master.
Clients render their own buffers (typically using OpenGL or Vulkan) and pass handles to these
buffers—often via the Direct Rendering Manager (DRM) and DMA-BUF subsystems—directly to the
compositor for presentation. This architecture eliminates the "middleman" of the X server rendering
API, enabling perfect frame pacing, strict isolation between applications, and the potential for
zero-copy data paths that are critical for high-performance machine learning applications.

However, the "Wayland Compositor" is not a single piece of software but a protocol specification.
The implementation is left to the developer. This report evaluates two divergent paths for
constructing a custom gaming compositor compatible with the Steam ecosystem: building a bespoke
solution using the Smithay framework in Rust, or leveraging the existing, highly specialized
Gamescope micro-compositor developed by Valve. This analysis delves into the technical trade-offs
of safety versus performance, modularity versus integration, and the specific mechanics of
integrating Steam, Proton, and high-speed capture pipelines for machine learning agents.

### 2. Smithay: The Constructivist Approach in Rust

Smithay represents the "constructivist" philosophy in the Wayland ecosystem. It is not a functional
compositor out of the box; rather, it is a library of Rust crates that provide the building blocks
—the "smithy"—necessary to forge a compositor. This approach appeals to developers requiring
absolute control over the window management logic, input handling policies, and rendering pipeline,
leveraging the memory safety guarantees of the Rust programming language.

#### 2.1 The Rust Safety Guarantee in Graphics Programming

The domain of display servers is notoriously fraught with memory safety hazards. Managing the
lifecycles of surfaces, buffers, pointer constraints, and keyboard states involves complex state
machines where use-after-free errors or race conditions can lead to catastrophic crashes (segfaults)
that bring down the entire graphical session. In C-based libraries like wlroots (used by Sway and
Hyprland), the developer must meticulously manage these lifetimes manually.

Smithay utilizes Rust’s ownership and borrowing system to enforce these constraints at compile time.
For example, a reference to a Wayland client’s surface cannot outlive the client connection itself.
The Smithay API is designed such that accessing a buffer requires proving that the buffer is still
valid and mapped in memory. This drastically reduces the surface area for bugs that cause the
compositor to crash, a critical feature for long-running embedded systems or unsupervised RL
training clusters.

#### 2.2 The Event Loop and State Management

Smithay is architected around calloop, a Rust-native, callback-oriented event loop. This aligns
perfectly with the Wayland design philosophy, where the compositor spends most of its time idle,
waiting for specific events: a hardware interrupt from an input device, a socket message from a
client, or a VBlank signal from the DRM subsystem.

In a Smithay implementation, the developer defines a central struct representing the compositor's
state (containing window lists, input focus, etc.). This state is wrapped in a mechanism that
allows it to be passed mutably into the event callbacks. This avoids the "callback hell" or complex
shared-pointer graphs (Arc/Mutex) often seen in other asynchronous architectures. The event loop
drives the entire system:

- Input Events: libinput events are caught, processed (e.g., applying custom sensitivity curves),
  and dispatched to the focused window.
- Wayland Events: Requests from clients (e.g., "maximize window," "lock pointer") are handled via
  trait implementations like XdgShellHandler.
- Render Loop: The loop waits for the display hardware to signal readiness (VBlank) before
  compositing the next frame, ensuring synchronization.

#### 2.3 The XWayland Hurdle for Steam Gaming

The most significant engineering challenge when building a Smithay compositor for Steam is XWayland
support. Despite the push for Wayland, the vast majority of the Steam back catalog relies on X11.
Even modern Windows games running through Proton (Valve's Wine fork) present themselves as X11
applications because Wine's Wayland driver is still maturing.

Therefore, a Smithay compositor cannot simply speak Wayland; it must spawn and manage a rootless X
server (XWayland). Smithay provides the xwayland crate to facilitate this, but the integration is
non-trivial. The compositor must act as an X11 Window Manager, handling:

- Window Mapping: Converting X11 windows into Wayland surfaces.
- Atom Handling: Translating X11 atoms (properties) to Wayland protocols.
- Selection/Clipboard: synchronizing the X11 clipboard with the Wayland clipboard data source.
- Focus Stealing: Managing the complex and aggressive focus-stealing logic often found in older
  games.

Failure to implement this correctly results in games that launch but do not render, input that
doesn't register, or "alt-tab" behavior that crashes the game. While Smithay provides the tools,
the logic is left to the implementer, representing a high development overhead compared to using an
existing solution.

#### 2.4 Rendering Backend Flexibility

One of Smithay's strengths is its backend agnosticism. It can run on "bare metal" using the
backend_drm crate, directly leasing the display connectors via KMS. Alternatively, it can run nested
inside another windowing system (via winit). This allows for rapid development on a desktop before
deploying to the target hardware.

For rendering, Smithay typically uses OpenGL (EGL) via glow or renderer_gl. However, because it is a
library, a developer is free to implement a custom renderer using wgpu (WebGPU for Rust) or vulkano.
This capability is theoretically powerful for RL, as one could write a renderer that doesn't output
to a screen at all, but instead copies the rendered texture directly to a shared memory tensor for
an AI agent, completely bypassing the display hardware overhead.

### 3. Gamescope: The Specialized Micro-Compositor Architecture

Gamescope represents the antithesis of the "build-it-yourself" model. It is a "micro-compositor"
designed by Valve with a singular, uncompromised goal: to provide the optimal environment for
running games on Linux. Derived originally from wlroots, it has diverged significantly to implement
a highly specialized rendering pipeline tailored for the Steam Deck.

#### 3.1 The Nested vs. Embedded Paradigm

Gamescope is engineered to operate in two distinct modes:

- Nested Mode: Gamescope runs as a client application window on top of an existing desktop
  environment. Inside this window, it runs its own nested Wayland and X11 sessions.
- Embedded Mode: Gamescope runs directly on the DRM/KMS backend, taking full control of the display
  hardware and input devices. In this mode, it is the display server.

#### 3.2 Asynchronous Vulkan Compute Composition

Unlike standard desktop compositors that use OpenGL for composition, Gamescope uses Vulkan. More
specifically, it utilizes asynchronous compute queues. In a traditional rendering loop, the
composition step often blocks the graphics queue, potentially stalling the game's own rendering if
the compositor is heavy. Gamescope decouples these processes. It submits composition commands
(scaling, color conversion) to the compute queue, which can execute in parallel with the game's
graphics work on the GPU. This architecture minimizes the "photon-to-motion" latency and ensures
that frame delivery is consistent, even if the game's frame times fluctuate.

#### 3.3 Integrated Upscaling and Color Management

Because Gamescope controls the final composition step, it integrates advanced signal processing
features directly into the pipeline:

- FSR and NIS: includes shaders for AMD FSR and NVIDIA Image Scaling.
- HDR and Colorimetry: robust HDR support and color transforms before output.

#### 3.4 The "Gamescope WSI" Layer

Gamescope creates a hermetic environment. It spawns a private XWayland instance that is invisible
to the outside world. Games running inside cannot see other windows, effectively sandboxing them.
Gamescope intercepts the X11 frames and presents them as Vulkan textures, enabling forced
resolutions, refresh rates, and aspect ratios.

### 4. Comparative Architecture Analysis

For a developer tasked with building a custom gaming system, the choice between Smithay and
Gamescope is a trade-off between extensibility and commodity.

#### 4.1 Development Velocity and Maintenance

Gamescope offers an immediate, production-grade solution. Integrating it involves spawning the
binary with the correct flags. The maintenance burden is low, as Valve actively maintains it.
However, it is a "black box." If Gamescope lacks a specific feature (e.g., a window layout policy
or custom input gesture), modifying its codebase can be significantly harder than extending a
Rust-based Smithay project.

Smithay requires a large upfront investment. The developer effectively becomes a display server
vendor, responsible for implementing XWayland support, input handling, and protocol negotiation.
However, the resulting codebase is entirely owned and understandable. For non-standard behaviors,
Smithay provides the necessary architectural freedom.

#### 4.2 Performance Profiles

In terms of raw gaming latency and frame pacing, Gamescope is currently unrivaled due to its async
Vulkan architecture. Replicating this in Smithay would require writing a custom Vulkan backend with
complex synchronization logic. Smithay's default backends are performant but optimize for
correctness and desktop usage rather than millisecond-critical competitive gaming.

#### 4.3 Reliability and Compatibility

Gamescope is the reference implementation for the Steam Deck. If a game works on the Deck, it works
on Gamescope. Using Smithay introduces the risk of "protocol gaps" where a game relies on obscure
behavior/extension that the implementer has not yet replicated.

### 5. Steam Runtime Integration Mechanics

Steam can be launched programmatically via:

- `steam -applaunch <AppID>`
- `steam://run/<AppID>` URI scheme

For Rust launchers, the `steamlocate` crate can be used to locate library folders and app manifests
(`appmanifest_<AppID>.acf`), enabling verification and metadata logging for reproducibility.

### 6. The Reinforcement Learning Pipeline: Zero-Copy Capture

The Linux kernel DMA-BUF subsystem enables zero-copy pipelines. A buffer allocated by the GPU for
the game's frame can be exported as a file descriptor (FD) and imported by another process (agent
or encoder) to read directly from GPU memory.

#### 6.2 Gamescope and PipeWire

Gamescope can expose a PipeWire source node. Consumers can negotiate DMA-BUF transport via a media
pipeline (e.g., GStreamer) and avoid CPU readback.

#### 6.3 RL-Specific: wlr-screencopy vs PipeWire

- `wlr-screencopy-unstable-v1`: pull model (client requests a frame) — good for lock-step RL.
- PipeWire: push model (frames pushed at refresh rate) — optimized for streaming, not deterministic
  stepping.

### 7. The Frontend: Tauri v2 and Overlay Integration

Tauri v2 provides a lightweight Rust backend with a web frontend. The "sidecar" pattern is useful
for launching an embedded session compositor (e.g., Gamescope) and keeping it tied to the UI’s
lifecycle.

### 8. Synthesis and Strategic Recommendations

#### Path A: Production Console (Gamescope + Tauri)

If the objective is a consumer-ready gaming device or streaming server, Gamescope is a strong
baseline: compatibility, performance features (HDR/upscaling), and well-tested behavior.

#### Path B: Research Rig (Smithay + Custom Protocols)

If the objective is deterministic RL stepping, Smithay is superior: custom protocols / lock-step
pull-style capture are easier to implement, and the compositor can be treated as a controllable
simulation clock.

**Summary:** for most “custom gaming environment” tasks, Gamescope is the engine and Rust/Tauri is
the steering wheel; for synchronous AI training, Smithay provides the necessary hook points to step
time that Gamescope abstracts away.

