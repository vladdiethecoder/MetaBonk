# OBS Browser Source Overlay Integration (Metabonk Stream UI)

This note explains why the Metabonk Stream UI can look cropped or misaligned in OBS, and how to make the browser source fit cleanly at broadcast resolutions.

## Summary

OBS Browser Source uses a fixed internal viewport (Width/Height in source properties). If that viewport is smaller than the UI expects, CEF will render a truncated layout. Resizing the red bounding box in OBS only scales the rendered texture; it does not change the browser viewport. The result is a stretched, blurry, or cropped overlay.

Fix: set the Browser Source Width/Height to match the UI’s native resolution (usually 1920x1080 for the Stream page), then crop or scale the source as needed in the scene.

## Rendering Pipeline (What Actually Happens)

1) CEF renders HTML/CSS into a bitmap at the Browser Source Width/Height.
2) OBS takes that bitmap and applies scene transforms (scale/crop/position).
3) If the CEF viewport is wrong, the layout is already broken before OBS can fix it.

The right mental model is: the Browser Source is a headless Chrome tab with a fixed window size.

## Recommended Settings for Metabonk Stream

- Browser Source URL: `http://127.0.0.1:5173/stream`
- Width: `1920`
- Height: `1080`
- FPS: `60`
- Use OBS transform to scale/crop if you need the UI to occupy only part of the canvas.

If you want a shorter strip overlay (e.g., bottom HUD), keep the browser source at 1920x1080 and crop the top with Alt-drag or a Crop/Pad filter. This preserves layout and text clarity.

## Diagnosing Layout Issues (Use the Interact Window)

Right-click the source in OBS → Interact.

- If the Interact window is cropped: the Browser Source viewport is too small.
- If the Interact window is fine but OBS preview is cropped: there is a scene crop or transform issue.
- If the Interact window shows scrollbars: the UI is larger than the viewport and is being clipped.

## DPI and Scaling Problems (Windows)

High DPI scaling (125%/150%) can cause the browser source to render “zoomed in.” Symptoms include:

- only the top-left of the UI visible
- text looks larger than expected
- layout reflows to a mobile-like grid

Mitigations:

- Ensure OBS is DPI aware (Windows display settings can affect this).
- Prefer matching the Browser Source size to the UI’s expected resolution.
- If using Window Capture (Overwolf or external UI), override High DPI behavior for that executable.

## CSS Injection (Optional)

If you must fit a 1920x1080 UI into a smaller space, use CSS in the Browser Source properties. Two common approaches:

### `zoom` (reflows layout)

```
body { zoom: 0.8; }
```

This shrinks the layout and reflows the grid. It is usually the most stable choice for dashboards.

### `transform: scale` (no reflow)

```
body {
  transform: scale(0.8);
  transform-origin: 0 0;
  width: 125%;
}
```

This scales the rendered result but does not reflow elements, which can leave extra whitespace.

### Transparent background

If you want the overlay to sit on top of gameplay:

```
html, body { background: transparent !important; }
```

## Common Fixes by Symptom

- **Content cut off right/bottom** → increase Browser Source width/height.
- **Content is blurry** → set Browser Source to the real resolution, don’t scale from 800x600.
- **Layout uses fewer columns than expected** → viewport is below a CSS breakpoint.
- **Overlay is clipped in OBS preview only** → remove scene crop (Alt-drag) or reset transform.
- **Black/frozen browser source** → toggle Browser Source hardware acceleration (rare GPU conflicts).

## Practical Overlay Workflow

1) Set Browser Source to the UI’s native size (1920x1080).
2) Verify layout in Interact window.
3) Crop down to the region you need.
4) Position and scale in the scene as needed.

This preserves pixel clarity and prevents the “UI elements don’t fit” class of issues.
