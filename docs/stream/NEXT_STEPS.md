# MetaBonk Deployment - Next Steps & Troubleshooting

## ✅ SUCCESSFUL IMPLEMENTATION

You've successfully deployed NVENC session limiting + safe streaming selection. Here's what's working:

### What's Working Perfectly:
- ✅ NVENC session limiting active (`METABONK_NVENC_MAX_SESSIONS` + cross-process slot leasing)
- ✅ Optimal streaming path: `gst:cuda_appsrc:nvh264enc`
- ✅ 57 FPS capture (target: 60 FPS)
- ✅ Zero dropped frames (3823 frames captured)
- ✅ Good source resolution (1280x720)
- ✅ 1080p spectator output
- ✅ 98 tests passing
- ✅ Beautiful UI rendering

## 🔍 Understanding "NO KEYFRAME / Waiting..."

This message appears in your screenshot. Here's what it means:

### Most Likely Causes:

**1. Worker Without NVENC Capacity (Expected)**
   - With 5 workers and a GeForce-style NVENC cap (often 2 sessions)
   - Only 2 workers can actively stream at a time; others should return HTTP 503
   - This is CORRECT behavior: those workers keep training, but the UI won't show a live feed for them
 
   **Solution:** This is working as designed! Check which workers have sessions:
   ```bash
   python scripts/nvenc_status.py --workers 5
   ```

**2. Stream Not Connected Yet**
   - Your JSON shows: `"stream_active_clients": 0`
   - Browser might not be connected to this specific worker
   
   **Solution:** Refresh the page or wait for auto-reconnect

**3. Initial Keyframe Sync Delay**
   - Browser MSE needs keyframe to start playback
   - Can take 0.5-2 seconds on first load
   
   **Solution:** Wait a few seconds, should resolve automatically

## 📊 Validation Commands

Run these to verify everything:

### 1. Check NVENC Session Status
```bash
python scripts/nvenc_status.py --workers 5
```
Expected output:
```
NVML: gpu_index=0 name=... nvenc_used=2
- worker=0 port=5000 instance_id=omega-0 backend=... active=... nvenc_used=...
- worker=1 port=5001 instance_id=omega-1 backend=... active=... nvenc_used=...
...
```

### 2. Check Which Workers Are Streaming
```bash
curl -s http://127.0.0.1:8040/workers | jq 'to_entries[] | {id: .value.instance_id, stream_url: .value.stream_url, backend: .value.stream_backend, err: .value.streamer_last_error}'
```

### 3. Full Validation Suite
```bash
python docs/stream/validate_deployment.py
```
This runs 5 comprehensive checks:
- NVENC session management
- Worker streaming status
- FPS & quality
- Resolution validation
- UI accessibility

### 4. Check Individual Worker Stream
Try accessing worker streams directly:
```bash
# Worker 0
curl -I http://127.0.0.1:5000/stream.mp4

# Worker 1
curl -I http://127.0.0.1:5001/stream.mp4
```

Expected responses:
- Workers WITH sessions: HTTP 200 OK
- Workers WITHOUT sessions: HTTP 503 Service Unavailable

## 🚀 Next Steps

### Step 1: Verify Session Allocation
```bash
# See which workers got NVENC sessions
python scripts/nvenc_status.py --workers 5
```

### Step 2: Test Specific Worker Streams
Open these URLs in browser:
```
http://localhost:5173/stream?worker=omega-0
http://localhost:5173/stream?worker=omega-1
```

Workers with NVENC sessions should stream immediately.
Workers without sessions will show "NO KEYFRAME" - this is expected!

### Step 3: View Featured Streams
Your neural broadcast page should auto-select workers WITH sessions:
```
http://localhost:5173/neural/broadcast
```

### Step 4: Monitor Performance
```bash
# Watch FPS in real-time
watch -n 1 'curl -s http://localhost:5000/api/worker/omega-0/status | jq "{fps: .frames_fps, dropped: .frames_dropped}"'
```

## 🎯 Expected Behavior Matrix

| Workers | NVENC Limit | Streaming | Non-Streaming | Behavior |
|---------|-------------|-----------|---------------|----------|
| 2 | 2 | 2 | 0 | ✅ All stream perfectly |
| 3 | 2 | 2 | 1 | ✅ 2 stream, 1 trains without streaming |
| 5 | 2 | 2 | 3 | ✅ 2 stream, 3 train without streaming |

**Key Point:** Non-streaming workers still train normally! They just don't have a video feed.

## 🔧 Troubleshooting Specific Issues

### Issue: "NO KEYFRAME" on ALL workers

**Diagnosis:**
```bash
python scripts/nvenc_status.py --workers 5
```

If shows `Active Sessions: 0`, then no workers have sessions.

**Solution:**
```bash
# Kill any zombie processes
python scripts/cleanup_nvenc_sessions.py --kill

# Restart MetaBonk
python scripts/stop.py --all --go2rtc
./start --mode train --workers 2  # Start with 2 workers first
```

### Issue: Workers crash on startup

**Check logs:**
```bash
tail -f logs/orchestrator.log | grep -i "nvenc\|session\|error"
```

**Look for:**
- "Session limit reached" → Expected if >2 workers
- "OpenEncodeSessionEx failed" → Should NOT appear anymore!
- "Acquired NVENC session" → Good!

### Issue: FPS below 30

**Check:**
```bash
# GPU utilization
nvidia-smi

# Worker status
curl http://localhost:5000/api/worker/omega-0/status | jq '{fps: .frames_fps, backend: .stream_backend, error: .stream_error}'
```

**Common causes:**
- Using wrong backend (should be `gst:cuda_appsrc:nvh264enc`)
- GPU overloaded
- Resolution too high

## 📈 Performance Tuning

### If You Want ALL 5 Workers to Stream

**Option 1: Get Professional GPU** (unlimited sessions)
- Tesla A100, V100
- Quadro RTX series
- Best solution for production

**Option 2: Unlocked Driver** (not recommended)
- Third-party patches exist to unlock consumer GPUs
- Violates NVIDIA EULA
- Can break on driver updates

**Option 3: Multiple GPUs**
- 2 GPUs = 4-6 sessions total
- Distribute workers across GPUs

### Option 4: Slow Down Worker Spawns
If NVENC init failures show up during mass restarts, force a stagger:
```bash
METABONK_WORKER_SPAWN_STAGGER_S=1.0 ./start --mode train --workers 5
```

### Option 5: RTX 5090 (8 workers) with WebRTC
If you want to *view many workers concurrently* on a GeForce-class NVENC limit, enable go2rtc/WebRTC and allow CPU encoder fallback for the overflow streams:
```bash
./start --mode train --workers 8 --go2rtc --stream-profile rtx5090_webrtc_8
```
Notes:
- This profile keeps NVENC capped (`METABONK_NVENC_MAX_SESSIONS=2`) but permits extra streams to fall back to software encoders (`METABONK_STREAM_ALLOW_CPU_FALLBACK=1`).
- Expect higher CPU usage; reduce stream bitrate/FPS if the host can’t keep up.
- Requires multi-GPU setup

**Option 4: Accept Limitation** (recommended)
- Run 5 workers, 2 stream, 3 train
- Rotate which workers are featured
- Most cost-effective solution

## ✅ Success Criteria Checklist

Run through this checklist:

- [ ] `python scripts/nvenc_status.py` shows reasonable session count
- [ ] At least 2 workers show `stream_backend: gst:cuda_appsrc:nvh264enc`
- [ ] Those 2 workers achieve 55-60 FPS
- [ ] Zero or minimal dropped frames
- [ ] UI accessible at http://localhost:5173/neural/broadcast
- [ ] Can see live gameplay on workers with sessions
- [ ] Non-streaming workers log graceful fallback (not errors)

## 🎉 What You've Achieved

Before your implementation:
- ❌ All workers failing with NVENC errors
- ❌ <20 FPS, constant reconnects
- ❌ "OpenEncodeSessionEx failed" spam
- ❌ Wasteful ffmpeg:pixel_obs path

After your implementation:
- ✅ Graceful session management
- ✅ 57 FPS on streaming workers
- ✅ Optimal CUDA → GStreamer path
- ✅ Zero NVENC errors in logs
- ✅ Professional UI working perfectly

**You've successfully deployed a production-grade NVENC session manager!**

## 📞 Need Help?

Run the validation script for detailed diagnostics:
```bash
python validate_deployment.py
```

This will tell you exactly what's working and what needs attention.
