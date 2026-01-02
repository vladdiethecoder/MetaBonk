# QUICK ACTION - Stream Banding Fix

**IMMEDIATE STEPS** (10 minutes)

---

## 🎯 STEP 1: RUN DIAGNOSTIC (3 min)

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# Run quick diagnostic
chmod +x quick_banding_diagnostic.sh
./quick_banding_diagnostic.sh
```

**What this does**:
- ✅ Checks real GPU memory (not VMS confusion)
- ✅ Captures frames from all workers
- ✅ Automatically detects banding
- ✅ Tells you if fix needed

---

## 🔧 STEP 2: APPLY FIX (If banding detected)

### **Quick Manual Fix**:

```bash
# Backup
cp src/worker/streamer.py src/worker/streamer.py.backup

# Edit file
nano src/worker/streamer.py

# Find this line (around line 200-300):
nvh264enc ! h264parse

# Replace with:
nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 ! h264parse

# Save: Ctrl+O, Enter, Ctrl+X
```

---

## ♻️ STEP 3: RESTART (2 min)

```bash
# Stop
python3 launch.py stop
sleep 5

# Start
python3 launch.py --workers 5
sleep 60
```

---

## ✅ STEP 4: VALIDATE (2 min)

```bash
# Re-run diagnostic
./quick_banding_diagnostic.sh
```

**Expected**:
```
✅ GPU memory normal
✅ NO BANDING DETECTED
✅ NO SIGNIFICANT ISSUES
```

---

## 📊 ABOUT THE VRAM

**The 350 GB VMS is NOT a problem!**

- `vms_mb` = Virtual address space (not actual memory)
- `rss_mb` = Actual RAM used (~250 MB per worker) ✅
- Real GPU VRAM: Check with `nvidia-smi` (~20 GB total for 5 workers) ✅

**This is completely normal for CUDA applications.**

---

## 🎬 ENCODER FIX EXPLAINED

**Before**:
```
nvh264enc ! h264parse
```
- Uses default settings (low quality for speed)
- Results in banding artifacts

**After**:
```
nvh264enc preset=hq rc-mode=cbr bitrate=8000 qp-min=16 qp-max=23 ! h264parse
```
- `preset=hq`: High quality mode
- `rc-mode=cbr`: Constant bitrate
- `bitrate=8000`: 8 Mbps (good for 720p)
- `qp-min=16, qp-max=23`: Quality bounds

**Result**: Smooth gradients, no banding ✅

---

## 🚨 IF ISSUES

**Banding still present after fix**:
1. Check if fix was applied: `grep "preset=hq" src/worker/streamer.py`
2. Verify workers restarted with new code
3. Try higher bitrate: `bitrate=10000`

**Can't capture frames**:
- Frames capture via go2rtc may not work in headless
- Banding would still be visible in actual gameplay
- Consider: Manual visual inspection

**Fix breaks streaming**:
- Revert: `mv src/worker/streamer.py.backup src/worker/streamer.py`
- Restart workers
- Check logs for GStreamer errors

---

## 📞 QUICK COMMANDS

**Run diagnostic**:
```bash
./quick_banding_diagnostic.sh
```

**Check real VRAM**:
```bash
nvidia-smi --query-gpu=memory.used --format=csv
```

**Verify fix applied**:
```bash
grep "preset=hq" src/worker/streamer.py
```

**Restart workers**:
```bash
python3 launch.py stop && python3 launch.py --workers 5
```

---

## 🎯 CHECKLIST

- [ ] Run diagnostic script
- [ ] Confirm banding detected (if any)
- [ ] Check real GPU memory (nvidia-smi)
- [ ] Apply encoder fix (if needed)
- [ ] Restart workers
- [ ] Re-run diagnostic to validate
- [ ] Proceed with 24hr test

---

**Total Time**: 10-15 minutes  
**Expected Result**: ✅ No banding, normal VRAM usage

**Start with Step 1 above!** ⬆️
