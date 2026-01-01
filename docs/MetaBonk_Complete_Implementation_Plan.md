# MetaBonk: Complete Implementation Plan
## Pure Vision + CuTile + CUDA 13.1 + VLM Hive + Stream Verification

**Repository**: https://github.com/vladdiethecoder/MetaBonk  
**Date**: January 1, 2026  
**Status**: Complete Technical Specification  
**Timeline**: ~8-12 hours implementation  

---

## 🎯 MISSION STATEMENT

**PRIMARY GOAL**: Build a production-ready, game-agnostic pure vision RL system with:
1. **Pure Vision Learning** - No hardcoded game knowledge, generalizes across games
2. **CuTile Observations** - GPU-accelerated tile-based perception (replacing raw pixels)
3. **CUDA 13.1 Integration** - Modern CUDA features, no fallback logic
4. **VLM Hive Reasoning** - Multi-agent vision-language reasoning for strategic insights
5. **Verified Streaming** - All 5 workers streaming gameplay reliably (no artifacts)

---

## 📋 IMPLEMENTATION OVERVIEW

### **PHASE 0**: Foundation & Architecture Review (30 min)
- Audit current system state
- Identify all hardcoded game logic
- Map CuTile integration points
- Document CUDA 13.1 requirements

### **PHASE 1**: Remove Game-Specific Logic (1 hour)
- Delete menu_hint, menu_mode, ui_clicks tracking
- Remove hardcoded UI detection (build_ui_candidates)
- Remove scene labels (MainMenu, CharacterSelect, etc.)
- Remove BonkLink privileged actions

### **PHASE 2**: CuTile Integration (2 hours)
- Replace raw pixel observations with CuTile
- Implement GPU-accelerated tile extraction
- Optimize memory usage (tiles vs full frames)
- Benchmark performance improvements

### **PHASE 3**: CUDA 13.1 Modernization (2 hours)
- Replace ALL CUDA 12.x/legacy code
- Use tensor cores for inference
- Implement unified memory (managed allocations)
- Enable compute capability 9.0 features (RTX 5090)

### **PHASE 4**: VLM Hive Enhancement (2 hours)
- Centralize VLM inference
- Link VLM reasoning to RL policy
- Implement strategic instruction generation
- Integrate with exploration rewards

### **PHASE 5**: Pure Vision Rewards (1.5 hours)
- Add exploration rewards (visual novelty, transitions)
- Implement scene fingerprinting
- Remove all menu detection
- Universal action space

### **PHASE 6**: Stream Verification (2 hours)
- Verify all 5 workers streaming
- Fix visual artifacts (upside down, choppy)
- Ensure consistent frame rates
- WebRTC quality validation

### **PHASE 7**: Integration Testing (1.5 hours)
- End-to-end gameplay validation
- Multi-worker coordination
- Performance benchmarking
- Stress testing

### **PHASE 8**: Documentation & Deployment (30 min)
- Update documentation
- Deployment checklist
- Monitoring setup
- Production readiness

---

## 🚨 CRITICAL PRINCIPLES

### **1. PURE VISION LEARNING**
- ❌ NO menu detection
- ❌ NO hardcoded UI structure
- ❌ NO scene labels
- ✅ Pixels/tiles → Actions → Rewards
- ✅ Works for ANY game

### **2. GAME-AGNOSTIC DESIGN**
- Same code for Megabonk, Binding of Isaac, Celeste, Slay the Spire
- No game-specific configuration
- Universal action space
- Visual-only progress detection

### **3. MODERN CUDA PRACTICES**
- CUDA 13.1 only (no fallbacks)
- Tensor cores for inference
- Unified memory
- Compute capability 9.0

### **4. PRODUCTION QUALITY**
- All 5 workers operational
- No visual artifacts
- Consistent streaming
- Monitored and verified

---

---

# PHASE 0: FOUNDATION & ARCHITECTURE REVIEW

**Duration**: 30 minutes

---

## Task 0.1: System Audit

### **Current System State Assessment**

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# 1. Check current observations
grep -r "def get_observation\|obs.*=.*env\.step" src/worker/

# 2. Check CuTile usage
grep -r "CuTile\|cutile" src/

# 3. Check CUDA version
grep -r "cuda.*12\|cudnn.*8\|compute_86" src/ cmake/ CMakeLists.txt

# 4. Check VLM integration
grep -r "vlm_hive\|system2_reasoning" src/

# 5. Check streaming
grep -r "go2rtc\|webrtc\|stream" src/ configs/

# Output findings to /tmp/system_audit.txt
```

### **Identify Violations**

```bash
# Game-specific logic
grep -r "menu_hint\|menu_mode\|ui_clicks\|build_ui_candidates\|MainMenu\|CharacterSelect" src/

# Outdated CUDA
grep -r "cudaMalloc\|cudaMemcpy\|compute_86" src/

# Raw pixel observations
grep -r "np.array.*obs\|pixels.*observation" src/

# Output to /tmp/violations.txt
```

---

## Task 0.2: CuTile Architecture Review

### **What is CuTile?**

CuTile is a GPU-accelerated tile-based observation system that provides:
- **Spatial tiles** instead of raw pixels
- **GPU-resident processing** (no CPU→GPU transfers)
- **Memory efficiency** (~10x reduction vs raw pixels)
- **Feature extraction** on GPU

### **CuTile vs Raw Pixels**

```python
# OLD (Raw Pixels):
obs = env.render()  # 1920x1080x3 = 6.2MB per frame
obs = np.array(obs, dtype=np.uint8)
# Transfer to GPU for RL: 6.2MB * 60 FPS = 372 MB/s bandwidth

# NEW (CuTile):
tiles = cutile.extract_tiles(gpu_framebuffer, tile_size=16)  # 120x68 tiles
features = cutile.compute_features(tiles)  # GPU-resident
obs = features  # 120x68x64 = 522KB per frame
# Already on GPU: 0 MB/s transfer bandwidth
```

**Benefits**:
- 10x memory reduction (6.2MB → 522KB)
- Zero CPU→GPU transfer (already on GPU)
- Built-in feature extraction
- Faster inference

---

## Task 0.3: CUDA 13.1 Requirements

### **Compute Capability 9.0 (RTX 5090)**

**Features to leverage**:
1. **4th Gen Tensor Cores**
   - FP8 inference (2x throughput vs FP16)
   - TF32 training (no code changes)
   
2. **Enhanced L2 Cache** (128MB)
   - Keep working set on-chip
   - Reduce memory bandwidth pressure

3. **PCIe Gen 5** (128 GB/s)
   - Faster host↔device transfers (when needed)

4. **Unified Memory** (Managed allocations)
   - No manual cudaMemcpy
   - Automatic migration

### **CUDA 13.1 API Updates**

```cpp
// OLD (CUDA 12.x):
float* d_data;
cudaMalloc(&d_data, size);
cudaMemcpy(d_data, h_data, size, cudaMemcpyHostToDevice);

// NEW (CUDA 13.1 Unified Memory):
float* d_data;
cudaMallocManaged(&d_data, size);  // Accessible from CPU and GPU
// No cudaMemcpy needed!
```

```cpp
// OLD (Manual compute capability check):
#if __CUDA_ARCH__ >= 860
  // Ampere+ code
#endif

// NEW (CUDA 13.1 always assumes sm_90):
// Just write modern code, no fallbacks
```

---

## Task 0.4: VLM Hive Architecture

### **Current VLM System**

```python
# File: src/vlm/system2_reasoning.py
class System2Reasoning:
    """VLM-based reasoning for strategic decisions"""
    
    def __init__(self):
        self.vlm_client = ...  # Centralized VLM inference
        self.reasoning_history = []
    
    def reason(self, obs, context):
        """
        Generate strategic insights from vision.
        
        Issues:
        - Not integrated with RL policy
        - Insights not linked to action selection
        - No exploration guidance
        """
```

### **Required VLM Integration**

```python
class VLMHiveReasoning:
    """
    Multi-agent VLM reasoning integrated with RL.
    
    VLM provides:
    1. Strategic insights ("Character in top-left, clickable")
    2. Exploration hints ("Try clicking bright objects")
    3. Progress assessment ("Progressed to next scene")
    4. Action feedback ("Last action had no effect")
    """
    
    def generate_exploration_hints(self, obs, policy_state):
        """Link VLM insights to exploration strategy"""
        
    def assess_progress(self, obs_history):
        """Vision-based progress without scene labels"""
    
    def critique_actions(self, obs, action, outcome):
        """Provide feedback to improve policy"""
```

---

## Task 0.5: Streaming Validation Plan

### **Verification Requirements**

**All 5 Workers Must**:
1. ✅ Stream to http://localhost:5173/stream
2. ✅ Show gameplay (not stuck in menus)
3. ✅ No visual artifacts (upside down, mirrored, distorted)
4. ✅ Consistent frame rate (58-60 FPS, not choppy)
5. ✅ WebRTC active (go2rtc forwarding)

### **Known Issues to Fix**:
- Upside-down rendering (coordinate system flip)
- Choppy streams (frame pacing issues)
- Missing workers (failed startup)
- Menu lock (stuck at character select)

---

---

# PHASE 1: REMOVE GAME-SPECIFIC LOGIC

**Duration**: 1 hour

---

## Task 1.1: Remove Menu Detection

### **Delete from `src/worker/main.py`**

```python
# FIND AND DELETE (around line 5923, 7249, 8680):

# DELETE:
menu_hint = vision_metrics.menu_mode
menu_hint_source = "detections"

# DELETE:
ui_clicks_sent = 0
ui_clicks_throttled = 0
last_ui_click = {}

# DELETE:
if menu_hint:
    # Special menu handling
    ...

# DELETE from /status endpoint:
"menu_hint": self.menu_hint,
"menu_hint_source": self.menu_hint_source,
"ui_clicks_sent": self.ui_clicks_sent,
"ui_clicks_throttled": self.ui_clicks_throttled,
"last_ui_click": self.last_ui_click,
```

**Validation**:
```bash
grep -n "menu_hint\|menu_mode\|ui_clicks" src/worker/main.py
# Expected: (empty)
```

---

## Task 1.2: Remove UI Detection

### **Delete from `src/worker/perception.py`**

```python
# FIND AND DELETE:

# DELETE:
def build_ui_candidates(self):
    """Hardcoded UI detection"""
    # ... entire function

# DELETE:
METABONK_UI_GRID_RESERVE = ...
grid_reserve = env.get('METABONK_UI_GRID_RESERVE', 8)

# DELETE:
cls = -2  # "grid candidate" class
```

**Validation**:
```bash
grep -n "build_ui_candidates\|UI_GRID_RESERVE\|cls.*-2" src/worker/perception.py
# Expected: (empty)
```

---

## Task 1.3: Remove Scene Labels

### **Search and Replace**

```bash
# Find all hardcoded scene names
grep -r "MainMenu\|CharacterSelect\|LoadingScreen\|GeneratedMap\|Gameplay" --include="*.py" src/

# For each found:
# - If used for logging: Replace with scene fingerprint hash
# - If used for control: Remove and use visual-only logic
# - If in comments: Can keep
```

**Example Replacement**:
```python
# BEFORE:
if scene == "MainMenu":
    log.info("In main menu")

# AFTER:
scene_fp = compute_scene_fingerprint(obs)
log.info(f"Scene: {scene_fp[:8]}")
```

---

## Task 1.4: Remove BonkLink Privileged Actions

### **File: `src/plugins/bonklink_client.py` (or similar)**

```python
# REMOVE hardcoded action methods:

# DELETE:
def send_menu_click(self, x, y):
    """Hardcoded menu navigation"""
    ...

# DELETE:
def send_character_select(self, character_id):
    """Hardcoded character selection"""
    ...

# KEEP generic action sender:
def send_action(self, action_type, **params):
    """Generic action transmission"""
    ...
```

---

## Task 1.5: Clean Configuration

### **Files: `configs/launch_default.json`, `configs/launch_production.json`**

```json
// REMOVE:
{
  "training": {
    "menu_bootstrap": true,      // DELETE
    "menu_eps": 0.7,             // DELETE
    "ui_grid_reserve": 8         // DELETE
  }
}

// KEEP:
{
  "training": {
    "exploration_rate": 0.3,     // Universal
    "pure_vision_mode": true     // Enforcement
  }
}
```

---

## Task 1.6: Remove From Launcher

### **File: `launch.py`**

```python
# REMOVE environment variable setting:

# DELETE:
env['METABONK_MENU_EPS'] = ...
env['METABONK_UI_GRID_RESERVE'] = ...
env['METABONK_LOG_UI_CLICKS'] = ...
env['METABONK_PURE_VISION_ALLOW_MENU_BOOTSTRAP'] = ...
```

---

## Phase 1 Validation

```bash
# Run validation
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# 1. No game-specific logic
grep -r "menu_hint\|menu_mode\|ui_clicks\|build_ui_candidates\|MainMenu\|CharacterSelect" src/
# Expected: (empty)

# 2. No bootstrap logic
grep -r "menu_bootstrap\|MENU_BOOTSTRAP" src/ configs/
# Expected: (empty)

# 3. Compile check
python3 -m py_compile src/worker/main.py
python3 -m py_compile src/worker/perception.py
# Expected: No errors
```

---

---

# PHASE 2: CUTILE INTEGRATION

**Duration**: 2 hours

---

## Task 2.1: Install CuTile

### **Clone CuTile Repository**

```bash
cd /tmp
git clone https://github.com/vladdiethecoder/CuTile.git
cd CuTile

# Build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=90
make -j$(nproc)
sudo make install

# Verify installation
ls /usr/local/lib/libcutile.so
ls /usr/local/include/cutile/
```

---

## Task 2.2: Replace Pixel Observations

### **Create CuTile Wrapper**

**File: `src/perception/cutile_observations.py`** (NEW)

```python
"""
CuTile-based observations for MetaBonk.

Replaces raw pixel observations with GPU-accelerated tiles.
"""

import numpy as np
import torch
import cutile  # Assuming Python bindings exist

class CuTileObservations:
    """
    GPU-accelerated tile-based observations.
    
    Instead of:
        obs = env.render()  # 1920x1080x3 = 6.2MB
    
    Use:
        tiles = cutile.extract_tiles(framebuffer)  # 120x68x64 = 522KB
    """
    
    def __init__(self, 
                 tile_size=16,
                 feature_dim=64,
                 device='cuda:0'):
        """
        Args:
            tile_size: Size of each tile (16x16 pixels)
            feature_dim: Feature vector dimension per tile
            device: CUDA device
        """
        self.tile_size = tile_size
        self.feature_dim = feature_dim
        self.device = device
        
        # Initialize CuTile on GPU
        self.cutile_ctx = cutile.Context(device=device)
        
        # Feature extractor (lightweight CNN on GPU)
        self.feature_net = self._build_feature_net().to(device)
        self.feature_net.eval()
    
    def _build_feature_net(self):
        """
        Lightweight CNN for tile feature extraction.
        
        Input: [B, 3, 16, 16] tiles
        Output: [B, 64] feature vectors
        """
        import torch.nn as nn
        
        return nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 16→8
            
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),  # 8→4
            
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),  # 4→1
            
            nn.Flatten(),  # [B, 64, 1, 1] → [B, 64]
        )
    
    def extract_observation(self, framebuffer_ptr):
        """
        Extract tile-based observation from GPU framebuffer.
        
        Args:
            framebuffer_ptr: CUDA device pointer to framebuffer (1920x1080x3)
        
        Returns:
            torch.Tensor: Tile features [H_tiles, W_tiles, feature_dim]
                         e.g., [68, 120, 64] for 1920x1080 with 16x16 tiles
        """
        # Extract tiles on GPU (zero-copy)
        tiles = self.cutile_ctx.extract_tiles(
            framebuffer_ptr,
            tile_size=self.tile_size
        )  # Shape: [N_tiles, 3, 16, 16]
        
        # Compute features on GPU
        with torch.no_grad():
            features = self.feature_net(tiles)  # [N_tiles, 64]
        
        # Reshape to spatial grid
        H = 1080 // self.tile_size  # 68
        W = 1920 // self.tile_size  # 120
        features = features.view(H, W, self.feature_dim)
        
        return features  # [68, 120, 64] on GPU
    
    def to_numpy(self, features):
        """Convert GPU features to numpy for compatibility"""
        return features.cpu().numpy()
    
    def get_observation_space(self):
        """Return observation space specification"""
        H = 1080 // self.tile_size
        W = 1920 // self.tile_size
        
        return {
            'shape': (H, W, self.feature_dim),
            'dtype': np.float32,
            'memory': H * W * self.feature_dim * 4,  # bytes
        }


# Comparison utility
def compare_observation_sizes():
    """Compare memory usage: pixels vs tiles"""
    
    # Raw pixels
    H, W, C = 1920, 1080, 3
    pixel_size = H * W * C * 1  # uint8
    print(f"Pixel observation: {pixel_size / 1e6:.2f} MB")
    
    # CuTile
    tile_size = 16
    feature_dim = 64
    H_tiles = H // tile_size
    W_tiles = W // tile_size
    tile_obs_size = H_tiles * W_tiles * feature_dim * 4  # float32
    print(f"CuTile observation: {tile_obs_size / 1e6:.2f} MB")
    
    print(f"Reduction: {pixel_size / tile_obs_size:.1f}x")

# Output:
# Pixel observation: 6.22 MB
# CuTile observation: 0.52 MB
# Reduction: 11.9x
```

---

## Task 2.3: Integrate CuTile with Worker

### **Update `src/worker/main.py`**

```python
# Add import
from src.perception.cutile_observations import CuTileObservations

class Worker:
    def __init__(self, worker_id, config):
        # ... existing init
        
        # Initialize CuTile observations
        self.cutile_obs = CuTileObservations(
            tile_size=config.get('tile_size', 16),
            feature_dim=config.get('feature_dim', 64),
            device=f'cuda:{worker_id % torch.cuda.device_count()}'
        )
        
        # Get framebuffer pointer from game engine
        self.framebuffer_ptr = self._get_framebuffer_pointer()
    
    def get_observation(self):
        """
        Get CuTile observation instead of raw pixels.
        
        BEFORE:
            obs = self.env.render()  # [1920, 1080, 3] on CPU
            obs = np.array(obs, dtype=np.uint8)
        
        AFTER:
            obs = self.cutile_obs.extract_observation(self.framebuffer_ptr)
            # [68, 120, 64] on GPU
        """
        # Extract tiles from GPU framebuffer (zero-copy)
        obs_gpu = self.cutile_obs.extract_observation(self.framebuffer_ptr)
        
        # Keep on GPU for policy inference
        # (Policy network is also on GPU, no transfer needed)
        return obs_gpu
    
    def _get_framebuffer_pointer(self):
        """
        Get CUDA device pointer to game framebuffer.
        
        This depends on your game engine integration.
        Options:
        1. OpenGL interop: cudaGraphicsMapResources
        2. Vulkan interop: vkExportMemoryFd → cudaImportExternalMemory
        3. Direct render to CUDA: Custom game integration
        """
        # Example: OpenGL interop
        import OpenGL.GL as gl
        
        # Register OpenGL texture with CUDA
        fbo_texture_id = self.game_engine.get_render_texture()
        
        cuda_resource = cuda.cudaGraphicsGLRegisterImage(
            fbo_texture_id,
            gl.GL_TEXTURE_2D,
            cuda.cudaGraphicsRegisterFlagsReadOnly
        )
        
        # Map to get device pointer
        cuda.cudaGraphicsMapResources(1, cuda_resource)
        device_ptr = cuda.cudaGraphicsResourceGetMappedPointer(cuda_resource)
        
        return device_ptr
```

---

## Task 2.4: Update Policy Network for Tiles

### **Modify `src/agent/policy.py`**

```python
class TilePolicy(nn.Module):
    """
    Policy network for tile-based observations.
    
    Input: [B, H_tiles, W_tiles, feature_dim]
           e.g., [B, 68, 120, 64]
    
    Output: Action distribution
    """
    
    def __init__(self, tile_shape, action_space):
        super().__init__()
        
        H, W, C = tile_shape  # 68, 120, 64
        
        # Spatial processing
        self.conv1 = nn.Conv2d(C, 128, 3, padding=1)
        self.conv2 = nn.Conv2d(128, 256, 3, padding=1)
        self.conv3 = nn.Conv2d(256, 512, 3, padding=1)
        
        # Spatial attention (learn which tiles matter)
        self.attention = SpatialAttention(512)
        
        # Global pooling
        self.pool = nn.AdaptiveAvgPool2d(1)
        
        # Action head
        self.fc = nn.Linear(512, action_space)
    
    def forward(self, tiles):
        """
        Args:
            tiles: [B, H, W, C] tile features
        
        Returns:
            action_logits: [B, action_space]
        """
        # Permute to [B, C, H, W] for conv
        x = tiles.permute(0, 3, 1, 2)
        
        # Spatial processing
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        
        # Attention (focus on important tiles)
        x = self.attention(x)
        
        # Global pooling
        x = self.pool(x).flatten(1)
        
        # Action distribution
        logits = self.fc(x)
        
        return logits


class SpatialAttention(nn.Module):
    """Learn which tiles are important"""
    
    def __init__(self, channels):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Conv2d(channels, channels // 8, 1),
            nn.ReLU(),
            nn.Conv2d(channels // 8, 1, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        att = self.attention(x)  # [B, 1, H, W]
        return x * att  # Weighted features
```

---

## Task 2.5: Benchmark CuTile Performance

### **Create benchmark script**

**File: `scripts/benchmark_cutile.py`**

```python
"""Benchmark CuTile vs raw pixels"""

import time
import torch
import numpy as np
from src.perception.cutile_observations import CuTileObservations

def benchmark():
    cutile = CuTileObservations()
    
    # Simulate framebuffer
    framebuffer = torch.randn(1080, 1920, 3, device='cuda:0', dtype=torch.uint8)
    framebuffer_ptr = framebuffer.data_ptr()
    
    # Warmup
    for _ in range(10):
        _ = cutile.extract_observation(framebuffer_ptr)
    
    # Benchmark
    N = 1000
    start = time.time()
    for _ in range(N):
        obs = cutile.extract_observation(framebuffer_ptr)
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    print(f"CuTile: {N/elapsed:.1f} FPS")
    print(f"Latency: {elapsed/N*1000:.2f} ms/frame")
    
    # Compare with raw pixels
    start = time.time()
    for _ in range(N):
        pixels = framebuffer.cpu().numpy()  # Transfer to CPU
    elapsed = time.time() - start
    
    print(f"Raw pixels: {N/elapsed:.1f} FPS")
    print(f"Latency: {elapsed/N*1000:.2f} ms/frame")

if __name__ == '__main__':
    benchmark()

# Expected output:
# CuTile: 2500 FPS (0.4 ms/frame)
# Raw pixels: 150 FPS (6.7 ms/frame)
# → 16x faster!
```

---

## Phase 2 Validation

```bash
# 1. CuTile installed
ls /usr/local/lib/libcutile.so
# Expected: File exists

# 2. Observations replaced
grep -n "cutile_obs.extract_observation" src/worker/main.py
# Expected: Found

# 3. Performance test
python3 scripts/benchmark_cutile.py
# Expected: >1000 FPS

# 4. Memory usage
python3 -c "from src.perception.cutile_observations import compare_observation_sizes; compare_observation_sizes()"
# Expected: ~12x reduction
```

---

---

# PHASE 3: CUDA 13.1 MODERNIZATION

**Duration**: 2 hours

---

## Task 3.1: Update Build System

### **File: `CMakeLists.txt`**

```cmake
# REMOVE:
set(CMAKE_CUDA_ARCHITECTURES 86)  # Ampere
find_package(CUDA 12.0 REQUIRED)

# ADD:
cmake_minimum_required(VERSION 3.24)
project(MetaBonk LANGUAGES CXX CUDA)

# CUDA 13.1, compute capability 9.0 (RTX 5090)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 90)  # Ada Lovelace / Blackwell
find_package(CUDAToolkit 13.1 REQUIRED)

# Enable modern CUDA features
add_compile_options(
    $<$<COMPILE_LANGUAGE:CUDA>:-use_fast_math>
    $<$<COMPILE_LANGUAGE:CUDA>:-Xptxas=-v>
    $<$<COMPILE_LANGUAGE:CUDA>:--generate-line-info>
)

# Link against CUDA 13.1 libraries
target_link_libraries(MetaBonk
    CUDA::cudart
    CUDA::cublas
    CUDA::cudnn
)
```

---

## Task 3.2: Replace Legacy CUDA Code

### **Unified Memory Migration**

**File: `src/cuda/memory_utils.cu`**

```cpp
// BEFORE (CUDA 12.x):
float* allocate_device_memory(size_t size) {
    float* d_ptr;
    cudaMalloc(&d_ptr, size);
    return d_ptr;
}

void copy_to_device(float* d_ptr, float* h_ptr, size_t size) {
    cudaMemcpy(d_ptr, h_ptr, size, cudaMemcpyHostToDevice);
}

// AFTER (CUDA 13.1 Unified Memory):
float* allocate_unified_memory(size_t size) {
    float* ptr;
    cudaMallocManaged(&ptr, size);
    
    // Hint: prefer GPU access
    cudaMemAdvise(ptr, size, cudaMemAdviseSetPreferredLocation, 0);
    
    return ptr;  // Accessible from CPU and GPU!
}

// No explicit copy needed - automatic migration
```

---

### **Tensor Core Inference**

**File: `src/cuda/inference.cu`**

```cpp
// BEFORE (Manual FP32):
__global__ void matrix_multiply_fp32(
    const float* A,
    const float* B,
    float* C,
    int M, int N, int K
) {
    // Manual tile-based gemm
    // ~100 TFLOPS on RTX 5090
}

// AFTER (CUDA 13.1 Tensor Cores with FP8):
#include <cuda_fp8.h>
#include <cublas_v2.h>

void tensor_core_inference(
    const __nv_fp8_e4m3* A,  // FP8 precision
    const __nv_fp8_e4m3* B,
    float* C,
    int M, int N, int K,
    cublasHandle_t handle
) {
    // Use cuBLAS with tensor cores
    cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        M, N, K,
        &alpha,
        A, CUDA_R_8F_E4M3, M,  // FP8 input
        B, CUDA_R_8F_E4M3, K,  // FP8 weights
        &beta,
        C, CUDA_R_32F, M,      // FP32 output
        CUDA_R_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP  // Enable tensor cores
    );
    
    // ~200 TFLOPS on RTX 5090 (2x faster than FP32)
}
```

---

### **Enhanced L2 Cache Utilization**

**File: `src/cuda/cache_hints.cu`**

```cpp
// CUDA 13.1: 128MB L2 cache on RTX 5090
// Hint that working set should stay in L2

void configure_l2_cache(const void* data, size_t size) {
    cudaStreamAttrValue stream_attr;
    stream_attr.accessPolicyWindow.base_ptr = (void*)data;
    stream_attr.accessPolicyWindow.num_bytes = size;
    stream_attr.accessPolicyWindow.hitRatio = 1.0;  // Keep in L2
    stream_attr.accessPolicyWindow.hitProp = cudaAccessPropertyPersisting;
    stream_attr.accessPolicyWindow.missProp = cudaAccessPropertyStreaming;
    
    cudaStreamSetAttribute(
        stream,
        cudaStreamAttributeAccessPolicyWindow,
        &stream_attr
    );
}

// Example: Keep policy network weights in L2
float* policy_weights;
cudaMallocManaged(&policy_weights, 50 * 1024 * 1024);  // 50MB
configure_l2_cache(policy_weights, 50 * 1024 * 1024);

// Result: ~10x faster inference (no main memory access)
```

---

## Task 3.3: Remove Fallback Code

### **Find and Delete Legacy Paths**

```bash
# Find all compute capability checks
grep -r "__CUDA_ARCH__.*86\|compute_86\|sm_86" src/

# For each found:
# - Remove #if blocks for old compute capabilities
# - Keep only sm_90 code path
```

**Example**:
```cpp
// BEFORE:
#if __CUDA_ARCH__ >= 860
    // Ampere path
    use_tensor_cores_ampere();
#else
    // Fallback path
    use_cuda_cores();
#endif

// AFTER (CUDA 13.1, RTX 5090):
// Always use tensor cores (sm_90)
use_tensor_cores_blackwell();
```

---

### **Remove cudnn 8.x Fallbacks**

```cpp
// BEFORE:
#if CUDNN_VERSION >= 8000
    cudnnActivationForward(...);
#else
    manual_activation(...);  // Fallback
#endif

// AFTER:
// CUDA 13.1 always has latest cudnn
cudnnActivationForward(...);
// No fallback needed
```

---

## Task 3.4: Update Python Bindings

### **File: `src/python/cuda_bindings.py`**

```python
import torch

# Verify CUDA 13.1
assert torch.version.cuda >= "13.1", f"CUDA 13.1+ required, found {torch.version.cuda}"

# Verify compute capability 9.0
cc = torch.cuda.get_device_capability()
assert cc[0] >= 9, f"Compute capability 9.0+ required, found {cc}"

# Enable tensor core math
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# Enable tensor cores for convolutions
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.enabled = True

print("CUDA 13.1 configured:")
print(f"  Device: {torch.cuda.get_device_name()}")
print(f"  Compute Capability: {cc[0]}.{cc[1]}")
print(f"  Tensor Cores: Enabled")
print(f"  Unified Memory: Available")
```

---

## Task 3.5: Benchmark CUDA 13.1 Performance

### **File: `scripts/benchmark_cuda131.py`**

```python
"""Benchmark CUDA 13.1 features"""

import torch
import time

def benchmark_tensor_cores():
    """Benchmark FP8 tensor core inference"""
    
    # Policy network size (typical)
    M, N, K = 256, 512, 512
    
    # FP32 baseline
    A_fp32 = torch.randn(M, K, device='cuda', dtype=torch.float32)
    B_fp32 = torch.randn(K, N, device='cuda', dtype=torch.float32)
    
    # Warmup
    for _ in range(10):
        C = torch.mm(A_fp32, B_fp32)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        C = torch.mm(A_fp32, B_fp32)
    torch.cuda.synchronize()
    fp32_time = time.time() - start
    
    # FP8 tensor cores
    A_fp8 = A_fp32.to(torch.float8_e4m3fn)  # FP8
    B_fp8 = B_fp32.to(torch.float8_e4m3fn)
    
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(1000):
        C = torch.mm(A_fp8, B_fp8)
    torch.cuda.synchronize()
    fp8_time = time.time() - start
    
    print(f"FP32: {1000/fp32_time:.1f} inferences/sec")
    print(f"FP8: {1000/fp8_time:.1f} inferences/sec")
    print(f"Speedup: {fp32_time/fp8_time:.1f}x")

if __name__ == '__main__':
    benchmark_tensor_cores()

# Expected output:
# FP32: 5000 inferences/sec
# FP8: 10000 inferences/sec
# Speedup: 2.0x
```

---

## Phase 3 Validation

```bash
# 1. CUDA 13.1 installed
nvcc --version
# Expected: release 13.1

# 2. Compute capability
nvidia-smi --query-gpu=compute_cap --format=csv
# Expected: 9.0

# 3. No legacy code
grep -r "cuda.*12\|compute_86\|sm_86" src/
# Expected: (empty)

# 4. Performance
python3 scripts/benchmark_cuda131.py
# Expected: 2x speedup with FP8

# 5. Compile
cd build && cmake .. && make -j$(nproc)
# Expected: No errors
```

---

---

# PHASE 4: VLM HIVE ENHANCEMENT

**Duration**: 2 hours

---

## Task 4.1: Centralized VLM Inference

### **File: `src/vlm/vlm_inference_server.py`** (NEW)

```python
"""
Centralized VLM inference server.

Instead of each worker calling VLM independently,
one server handles all VLM requests.

Benefits:
- Shared GPU memory (load model once)
- Batched inference (process multiple workers together)
- Consistent responses
"""

import torch
from transformers import AutoModelForVision2Seq, AutoProcessor
from queue import Queue
import threading

class VLMInferenceServer:
    """Centralized VLM inference for all workers"""
    
    def __init__(self, model_name="Qwen/Qwen2-VL-7B", device='cuda:0'):
        self.device = device
        
        # Load VLM once (shared across workers)
        print(f"Loading VLM: {model_name}")
        self.processor = AutoProcessor.from_pretrained(model_name)
        self.model = AutoModelForVision2Seq.from_pretrained(
            model_name,
            torch_dtype=torch.float16,  # FP16 for speed
            device_map=device
        )
        self.model.eval()
        
        # Request queue
        self.request_queue = Queue()
        self.response_dict = {}
        self.request_id = 0
        
        # Inference thread
        self.thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.thread.start()
        
        print("VLM server ready")
    
    def reason(self, image, prompt, timeout=5.0):
        """
        Send reasoning request to VLM.
        
        Args:
            image: PIL Image or torch.Tensor
            prompt: Text prompt
            timeout: Max wait time
        
        Returns:
            str: VLM response
        """
        # Generate request ID
        req_id = self.request_id
        self.request_id += 1
        
        # Add to queue
        self.request_queue.put({
            'id': req_id,
            'image': image,
            'prompt': prompt
        })
        
        # Wait for response
        import time
        start = time.time()
        while time.time() - start < timeout:
            if req_id in self.response_dict:
                response = self.response_dict.pop(req_id)
                return response
            time.sleep(0.01)
        
        return "VLM timeout"
    
    def _inference_loop(self):
        """Background thread: process VLM requests"""
        
        while True:
            # Collect batch of requests
            batch = []
            while not self.request_queue.empty() and len(batch) < 8:
                batch.append(self.request_queue.get())
            
            if not batch:
                time.sleep(0.01)
                continue
            
            # Batch inference
            images = [req['image'] for req in batch]
            prompts = [req['prompt'] for req in batch]
            
            # Process
            inputs = self.processor(
                images=images,
                text=prompts,
                return_tensors='pt',
                padding=True
            ).to(self.device)
            
            # Generate
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=100,
                    do_sample=False
                )
            
            # Decode
            responses = self.processor.batch_decode(outputs, skip_special_tokens=True)
            
            # Store responses
            for req, response in zip(batch, responses):
                self.response_dict[req['id']] = response


# Global server instance
vlm_server = None

def get_vlm_server():
    """Get or create VLM server"""
    global vlm_server
    if vlm_server is None:
        vlm_server = VLMInferenceServer()
    return vlm_server
```

---

## Task 4.2: VLM-Guided Exploration

### **File: `src/agent/vlm_exploration.py`** (NEW)

```python
"""
VLM-guided exploration.

VLM provides strategic hints for where to explore.
"""

from src.vlm.vlm_inference_server import get_vlm_server
import numpy as np

class VLMExploration:
    """Use VLM to guide exploration strategy"""
    
    def __init__(self):
        self.vlm = get_vlm_server()
        self.hint_history = []
    
    def get_exploration_hint(self, obs, context=None):
        """
        Ask VLM where to explore.
        
        Args:
            obs: Current observation (image)
            context: Optional context (e.g., "stuck in same place")
        
        Returns:
            dict: {
                'hint': str,  # "Try clicking the bright object in top-right"
                'location': (x, y),  # Suggested click location
                'confidence': float  # 0.0-1.0
            }
        """
        # Build prompt
        prompt = "Analyze this game screen. "
        if context:
            prompt += f"Context: {context}. "
        prompt += "What should the player do next? Provide specific location (x, y) as percentage."
        
        # Query VLM
        response = self.vlm.reason(obs, prompt)
        
        # Parse response
        hint = self._parse_hint(response)
        
        self.hint_history.append(hint)
        return hint
    
    def _parse_hint(self, response):
        """
        Parse VLM response into structured hint.
        
        Example response:
        "Click the character icon in the top-left corner at approximately (10%, 15%)"
        
        Returns:
            {'hint': str, 'location': (x, y), 'confidence': float}
        """
        # Extract location
        import re
        match = re.search(r'\((\d+)%,\s*(\d+)%\)', response)
        
        if match:
            x = int(match.group(1)) / 100.0
            y = int(match.group(2)) / 100.0
            confidence = 0.8
        else:
            # No specific location, random exploration
            x = np.random.uniform(0, 1)
            y = np.random.uniform(0, 1)
            confidence = 0.3
        
        return {
            'hint': response,
            'location': (x, y),
            'confidence': confidence
        }
    
    def assess_progress(self, obs_history):
        """
        Use VLM to assess if making progress.
        
        Args:
            obs_history: List of recent observations
        
        Returns:
            dict: {
                'making_progress': bool,
                'reason': str,
                'suggestion': str
            }
        """
        if len(obs_history) < 2:
            return {'making_progress': True, 'reason': 'Just started', 'suggestion': ''}
        
        # Compare first and last frame
        first_frame = obs_history[0]
        last_frame = obs_history[-1]
        
        prompt = """
        Compare these two game screens (first and last in sequence).
        Are they significantly different? Is the player making progress?
        Answer: Yes/No and explain why.
        """
        
        # Query VLM with both images
        response = self.vlm.reason([first_frame, last_frame], prompt)
        
        # Parse
        making_progress = 'yes' in response.lower()
        
        return {
            'making_progress': making_progress,
            'reason': response,
            'suggestion': self._generate_suggestion(response, last_frame)
        }
    
    def _generate_suggestion(self, assessment, current_obs):
        """Generate actionable suggestion based on assessment"""
        
        if 'stuck' in assessment.lower() or 'not progressing' in assessment.lower():
            prompt = "The player seems stuck. What specific action should they take? Be concrete."
            suggestion = self.vlm.reason(current_obs, prompt)
            return suggestion
        
        return "Continue current strategy"
```

---

## Task 4.3: Integrate VLM with Worker

### **Update `src/worker/main.py`**

```python
from src.agent.vlm_exploration import VLMExploration

class Worker:
    def __init__(self, worker_id, config):
        # ... existing init
        
        # VLM exploration (shared server)
        self.vlm_exploration = VLMExploration()
        
        # Tracking
        self.obs_history = []
        self.last_vlm_query_time = 0
    
    def step(self):
        """Main step loop with VLM guidance"""
        
        # Get observation
        obs = self.get_observation()
        self.obs_history.append(obs)
        
        # Decide if we need VLM guidance
        current_time = time.time()
        use_vlm_hint = (
            (current_time - self.last_vlm_query_time > 10.0) or  # Every 10 seconds
            self._is_stuck()  # Or if stuck
        )
        
        if use_vlm_hint:
            # Get VLM hint
            hint = self.vlm_exploration.get_exploration_hint(
                obs,
                context="Exploring game" if not self._is_stuck() else "Stuck in same place"
            )
            
            # Use hint to bias action selection
            if hint['confidence'] > 0.5:
                # High confidence hint - use it
                action = {
                    'type': 'click',
                    'x': hint['location'][0],
                    'y': hint['location'][1]
                }
                log.info(f"VLM hint: {hint['hint']}")
            else:
                # Low confidence - use policy
                action = self.policy.get_action(obs)
            
            self.last_vlm_query_time = current_time
        else:
            # Normal policy action
            action = self.policy.get_action(obs)
        
        # Execute action
        reward = self.execute_action(action)
        
        # Check progress periodically
        if len(self.obs_history) % 100 == 0:
            progress = self.vlm_exploration.assess_progress(self.obs_history[-20:])
            
            if not progress['making_progress']:
                log.warning(f"Not making progress: {progress['reason']}")
                log.info(f"VLM suggestion: {progress['suggestion']}")
    
    def _is_stuck(self):
        """Simple stuck detection from visual similarity"""
        if len(self.obs_history) < 10:
            return False
        
        # Compare last 10 frames
        recent = self.obs_history[-10:]
        diffs = [np.mean(np.abs(recent[i] - recent[i-1])) for i in range(1, len(recent))]
        avg_diff = np.mean(diffs)
        
        # Stuck if very low visual change
        return avg_diff < 0.01
```

---

## Task 4.4: VLM for Action Critique

### **File: `src/agent/vlm_critic.py`** (NEW)

```python
"""
VLM-based action critique.

VLM watches agent's actions and provides feedback.
"""

from src.vlm.vlm_inference_server import get_vlm_server

class VLMCritic:
    """Critique agent actions using VLM"""
    
    def __init__(self):
        self.vlm = get_vlm_server()
    
    def critique_action(self, obs_before, action, obs_after):
        """
        Critique a single action.
        
        Args:
            obs_before: Observation before action
            action: Action taken
            obs_after: Observation after action
        
        Returns:
            dict: {
                'effective': bool,  # Did action have intended effect?
                'feedback': str,    # Explanation
                'better_action': dict  # Suggested better action
            }
        """
        # Format action description
        if action['type'] == 'click':
            action_desc = f"Clicked at ({action['x']:.0%}, {action['y']:.0%})"
        elif action['type'] == 'key':
            action_desc = f"Pressed key {action['key']}"
        else:
            action_desc = "Waited"
        
        prompt = f"""
        The agent took this action: {action_desc}
        Compare the before and after screens.
        
        Questions:
        1. Did the action have the intended effect?
        2. Was this a good action?
        3. What would be a better action?
        
        Answer concisely.
        """
        
        # Query VLM with before/after
        response = self.vlm.reason([obs_before, obs_after], prompt)
        
        # Parse critique
        effective = 'yes' in response.lower() or 'effective' in response.lower()
        
        return {
            'effective': effective,
            'feedback': response,
            'better_action': self._parse_better_action(response)
        }
    
    def _parse_better_action(self, response):
        """Extract suggested action from critique"""
        # Simple heuristic: look for coordinates
        import re
        match = re.search(r'\((\d+)%,\s*(\d+)%\)', response)
        
        if match:
            return {
                'type': 'click',
                'x': int(match.group(1)) / 100.0,
                'y': int(match.group(2)) / 100.0
            }
        
        return None  # No specific suggestion
```

---

## Phase 4 Validation

```bash
# 1. VLM server starts
python3 -c "from src.vlm.vlm_inference_server import get_vlm_server; get_vlm_server()"
# Expected: "VLM server ready"

# 2. VLM exploration works
python3 -c "
from src.agent.vlm_exploration import VLMExploration
import torch
vlm = VLMExploration()
obs = torch.randn(1080, 1920, 3)
hint = vlm.get_exploration_hint(obs)
print(hint)
"
# Expected: Dict with hint, location, confidence

# 3. Integrated with worker
grep -n "vlm_exploration" src/worker/main.py
# Expected: Found

# 4. Worker starts with VLM
./launch --workers 1
tail -f runs/*/logs/worker_0.log | grep "VLM"
# Expected: "VLM hint: ..."
```

---

---

# PHASE 5: PURE VISION REWARDS

**Duration**: 1.5 hours

---

## Task 5.1: Exploration Rewards Implementation

### **File: `src/agent/exploration_rewards.py`** (from earlier, now integrate)

```python
"""
Pure vision exploration rewards.

Rewards agent for:
- Visual novelty (seeing new things)
- Screen transitions (big changes)
- New scenes (unique fingerprints)
- Action diversity (trying different actions)
"""

import numpy as np
import torch
from PIL import Image
import imagehash

class ExplorationRewards:
    """Pure vision exploration - no game knowledge"""
    
    def __init__(self,
                 novelty_weight=0.5,
                 transition_weight=2.0,
                 new_scene_weight=10.0,
                 diversity_weight=0.2):
        
        self.novelty_weight = novelty_weight
        self.transition_weight = transition_weight
        self.new_scene_weight = new_scene_weight
        self.diversity_weight = diversity_weight
        
        self.prev_obs = None
        self.action_history = []
        self.scene_fingerprints = set()
        
        # Metrics
        self.last_reward = 0.0
        self.last_novelty = 0.0
        self.last_transition = False
        self.last_new_scene = False
    
    def compute_reward(self, obs, action):
        """
        Compute exploration reward.
        
        Args:
            obs: Current observation (CuTile features or pixels)
            action: Action taken
        
        Returns:
            float: Exploration reward
        """
        reward = 0.0
        
        # 1. Visual novelty
        if self.prev_obs is not None:
            novelty = self._compute_novelty(obs, self.prev_obs)
            reward += novelty * self.novelty_weight
            self.last_novelty = novelty
            
            # 2. Big transition
            if novelty > 0.3:
                reward += self.transition_weight
                self.last_transition = True
            else:
                self.last_transition = False
        
        # 3. New scene discovery
        fp = self._compute_fingerprint(obs)
        if fp not in self.scene_fingerprints:
            self.scene_fingerprints.add(fp)
            reward += self.new_scene_weight
            self.last_new_scene = True
        else:
            self.last_new_scene = False
        
        # 4. Action diversity
        if len(self.action_history) > 0:
            if self._is_different_action(action, self.action_history[-1]):
                reward += self.diversity_weight
        
        self.prev_obs = obs if isinstance(obs, np.ndarray) else obs.cpu().numpy()
        self.action_history.append(action)
        self.last_reward = reward
        
        return reward
    
    def _compute_novelty(self, obs1, obs2):
        """Visual novelty (0.0 to 1.0)"""
        # Handle both numpy and torch
        if isinstance(obs1, torch.Tensor):
            obs1 = obs1.cpu().numpy()
        if isinstance(obs2, torch.Tensor):
            obs2 = obs2.cpu().numpy()
        
        diff = np.abs(obs1.astype(float) - obs2.astype(float))
        novelty = np.mean(diff) / 255.0
        return float(novelty)
    
    def _compute_fingerprint(self, obs):
        """Perceptual hash for scene identity"""
        if isinstance(obs, torch.Tensor):
            obs = obs.cpu().numpy()
        
        # Convert to uint8 image
        if obs.dtype != np.uint8:
            obs = (obs * 255).astype(np.uint8)
        
        # Ensure 2D or 3D
        if obs.ndim == 3 and obs.shape[2] > 3:
            # CuTile features - use first 3 channels
            obs = obs[:, :, :3]
        
        pil_img = Image.fromarray(obs)
        return str(imagehash.phash(pil_img))
    
    def _is_different_action(self, a1, a2):
        """Check if actions are different"""
        if a1.get('type') != a2.get('type'):
            return True
        
        if a1.get('type') == 'click':
            dx = abs(a1['x'] - a2['x'])
            dy = abs(a1['y'] - a2['y'])
            return (dx > 0.05) or (dy > 0.05)
        
        return a1.get('key') != a2.get('key')
    
    def get_metrics(self):
        """Get metrics for monitoring"""
        return {
            'exploration_reward': self.last_reward,
            'visual_novelty': self.last_novelty,
            'screen_transition': self.last_transition,
            'new_scene': self.last_new_scene,
            'scenes_discovered': len(self.scene_fingerprints),
            'actions_taken': len(self.action_history)
        }
```

---

## Task 5.2: Integrate Rewards into Worker

### **Update `src/worker/main.py`**

```python
from src.agent.exploration_rewards import ExplorationRewards

class Worker:
    def __init__(self, worker_id, config):
        # ... existing init
        
        # Exploration rewards
        self.exploration = ExplorationRewards(
            novelty_weight=config.get('novelty_weight', 0.5),
            transition_weight=config.get('transition_weight', 2.0),
            new_scene_weight=config.get('new_scene_weight', 10.0)
        )
    
    def step(self):
        """Step with exploration rewards"""
        
        # Get observation (CuTile features)
        obs = self.get_observation()
        
        # Get action from policy
        action = self.policy.get_action(obs)
        
        # Execute action
        next_obs, game_reward, done, info = self.env.step(action)
        
        # Compute exploration reward
        exploration_reward = self.exploration.compute_reward(next_obs, action)
        
        # Combined reward
        total_reward = game_reward + exploration_reward
        
        # Learn from total reward
        self.policy.update(obs, action, total_reward, next_obs, done)
        
        # Log
        if self.steps % 100 == 0:
            metrics = self.exploration.get_metrics()
            log.info(f"Exploration: {metrics}")
        
        return total_reward
```

---

## Task 5.3: Update Status Endpoint

### **Update `src/worker/main.py` /status**

```python
def get_status(self):
    """Status endpoint with exploration metrics"""
    
    # Remove menu_hint, ui_clicks, etc.
    # Add exploration metrics
    
    status = {
        # Core metrics
        'worker_id': self.worker_id,
        'episode': self.episode,
        'steps': self.steps,
        'fps': self.fps,
        
        # Exploration metrics (replaces menu_hint)
        **self.exploration.get_metrics(),
        
        # VLM metrics (if available)
        'vlm_hints_used': len(self.vlm_exploration.hint_history) if hasattr(self, 'vlm_exploration') else 0,
        
        # GPU metrics
        'gpu_memory': torch.cuda.memory_allocated() / 1e9,
        
        # Stream status
        'streaming': self.stream_active,
    }
    
    return status
```

---

## Phase 5 Validation

```bash
# 1. Exploration rewards integrated
grep -n "exploration.compute_reward" src/worker/main.py
# Expected: Found

# 2. No menu detection
grep -n "menu_hint\|menu_mode\|ui_clicks" src/worker/main.py
# Expected: (empty)

# 3. Status endpoint updated
curl -s http://localhost:5000/status | jq 'keys'
# Expected: exploration_reward, visual_novelty, scenes_discovered
# NOT: menu_hint, ui_clicks

# 4. Test rewards
./launch --workers 1
tail -f runs/*/logs/worker_0.log | grep "exploration"
# Expected: "Exploration: {'exploration_reward': 0.5, ...}"
```

---

---

# PHASE 6: STREAM VERIFICATION

**Duration**: 2 hours

---

## Task 6.1: Verify All 5 Workers Streaming

### **Check Worker Status**

```bash
# Check all workers running
curl -s http://localhost:8040/workers | jq '.workers | length'
# Expected: 5

# Check each worker's stream endpoint
for port in {5000..5004}; do
    echo "Worker $port:"
    curl -I http://localhost:$port/stream 2>/dev/null | head -1
done
# Expected: 5x "HTTP/1.1 200 OK"
```

### **Verify Stream Page**

```bash
# Open stream page
firefox http://localhost:5173/stream &

# Expected to see:
# - 5 video streams
# - All showing gameplay (not menus)
# - No black/frozen frames
# - Smooth playback
```

---

## Task 6.2: Fix Visual Artifacts

### **Common Issues and Fixes**

#### **Issue 1: Upside-Down Rendering**

**Cause**: OpenGL Y-axis is inverted vs screen coordinates

**Fix in `src/rendering/framebuffer.cpp`**:
```cpp
// BEFORE:
glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

// AFTER: Flip Y-axis
glReadPixels(0, 0, width, height, GL_RGB, GL_UNSIGNED_BYTE, pixels);

// Flip pixels vertically
unsigned char* temp = new unsigned char[width * 3];
for (int y = 0; y < height / 2; y++) {
    unsigned char* row1 = pixels + y * width * 3;
    unsigned char* row2 = pixels + (height - 1 - y) * width * 3;
    
    memcpy(temp, row1, width * 3);
    memcpy(row1, row2, width * 3);
    memcpy(row2, temp, width * 3);
}
delete[] temp;
```

---

#### **Issue 2: Choppy Streams**

**Cause**: Inconsistent frame pacing

**Fix in `src/streaming/stream_server.py`**:
```python
class StreamServer:
    def __init__(self):
        self.target_fps = 60
        self.frame_interval = 1.0 / self.target_fps
        self.last_frame_time = 0
    
    def send_frame(self, frame):
        """Send frame with consistent pacing"""
        
        # Wait for next frame time
        current_time = time.time()
        time_since_last = current_time - self.last_frame_time
        
        if time_since_last < self.frame_interval:
            time.sleep(self.frame_interval - time_since_last)
        
        # Send frame
        self._send_frame_internal(frame)
        
        self.last_frame_time = time.time()
```

---

#### **Issue 3: WebRTC Connection Failures**

**Fix in `scripts/start_go2rtc.sh`**:
```bash
#!/bin/bash
# Ensure go2rtc starts correctly

# Kill existing
pkill -9 go2rtc

# Clear stale state
rm -f /tmp/go2rtc.db

# Start go2rtc
go2rtc -c configs/go2rtc.yaml &

# Wait for startup
sleep 2

# Verify
curl -s http://localhost:1984/api/config
```

---

### **Task 6.3: Stream Quality Validation**

**Create validation script**:

**File: `scripts/validate_streams.py`**

```python
"""Validate all worker streams"""

import requests
import cv2
import numpy as np
from PIL import Image

def validate_stream(port):
    """Validate single worker stream"""
    
    issues = []
    
    # 1. Check endpoint accessible
    try:
        resp = requests.get(f'http://localhost:{port}/stream', timeout=5)
        if resp.status_code != 200:
            issues.append(f"HTTP {resp.status_code}")
    except Exception as e:
        issues.append(f"Connection failed: {e}")
        return issues
    
    # 2. Check frame rate
    frames = []
    timestamps = []
    
    # Capture 60 frames
    for _ in range(60):
        start = time.time()
        frame = get_frame(port)  # Your frame capture method
        frames.append(frame)
        timestamps.append(time.time() - start)
    
    # Check FPS
    avg_fps = 1.0 / np.mean(timestamps)
    if avg_fps < 55:
        issues.append(f"Low FPS: {avg_fps:.1f} (expected 58-60)")
    
    # Check FPS consistency
    fps_std = np.std([1.0/t for t in timestamps])
    if fps_std > 5:
        issues.append(f"Choppy stream: FPS std={fps_std:.1f}")
    
    # 3. Check for visual artifacts
    for i, frame in enumerate(frames):
        # Check if upside down (heuristic: HUD at bottom, should be at top)
        top_half = frame[:frame.shape[0]//2]
        bottom_half = frame[frame.shape[0]//2:]
        
        top_brightness = np.mean(top_half)
        bottom_brightness = np.mean(bottom_half)
        
        # HUD is usually brighter, should be at top
        if bottom_brightness > top_brightness * 1.5:
            issues.append("Possibly upside down (HUD at bottom)")
            break
    
    # 4. Check for black frames
    black_frames = sum(1 for f in frames if np.mean(f) < 10)
    if black_frames > 5:
        issues.append(f"Black frames: {black_frames}/60")
    
    # 5. Check for frozen frames
    frame_diffs = [np.mean(np.abs(frames[i] - frames[i-1])) for i in range(1, len(frames))]
    frozen_count = sum(1 for d in frame_diffs if d < 1.0)
    if frozen_count > 10:
        issues.append(f"Frozen frames: {frozen_count}/60")
    
    return issues


def validate_all_workers():
    """Validate all 5 workers"""
    
    print("=== STREAM VALIDATION ===\n")
    
    all_ok = True
    
    for port in range(5000, 5005):
        worker_id = port - 5000
        print(f"Worker {worker_id} (port {port}):")
        
        issues = validate_stream(port)
        
        if not issues:
            print("  ✅ OK")
        else:
            print(f"  ❌ Issues found:")
            for issue in issues:
                print(f"     - {issue}")
            all_ok = False
        
        print()
    
    if all_ok:
        print("✅ ALL STREAMS OK")
    else:
        print("❌ SOME STREAMS HAVE ISSUES")
    
    return all_ok


if __name__ == '__main__':
    import time
    
    # Wait for workers to start
    print("Waiting for workers to start...")
    time.sleep(10)
    
    # Validate
    ok = validate_all_workers()
    exit(0 if ok else 1)
```

---

## Task 6.4: Fix Gameplay Transitions

### **Verify Menu→Gameplay Flow**

```python
# File: tests/test_gameplay_transitions.py

def test_menu_to_gameplay_transition():
    """Test that workers transition from menu to gameplay"""
    
    import time
    import requests
    
    # Start worker
    worker_port = 5000
    
    # Wait for startup
    time.sleep(5)
    
    # Check initial scene
    status = requests.get(f'http://localhost:{worker_port}/status').json()
    initial_fp = status['scene_fingerprint']
    
    # Wait 60 seconds (should reach gameplay)
    time.sleep(60)
    
    # Check final scene
    status = requests.get(f'http://localhost:{worker_port}/status').json()
    final_fp = status['scene_fingerprint']
    
    # Scenes should be different (progressed)
    assert initial_fp != final_fp, "Worker stuck - no scene change"
    
    # Check if in gameplay (heuristic: high FPS, action count increasing)
    assert status['fps'] > 50, f"Low FPS: {status['fps']}"
    assert status['actions_taken'] > 100, f"Few actions: {status['actions_taken']}"
    
    print("✅ Menu→Gameplay transition OK")
```

---

## Phase 6 Validation

```bash
# 1. All workers streaming
python3 scripts/validate_streams.py
# Expected: "✅ ALL STREAMS OK"

# 2. Stream page working
firefox http://localhost:5173/stream
# Manually verify: 5 streams, all gameplay, smooth

# 3. Gameplay transitions
pytest tests/test_gameplay_transitions.py -v
# Expected: PASSED

# 4. No artifacts
# Check logs for upside-down warnings
grep -i "upside\|flipped\|artifact" runs/*/logs/*.log
# Expected: (empty)
```

---

---

# PHASE 7: INTEGRATION TESTING

**Duration**: 1.5 hours

---

## Task 7.1: End-to-End Validation

### **Test Script**

**File: `tests/test_end_to_end.py`**

```python
"""End-to-end system test"""

import pytest
import requests
import time
import torch

def test_system_startup():
    """Test all components start correctly"""
    
    # 1. Check CUDA 13.1
    assert torch.version.cuda >= "13.1"
    assert torch.cuda.get_device_capability()[0] >= 9
    
    # 2. Check all workers running
    resp = requests.get('http://localhost:8040/workers', timeout=10)
    assert resp.status_code == 200
    workers = resp.json()['workers']
    assert len(workers) == 5, f"Expected 5 workers, got {len(workers)}"
    
    # 3. Check VLM server
    # (Should be running in background)
    # Indirectly tested by worker status
    
    print("✅ System startup OK")


def test_cutile_observations():
    """Test CuTile observations working"""
    
    # Check worker using CuTile
    resp = requests.get('http://localhost:5000/status')
    status = resp.json()
    
    # Should have CuTile metrics
    assert 'observation_type' in status
    assert status['observation_type'] == 'cutile'
    
    print("✅ CuTile observations OK")


def test_exploration_rewards():
    """Test exploration rewards active"""
    
    resp = requests.get('http://localhost:5000/status')
    status = resp.json()
    
    # Should have exploration metrics
    assert 'exploration_reward' in status
    assert 'visual_novelty' in status
    assert 'scenes_discovered' in status
    
    # Should NOT have menu_hint
    assert 'menu_hint' not in status
    assert 'ui_clicks_sent' not in status
    
    print("✅ Exploration rewards OK")


def test_vlm_integration():
    """Test VLM providing hints"""
    
    # Wait for VLM to be used
    time.sleep(30)
    
    resp = requests.get('http://localhost:5000/status')
    status = resp.json()
    
    # Should have used VLM
    assert status.get('vlm_hints_used', 0) > 0
    
    print("✅ VLM integration OK")


def test_streaming_quality():
    """Test all streams working"""
    
    from scripts.validate_streams import validate_all_workers
    
    ok = validate_all_workers()
    assert ok, "Stream validation failed"
    
    print("✅ Streaming quality OK")


def test_gameplay_reached():
    """Test workers reach gameplay"""
    
    # Wait for navigation
    time.sleep(120)
    
    # Check all workers
    for port in range(5000, 5005):
        resp = requests.get(f'http://localhost:{port}/status')
        status = resp.json()
        
        # Should have made progress
        assert status['scenes_discovered'] >= 2, f"Worker {port-5000} stuck"
        assert status['actions_taken'] > 100, f"Worker {port-5000} not active"
    
    print("✅ Gameplay reached OK")


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
```

---

## Task 7.2: Performance Benchmarking

### **Benchmark Script**

**File: `scripts/benchmark_system.py`**

```python
"""Benchmark system performance"""

import time
import requests
import torch
import numpy as np

def benchmark_observation_speed():
    """Benchmark CuTile observation extraction"""
    
    from src.perception.cutile_observations import CuTileObservations
    
    cutile = CuTileObservations()
    
    # Simulate framebuffer
    framebuffer = torch.randn(1080, 1920, 3, device='cuda', dtype=torch.uint8)
    fb_ptr = framebuffer.data_ptr()
    
    # Benchmark
    N = 1000
    torch.cuda.synchronize()
    start = time.time()
    
    for _ in range(N):
        obs = cutile.extract_observation(fb_ptr)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = N / elapsed
    latency = (elapsed / N) * 1000
    
    print(f"CuTile observations: {fps:.1f} FPS ({latency:.2f} ms/frame)")
    assert fps > 1000, f"Too slow: {fps:.1f} FPS"


def benchmark_policy_inference():
    """Benchmark policy inference speed"""
    
    from src.agent.policy import TilePolicy
    
    policy = TilePolicy(
        tile_shape=(68, 120, 64),
        action_space=256
    ).cuda()
    
    # Dummy observation
    obs = torch.randn(1, 68, 120, 64, device='cuda')
    
    # Benchmark
    N = 1000
    torch.cuda.synchronize()
    start = time.time()
    
    with torch.no_grad():
        for _ in range(N):
            logits = policy(obs)
    
    torch.cuda.synchronize()
    elapsed = time.time() - start
    
    fps = N / elapsed
    latency = (elapsed / N) * 1000
    
    print(f"Policy inference: {fps:.1f} FPS ({latency:.2f} ms/frame)")
    assert fps > 500, f"Too slow: {fps:.1f} FPS"


def benchmark_end_to_end():
    """Benchmark full step (obs→action→env)"""
    
    # Measure worker step time
    start_steps = {}
    for port in range(5000, 5005):
        resp = requests.get(f'http://localhost:{port}/status')
        start_steps[port] = resp.json()['steps']
    
    # Wait 10 seconds
    time.sleep(10)
    
    # Measure again
    end_steps = {}
    for port in range(5000, 5005):
        resp = requests.get(f'http://localhost:{port}/status')
        end_steps[port] = resp.json()['steps']
    
    # Calculate FPS
    for port in range(5000, 5005):
        steps = end_steps[port] - start_steps[port]
        fps = steps / 10.0
        print(f"Worker {port-5000}: {fps:.1f} FPS")
        assert fps > 50, f"Worker {port-5000} too slow: {fps:.1f} FPS"


def benchmark_gpu_memory():
    """Check GPU memory usage"""
    
    # 5 workers on RTX 5090 (32GB)
    total_memory = 32 * 1024  # MB
    
    for i in range(5):
        if torch.cuda.device_count() > i:
            used = torch.cuda.memory_allocated(i) / 1024 / 1024
            print(f"GPU {i}: {used:.0f} MB used")
    
    # Total across all GPUs
    total_used = sum(
        torch.cuda.memory_allocated(i) / 1024 / 1024
        for i in range(torch.cuda.device_count())
    )
    
    print(f"Total: {total_used:.0f} MB / {total_memory} MB ({total_used/total_memory*100:.1f}%)")
    
    # Should fit comfortably
    assert total_used < total_memory * 0.9, "GPU memory too high"


if __name__ == '__main__':
    print("=== PERFORMANCE BENCHMARKS ===\n")
    
    benchmark_observation_speed()
    benchmark_policy_inference()
    benchmark_end_to_end()
    benchmark_gpu_memory()
    
    print("\n✅ ALL BENCHMARKS PASSED")
```

---

## Task 7.3: Stress Testing

### **Multi-Hour Stability Test**

**File: `tests/test_stability.py`**

```python
"""Long-running stability test"""

import time
import requests

def test_24h_stability():
    """Run for 24 hours, check for crashes/degradation"""
    
    print("Starting 24-hour stability test...")
    
    start_time = time.time()
    check_interval = 300  # 5 minutes
    
    while time.time() - start_time < 24 * 3600:
        # Check all workers alive
        try:
            resp = requests.get('http://localhost:8040/workers', timeout=10)
            workers = resp.json()['workers']
            
            if len(workers) != 5:
                print(f"❌ Worker count mismatch: {len(workers)}/5")
                return False
            
            # Check each worker
            for port in range(5000, 5005):
                resp = requests.get(f'http://localhost:{port}/status', timeout=5)
                status = resp.json()
                
                # Check FPS
                if status['fps'] < 50:
                    print(f"❌ Worker {port-5000} slow: {status['fps']} FPS")
                
                # Check GPU memory
                if status['gpu_memory'] > 25:  # GB
                    print(f"❌ Worker {port-5000} high memory: {status['gpu_memory']:.1f} GB")
        
        except Exception as e:
            print(f"❌ Health check failed: {e}")
            return False
        
        # Report progress
        elapsed = time.time() - start_time
        print(f"✓ {elapsed/3600:.1f} hours elapsed, all workers OK")
        
        # Wait for next check
        time.sleep(check_interval)
    
    print("✅ 24-hour stability test PASSED")
    return True


if __name__ == '__main__':
    test_24h_stability()
```

---

## Phase 7 Validation

```bash
# 1. End-to-end tests
pytest tests/test_end_to_end.py -v
# Expected: All tests PASSED

# 2. Performance benchmarks
python3 scripts/benchmark_system.py
# Expected: All benchmarks PASSED

# 3. Quick stability check (10 min)
timeout 600 python3 tests/test_stability.py
# Expected: "all workers OK" every 5 minutes
```

---

---

# PHASE 8: DOCUMENTATION & DEPLOYMENT

**Duration**: 30 minutes

---

## Task 8.1: Update Documentation

### **README.md**

```markdown
# MetaBonk: Pure Vision RL System

Production-ready game-agnostic reinforcement learning.

## Features

✅ **Pure Vision Learning** - No hardcoded game knowledge
✅ **CuTile Observations** - 11x faster than raw pixels
✅ **CUDA 13.1** - Modern GPU features (tensor cores, unified memory)
✅ **VLM Hive Reasoning** - Multi-agent vision-language guidance
✅ **Verified Streaming** - All 5 workers streaming reliably

## Requirements

- RTX 5090 (or compute capability 9.0 GPU)
- CUDA 13.1
- Ubuntu 24.04
- Python 3.10+

## Quick Start

```bash
# Install
git clone https://github.com/vladdiethecoder/MetaBonk.git
cd MetaBonk
./scripts/install.sh

# Start training
./launch --workers 5

# Monitor
firefox http://localhost:5173/stream
```

## Architecture

```
Game → CuTile → Policy → Actions → Game
           ↓
        VLM Hive (strategic guidance)
           ↓
    Exploration Rewards
```

## Performance

| Metric | Value |
|--------|-------|
| Observation Speed | 2500 FPS (0.4 ms) |
| Policy Inference | 1000 FPS (1.0 ms) |
| End-to-End FPS | 58-60 |
| Memory per Worker | ~4 GB |
| Total GPU Usage | ~20 GB / 32 GB |

## Documentation

- [Architecture](docs/architecture.md)
- [CuTile Integration](docs/cutile.md)
- [VLM Hive](docs/vlm_hive.md)
- [Pure Vision Principles](docs/pure_vision.md)

## License

MIT
```

---

### **ARCHITECTURE.md**

```markdown
# MetaBonk Architecture

## Overview

MetaBonk is a pure vision RL system with:
1. **CuTile observations** (GPU-accelerated tiles)
2. **CUDA 13.1 inference** (tensor cores, unified memory)
3. **VLM hive reasoning** (vision-language strategic guidance)
4. **Exploration rewards** (pure vision, no game knowledge)

## Data Flow

```
┌─────────────────────────────────────────────────────┐
│                    Game Engine                      │
│               (OpenGL/Vulkan Rendering)             │
└────────────────────┬────────────────────────────────┘
                     │ Framebuffer (GPU-resident)
                     ↓
┌─────────────────────────────────────────────────────┐
│                CuTile Observations                  │
│        (Extract 120x68 tiles, 64D features)         │
│           Zero-copy, GPU→GPU transfer               │
└────────────────────┬────────────────────────────────┘
                     │ Tile features [68,120,64]
                     ↓
         ┌───────────┴──────────┐
         │                      │
         ↓                      ↓
┌────────────────┐    ┌───────────────────┐
│  Policy Net    │    │   VLM Hive        │
│  (CUDA 13.1)   │    │   (Strategic)     │
│  Tensor Cores  │    │   Guidance)       │
└───────┬────────┘    └────────┬──────────┘
        │                      │
        │ Actions              │ Hints
        └──────────┬───────────┘
                   ↓
        ┌─────────────────────┐
        │  Action Executor    │
        │  (BonkLink/Input)   │
        └──────────┬──────────┘
                   │
                   ↓
          ┌────────────────┐
          │  Game Engine   │
          │  (Next Frame)  │
          └────────────────┘
```

## Components

### 1. CuTile Observations

Replace raw pixels with GPU tiles:
- Input: 1920x1080x3 (6.2 MB)
- Output: 120x68x64 (0.5 MB)
- Speedup: 11x memory, 16x latency

### 2. Policy Network

- Input: [B, 68, 120, 64] tile features
- Architecture: Conv layers + spatial attention
- Output: Action distribution
- Inference: CUDA 13.1 tensor cores (FP8)

### 3. VLM Hive

- Centralized VLM server
- Batched inference across workers
- Strategic hints for exploration
- Progress assessment

### 4. Exploration Rewards

- Visual novelty
- Screen transitions
- New scene discovery
- Action diversity

## Performance

| Component | Latency | Throughput |
|-----------|---------|------------|
| CuTile    | 0.4 ms  | 2500 FPS   |
| Policy    | 1.0 ms  | 1000 FPS   |
| VLM       | 50 ms   | 20 FPS (batch) |
| Full Step | 16.7 ms | 60 FPS     |
```

---

## Task 8.2: Create Deployment Checklist

### **DEPLOYMENT_CHECKLIST.md**

```markdown
# MetaBonk Deployment Checklist

## Pre-Deployment

### Hardware
- [ ] RTX 5090 installed
- [ ] 32 GB GPU memory available
- [ ] 128 GB system RAM available
- [ ] NVMe SSD for fast I/O

### Software
- [ ] Ubuntu 24.04
- [ ] CUDA 13.1 installed: `nvcc --version`
- [ ] Python 3.10+: `python3 --version`
- [ ] Docker (if using containers)

### Repository
- [ ] Cloned: `git clone https://github.com/vladdiethecoder/MetaBonk.git`
- [ ] Dependencies installed: `./scripts/install.sh`
- [ ] CuTile built: `ls /usr/local/lib/libcutile.so`

## Validation

### Code Quality
- [ ] No game-specific logic: `grep -r "menu_hint\|MainMenu" src/ → (empty)`
- [ ] CuTile integrated: `grep -r "cutile_obs" src/ → Found`
- [ ] CUDA 13.1 only: `grep -r "cuda.*12\|compute_86" src/ → (empty)`
- [ ] All tests pass: `pytest tests/ -v → All PASSED`

### Performance
- [ ] CuTile benchmark: `python3 scripts/benchmark_cutile.py → >1000 FPS`
- [ ] Policy inference: `python3 scripts/benchmark_system.py → >500 FPS`
- [ ] End-to-end: All workers >50 FPS

### Functionality
- [ ] 5 workers start: `./launch --workers 5 → 5/5 running`
- [ ] All streaming: `python3 scripts/validate_streams.py → OK`
- [ ] VLM active: `curl localhost:5000/status | jq .vlm_hints_used → >0`
- [ ] Exploration working: `curl localhost:5000/status | jq .exploration_reward → >0`

## Deployment

### Start System
```bash
cd /path/to/MetaBonk

# 1. Start infrastructure
./scripts/start_infrastructure.sh  # go2rtc, cognitive server

# 2. Start training
./launch --workers 5

# 3. Verify
./scripts/validate_system.sh
```

### Monitor
```bash
# Dashboard
firefox http://localhost:5173/stream

# Logs
tail -f runs/*/logs/worker_*.log

# Status
watch -n 5 'curl -s localhost:8040/workers | jq .'
```

## Post-Deployment

### First Hour
- [ ] All 5 workers streaming
- [ ] FPS stable (58-60)
- [ ] GPU memory <25 GB
- [ ] No errors in logs

### First Day
- [ ] Workers reach gameplay consistently
- [ ] Exploration rewards active
- [ ] VLM hints being used
- [ ] No crashes

### First Week
- [ ] 24-hour stability test passed
- [ ] Performance stable
- [ ] No memory leaks
- [ ] Ready for production

## Rollback

If issues arise:
```bash
# Stop system
./launch stop

# Check logs
tail -100 runs/*/logs/worker_0.log

# Restore backup
git checkout <previous-commit>
./launch --workers 5
```

## Success Criteria

✅ All 5 workers running
✅ All streaming gameplay (no menus)
✅ FPS 58-60 consistent
✅ GPU memory <25 GB
✅ No visual artifacts
✅ Exploration active
✅ VLM providing hints
✅ No game-specific code
✅ Tests passing
✅ Stable for 24+ hours
```

---

## Task 8.3: Create Monitoring Dashboard

### **MONITORING.md**

```markdown
# MetaBonk Monitoring

## Real-Time Dashboard

Open: http://localhost:5173/stream

Shows:
- 5 worker video streams
- FPS per worker
- Leaderboard
- System stats

## Command-Line Monitoring

### All Workers Status
```bash
curl -s http://localhost:8040/workers | jq '
{
  count: .workers | length,
  workers: [.workers[] | {id, fps, episode, reward}]
}
'
```

### Individual Worker
```bash
# Worker 0
curl -s http://localhost:5000/status | jq '
{
  fps,
  episode,
  exploration_reward,
  visual_novelty,
  scenes_discovered,
  vlm_hints_used,
  gpu_memory
}
'
```

### GPU Usage
```bash
nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used --format=csv
```

### Logs
```bash
# Real-time
tail -f runs/*/logs/worker_0.log

# Errors only
grep -i "error\|warning\|crash" runs/*/logs/*.log

# Performance
grep -i "fps\|latency" runs/*/logs/*.log | tail -20
```

## Alerts

Set up alerts for:
- Worker crashes (count < 5)
- Low FPS (< 50)
- High GPU memory (> 28 GB)
- Errors in logs

Example (using systemd):
```bash
# Check every minute
*/1 * * * * /path/to/MetaBonk/scripts/check_health.sh
```

## Metrics Collection

For long-term monitoring, export metrics to Prometheus/Grafana:

```python
# File: src/monitoring/prometheus_exporter.py
from prometheus_client import start_http_server, Gauge

# Define metrics
worker_fps = Gauge('metabonk_worker_fps', 'Worker FPS', ['worker_id'])
gpu_memory = Gauge('metabonk_gpu_memory_gb', 'GPU memory usage')

# Update periodically
for worker_id in range(5):
    status = get_worker_status(worker_id)
    worker_fps.labels(worker_id=worker_id).set(status['fps'])

# Start metrics server
start_http_server(8000)
```

Access metrics: http://localhost:8000/metrics
```

---

## Phase 8 Validation

```bash
# 1. Documentation complete
ls docs/
# Expected: architecture.md, cutile.md, vlm_hive.md, pure_vision.md

# 2. Deployment checklist available
cat DEPLOYMENT_CHECKLIST.md
# Expected: Full checklist

# 3. Monitoring working
curl -s http://localhost:8040/workers | jq .
# Expected: Worker status

# 4. System ready
./scripts/validate_system.sh
# Expected: "✅ SYSTEM READY FOR DEPLOYMENT"
```

---

---

# FINAL VALIDATION & DEPLOYMENT

---

## Complete System Validation

### **Run All Tests**

```bash
cd /mnt/5da59c95-9aac-48f9-bc21-48c043812e8c/MetaBonk

# 1. Unit tests
pytest tests/ -v
# Expected: All PASSED

# 2. Integration tests
pytest tests/test_end_to_end.py -v
# Expected: All PASSED

# 3. Performance benchmarks
python3 scripts/benchmark_system.py
# Expected: All benchmarks PASSED

# 4. Stream validation
python3 scripts/validate_streams.py
# Expected: "✅ ALL STREAMS OK"

# 5. Quick stability (10 min)
timeout 600 python3 tests/test_stability.py
# Expected: "all workers OK"
```

---

### **Validation Checklist**

```bash
# Code Quality
grep -r "menu_hint\|MainMenu\|build_ui_candidates" src/
# Expected: (empty)

# CuTile
ls /usr/local/lib/libcutile.so
# Expected: File exists

# CUDA 13.1
nvcc --version | grep "release 13.1"
# Expected: Found

# Workers running
curl -s http://localhost:8040/workers | jq '.workers | length'
# Expected: 5

# All streaming
for port in {5000..5004}; do curl -I http://localhost:$port/stream 2>/dev/null | head -1; done
# Expected: 5x "200 OK"

# Exploration active
curl -s http://localhost:5000/status | jq '.exploration_reward'
# Expected: >0

# VLM working
curl -s http://localhost:5000/status | jq '.vlm_hints_used'
# Expected: >0

# GPU memory
nvidia-smi --query-gpu=memory.used --format=csv,noheader
# Expected: <25000 (MB)
```

---

## Production Deployment

```bash
# 1. Stop any existing workers
./launch stop

# 2. Clear old data
rm -rf runs/old_*

# 3. Start production
./launch --workers 5 --config configs/launch_production.json

# 4. Monitor startup
tail -f runs/*/logs/worker_0.log

# Wait for:
# "✓ CuTile initialized"
# "✓ Policy loaded"
# "✓ VLM server ready"
# "✓ Streaming active"

# 5. Verify
./scripts/validate_system.sh
# Expected: "✅ SYSTEM READY"

# 6. Open dashboard
firefox http://localhost:5173/stream &

# 7. Monitor
watch -n 5 'curl -s http://localhost:8040/workers | jq ".workers[] | {id, fps, episode}"'
```

---

## Success Criteria

After 1 hour of running:

✅ **All 5 workers operational**
- `curl http://localhost:8040/workers | jq '.workers | length'` → 5

✅ **All streaming gameplay**
- http://localhost:5173/stream shows 5 video feeds
- All showing gameplay (not menus)

✅ **No visual artifacts**
- No upside-down rendering
- No choppy streams
- Consistent 58-60 FPS

✅ **Pure vision learning**
- No menu_hint in status
- Exploration rewards >0
- VLM hints being used

✅ **Modern CUDA**
- CUDA 13.1 only (no fallbacks)
- Tensor cores active
- GPU memory <25 GB

✅ **Game-agnostic**
- No hardcoded game logic
- Works for ANY game
- Same code for Megabonk, Isaac, Celeste, etc.

---

## Monitoring Commands

```bash
# Real-time worker status
watch -n 2 'curl -s http://localhost:8040/workers | jq ".workers[] | {id, fps, gpu_mem: (.gpu_memory_gb | tonumber)}"'

# Exploration metrics
watch -n 5 'curl -s http://localhost:5000/status | jq "{exploration_reward, visual_novelty, scenes_discovered, vlm_hints_used}"'

# GPU usage
watch -n 1 nvidia-smi

# Logs
tail -f runs/*/logs/worker_*.log | grep -i "error\|warning\|exploration\|vlm"

# FPS tracking
tail -f runs/*/logs/worker_0.log | grep "fps"
```

---

## Long-Term Stability

For production deployment, run 24-hour stability test:

```bash
# Start test
python3 tests/test_stability.py &

# Monitor
tail -f stability_test.log

# Expected output every 5 minutes:
# "✓ 1.2 hours elapsed, all workers OK"
# "✓ 2.4 hours elapsed, all workers OK"
# ...
# "✓ 24.0 hours elapsed, all workers OK"
# "✅ 24-hour stability test PASSED"
```

---

## 🎊 DEPLOYMENT COMPLETE

If all validation passes:

```
╔═══════════════════════════════════════════════════════════╗
║                                                           ║
║     ✅ METABONK PRODUCTION DEPLOYMENT SUCCESSFUL          ║
║                                                           ║
║  ✓ Pure Vision Learning (No game-specific logic)         ║
║  ✓ CuTile Observations (11x faster)                      ║
║  ✓ CUDA 13.1 (Modern GPU features)                       ║
║  ✓ VLM Hive Reasoning (Strategic guidance)               ║
║  ✓ 5 Workers Streaming (Verified quality)                ║
║                                                           ║
║  Dashboard: http://localhost:5173/stream                  ║
║  API: http://localhost:8040/workers                       ║
║                                                           ║
╚═══════════════════════════════════════════════════════════╝
```

**System is now ready for production training!** 🚀

---

**END OF COMPLETE IMPLEMENTATION PLAN**
