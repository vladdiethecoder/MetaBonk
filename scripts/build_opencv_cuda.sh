#!/bin/bash
# Build OpenCV with CUDA support for RTX 5090 (SM 8.9 / Blackwell)
# Run with: bash scripts/build_opencv_cuda.sh

set -e

echo "============================================================"
echo "🔧 BUILDING OPENCV WITH CUDA SUPPORT"
echo "============================================================"
echo "Target: RTX 5090 (SM 8.9 / Blackwell)"
echo "This will take ~15-30 minutes depending on CPU cores"
echo "============================================================"
echo ""

# Check for CUDA
if ! command -v nvcc &> /dev/null; then
    echo "❌ CUDA toolkit not found. Install with:"
    echo "   sudo dnf install cuda-toolkit"
    exit 1
fi

CUDA_VERSION=$(nvcc --version | grep release | sed 's/.*release \([0-9]*\.[0-9]*\).*/\1/')
echo "✓ Found CUDA version: $CUDA_VERSION"

# Install build dependencies
echo ""
echo "📦 Installing build dependencies..."
sudo dnf install -y cmake gcc-c++ git python3-devel python3-numpy \
    gtk3-devel ffmpeg-devel libpng-devel libjpeg-devel libtiff-devel \
    libv4l-devel gstreamer1-devel gstreamer1-plugins-base-devel \
    tbb-devel eigen3-devel || true

# Clean previous build
BUILD_DIR="/tmp/opencv_build_$$"
mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

echo ""
echo "📥 Cloning OpenCV repositories..."
git clone --depth 1 --branch 4.9.0 https://github.com/opencv/opencv.git
git clone --depth 1 --branch 4.9.0 https://github.com/opencv/opencv_contrib.git

# Create build directory
mkdir -p opencv/build
cd opencv/build

echo ""
echo "⚙️  Configuring CMake (this takes a minute)..."

# Detect Python paths
PYTHON_EXEC=$(which python3)
PYTHON_INCLUDE=$(python3 -c "from sysconfig import get_paths; print(get_paths()['include'])")
PYTHON_PACKAGES=$(python3 -c "import site; print(site.getsitepackages()[0])")

cmake -D CMAKE_BUILD_TYPE=RELEASE \
      -D CMAKE_INSTALL_PREFIX=/usr/local \
      -D OPENCV_EXTRA_MODULES_PATH="$BUILD_DIR/opencv_contrib/modules" \
      -D WITH_CUDA=ON \
      -D CUDA_ARCH_BIN="8.9" \
      -D CUDA_ARCH_PTX="" \
      -D WITH_CUDNN=ON \
      -D OPENCV_DNN_CUDA=ON \
      -D ENABLE_FAST_MATH=ON \
      -D CUDA_FAST_MATH=ON \
      -D WITH_CUBLAS=ON \
      -D WITH_TBB=ON \
      -D WITH_V4L=ON \
      -D WITH_FFMPEG=ON \
      -D WITH_GSTREAMER=ON \
      -D BUILD_opencv_python3=ON \
      -D PYTHON3_EXECUTABLE="$PYTHON_EXEC" \
      -D PYTHON3_INCLUDE_DIR="$PYTHON_INCLUDE" \
      -D PYTHON3_PACKAGES_PATH="$PYTHON_PACKAGES" \
      -D OPENCV_GENERATE_PKGCONFIG=ON \
      -D BUILD_EXAMPLES=OFF \
      -D BUILD_TESTS=OFF \
      -D BUILD_PERF_TESTS=OFF \
      -D BUILD_DOCS=OFF \
      ..

echo ""
echo "🔨 Building OpenCV (using $(nproc) cores)..."
echo "   This will take 15-30 minutes..."
make -j$(nproc)

echo ""
echo "📦 Installing OpenCV..."
sudo make install
sudo ldconfig

# Create symlink in Python site-packages
CV2_PATH=$(find /usr/local/lib -name "cv2*.so" 2>/dev/null | head -1)
if [ -n "$CV2_PATH" ]; then
    sudo ln -sf "$CV2_PATH" "$PYTHON_PACKAGES/cv2.so" 2>/dev/null || true
fi

echo ""
echo "============================================================"
echo "✅ OpenCV CUDA BUILD COMPLETE!"
echo "============================================================"
echo ""
echo "Verifying installation..."
python3 -c "
import cv2
print(f'OpenCV version: {cv2.__version__}')
print(f'CUDA support: {cv2.cuda.getCudaEnabledDeviceCount() > 0}')
print(f'CUDA devices: {cv2.cuda.getCudaEnabledDeviceCount()}')
if cv2.cuda.getCudaEnabledDeviceCount() > 0:
    print('✅ OpenCV CUDA is ready!')
else:
    print('⚠️  CUDA devices not detected - check driver')
"

echo ""
echo "You may need to restart your shell or run:"
echo "  export LD_LIBRARY_PATH=/usr/local/lib:\$LD_LIBRARY_PATH"
echo ""
