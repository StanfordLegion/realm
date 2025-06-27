---
permalink: /quick-start/
toc: true
---

### 1. Clone
```bash
git clone https://github.com/StanfordLegion/realm.git
cd realm
```

### 2. Build with CMake (recommended)
```bash
# Create an out-of-tree build directory
mkdir build && cd build
# Configure – pick the options that match your system
cmake .. \
      -DCMAKE_BUILD_TYPE=Release \
      -DREALM_ENABLE_OPENMP=ON \   # OpenMP support
      -DREALM_ENABLE_CUDA=OFF       # flip ON to target NVIDIA GPUs

# Compile everything
make -j$(nproc)

# (optional) run the unit tests
ctest --output-on-failure
```
The full list of CMake toggles is documented inside [`CMakeLists.txt`](CMakeLists.txt).  Common switches include:

| Option | Default | Purpose |
| ------ | ------- | ------- |
| `REALM_ENABLE_CUDA` | `ON`  | Build CUDA backend |
| `REALM_ENABLE_HIP`  | `ON`  | Build HIP/ROCm backend |
| `REALM_ENABLE_GASNETEX` | `ON on Linux` | GASNet-EX network |
| `REALM_ENABLE_UCX` | `ON on Linux` | UCX network |
| `REALM_ENABLE_MPI` | `OFF` | MPI network |
| `REALM_LOG_LEVEL`  | `WARNING` | Compile-time log level |

> **TIP:** combine `cmake -LAH` or `ccmake` to explore every option.

### 3. Install (optional)
```bash
make install   # honour DESTDIR / CMAKE_INSTALL_PREFIX as usual
```
Libraries, headers and CMake packages will be placed under `include/realm`, `lib/`, and `share/realm/` so that external projects can consume Realm via
```cmake
find_package(Realm REQUIRED)
```
