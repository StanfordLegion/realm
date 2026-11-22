<!--
Copyright 2026 Stanford University, NVIDIA Corporation
SPDX-License-Identifier: Apache-2.0
-->

# Prompt for the CUDA cluster agent

Validate the Realm MLIR follow-up in a checkout of the supplied revision. Use a
scratch build directory and the cluster's ordinary GPU job launcher. Record the
Realm revision, LLVM/MLIR version, Toolkit and NVIDIA driver versions, GPU
models, and all commands and test results. Keep the code under test with the
reports and share any fixes as a patch.

1. Build and install a CUDA-enabled Realm. The MLIR direct LLVM examples work
   with Realm LLVM JIT disabled. An example configuration follows. Adjust the
   network, CUDA architecture, and toolchain options for the cluster.

   ```sh
   cmake -S . -B build/realm-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DCMAKE_INSTALL_PREFIX="$PWD/build/realm-cuda-install" -DCMAKE_INSTALL_LIBDIR=lib \
     -DREALM_ENABLE_CUDA=ON -DREALM_ENABLE_LLVM=OFF -DREALM_MAX_DIM=1 \
     -DREALM_LOG_LEVEL=INFO \
     -DREALM_BUILD_TESTS=OFF -DREALM_BUILD_EXAMPLES=OFF \
     -DREALM_BUILD_BENCHMARKS=OFF -DREALM_BUILD_TUTORIALS=OFF -DREALM_BUILD_DOCS=OFF
   cmake --build build/realm-cuda --parallel
   cmake --install build/realm-cuda
   ```

2. Set `MLIR_DIR` to the directory containing LLVM 22's `MLIRConfig.cmake`. Verify
   that this LLVM includes NVPTX with `llc --version`. Configure the CUDA and CPU
   examples and run the suite in a single-process GPU allocation.

   ```sh
   cmake -S mlir -B build/mlir-cuda -DCMAKE_BUILD_TYPE=RelWithDebInfo \
     -DMLIR_DIR="$MLIR_DIR" -DRealm_DIR="$PWD/build/realm-cuda-install/lib/cmake/realm" \
     -DREALM_MLIR_BUILD_LLVM_EXAMPLES=ON -DREALM_MLIR_BUILD_CUDA_EXAMPLE=ON
   cmake --build build/mlir-cuda --parallel
   ctest --test-dir build/mlir-cuda --output-on-failure
   build/mlir-cuda/examples/realm-llvm-cuda -ll:cpu 1 -ll:gpu 1
   ```

   The CUDA demo should report that the three upload/compute/download paths
   produced `i + 14` and that the CUDA pointer and event round-trip produced
   `28`. Rerun the single-process executable with each of two GPUs selected
   using the scheduler's device visibility controls. Also run with
   `-ll:gpu 2` in a two-GPU allocation to check the local framebuffer selection.
   The executable selects one local GPU in each invocation. If available,
   run Compute Sanitizer's memcheck and capture an Nsight Systems CUDA trace.
   Use the trace to compare the two separate Realm task launches with the
   two launches submitted from one task. Treat this demo as a correctness
   check and use repeated warm launches for any latency measurements.

3. With a network-enabled Realm build, run `realm-llvm-cpu -ll:cpu 1
   -realm:prof 2 -level collective=2,taskreg=2` with one Realm process on each of two nodes.
   Confirm that Realm reports two distinct address spaces and writes
   `realmprof_0.dat` and `realmprof_1.dat`. Check that each process registers its
   local tasks once and participates in the startup collective, then both exit
   successfully. The selected host
   process runs the CPU pipeline twice using the same task bindings. To test
   the CUDA demo with the same startup protocol, reserve a GPU on the node
   that owns the first Realm CPU processor. Both processes register their
   local GPU launchers, and the selected host runs the CUDA checks on one GPU.

4. Report correctness for each configuration, attach any diagnostics or trace
   summaries, and identify the earliest failing stage for each problem
   (configure, MLIR lowering, object compilation or linking, PTX loading,
   CUDA launch, result comparison, or teardown).
