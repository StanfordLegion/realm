---
layout: home
---

# Overview

Realm is a distributed, **event–based tasking runtime** for building
high-performance applications that span clusters of CPUs, GPUs, and
other accelerators.

It began life as the low-level substrate underneath the
[Legion](https://github.com/StanfordLegion/legion) programming system but is
now maintained as a standalone project for developers who want direct,
fine-grained control of parallel and heterogeneous machines.

## Why Realm?

* **Asynchronous tasks and events** — Compose applications out of many
  light-weight tasks connected by events instead of blocking synchronization.
* **Heterogeneous execution** — Target CPUs, NVIDIA CUDA/HIP GPUs, OpenMP
  threads, and specialized fabrics with a single API.
* **Scalable networking** — Integrate GASNet-EX, UCX, MPI or shared memory
  transports for efficient inter-node communication.
* **Extensible modules** — Enable/disable features (CUDA, HIP, LLVM JIT, NVTX,
  PAPI …) at build time with simple CMake flags.
* **Portable performance** — Realm applications routinely scale from laptops
  to the world's largest supercomputers.

The runtime follows a *data-flow* execution model: tasks are launched
asynchronously and start when their pre-condition events trigger. This design
hides network and device latency, maximizes overlap, and gives programmers
explicit control over when work becomes runnable.

For a deeper dive see the
[Realm white-paper](https://cs.stanford.edu/~sjt/pubs/pact14.pdf) published
at PACT 2014.

## Acknowledgements

Realm is developed and maintained by the Stanford Legion team with significant
contributions from NVIDIA, Los Alamos, Livermore, Sandia, and many members of
the broader HPC community.
