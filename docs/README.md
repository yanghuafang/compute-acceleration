# compute-acceleration documentation

Two numeric kernels — a GEMM and a sum reduction — implemented four ways each,
and measured. The [root README](../README.md) has the clone-and-run path; this
page indexes everything else.

## How-to

| Document | Contents |
|----------|----------|
| [Install.md](Install.md) | Dependencies (macOS, Ubuntu), `build-accel.sh` options, CUDA on Ubuntu, manual CMake |
| [Testing.md](Testing.md) | The six suites, what the shapes are chosen for, and the guards against a quietly smaller run |
| [Benchmarks.md](Benchmarks.md) | **Published numbers** — every kernel, every backend, with the hardware and the caveats |
| [Sanitizers.md](Sanitizers.md) | Host and device passes, and what CI cannot cover |

## Reference

| Document | Contents |
|----------|----------|
| [Architecture.md](Architecture.md) | Layer map, the dependency rule, and every design decision with its reasoning |

## Generated API reference

The headers under `src/` carry `//` comments, whose first sentence Doxygen
takes as the brief. `scripts/docs.sh` renders them
with Doxygen into `../compute-acceleration-build/docs/html/index.html`, beside
the build tree for the same reason object files go there.

The prose is the same either way — reading it in the header is often quicker.
What the generated site adds is the two things a header cannot show: which layer
includes which, and a link from every declaration to the places that use it.

Configuration is [Doxyfile](doxygen/Doxyfile), which lists only the settings
that differ from Doxygen's defaults, each with its reason. `docs.sh` gates on
the warning log rather than on Doxygen's exit status, because Doxygen exits 0
after warning about a broken reference.

## Where to start

| If you want to know | Read |
|---------------------|------|
| what the numbers actually are | [Benchmarks.md](Benchmarks.md) |
| why a kernel is written the way it is | [Architecture.md](Architecture.md) |
| why `SumBlocked` exists beside `SumSequential` | [Benchmarks.md](Benchmarks.md#openmp-sum-reduction) |
| why ThreadSanitizer skips the OpenMP kernels | [Sanitizers.md](Sanitizers.md#what-ci-covers-and-what-it-cannot) |
| how to add a kernel or a backend | [Architecture.md](Architecture.md#extending) |
