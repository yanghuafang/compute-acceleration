# Architecture

## Layout

```
src/                       The only include root, so every include names its
                           layer: "cpu/cpu_gemm.h", not "cpu_gemm.h"
  core/                    Toolchain-independent; the other three all include it
    span.h                 Non-owning bounded view (a C++17 std::span stand-in)
    gemm_shape.h           Extents, element counts, GFLOP/s
    matrix_utils.h         Fill, transpose, tolerance-based comparison
    stopwatch.h            Host wall-clock timing
    cli_options.h          --key=value parser with unknown-option rejection
  cpu/                     What a plain C++ compiler builds
    cpu_gemm.h             The five serial GEMM variants
    reduction.h            Serial sum reductions: the baseline and the lane-split
  omp/                     What it builds given -fopenmp and a runtime to link
    omp_control.h          Thread-count control and reporting
    gemm_omp.h             Row-parallel GEMM kernels
    reduction_omp.h        Reduction with an OpenMP reduction clause
  cuda/                    What NVCC builds
    cuda_check.cuh         CUDA_CHECK / CUDA_CHECK_LAUNCH, CudaError
    device_buffer.cuh      RAII owner of a cudaMalloc allocation
    cuda_timer.cuh         RAII cudaEvent_t pair, device-side timing
    gemm_tiled.cuh         Tiled GEMM kernel + launcher
    sum_reduction.cuh      Block reduction kernel + launcher

benchmarks/                One driver per experiment; each verifies its own result
  support/reporting.h      Shared table formatting (benchmark-private)
tests/                     One suite per header group
  support/                 Test framework and an independent double-precision GEMM
cmake/Sanitizers.cmake     Sanitizer flag wiring
CMakePresets.json          Every named build configuration; CI runs these verbatim
.github/workflows/         lint, build, cuda, sanitizers, Pages; all name presets
scripts/                   build-accel, run-tests, bench, sanitize, format, tidy, docs
  build-env.sh             Sourced by the rest; bash 3.2 compatible
.clang-format              Google style, 80 columns
.clang-tidy                Curated check list, each exclusion with its reason
docs/                      This directory
```

Headers sit beside the sources that implement them rather than in a separate
`include/` tree. With `src/` as the only include root, every include names the
layer it reaches into — `#include "core/span.h"` — so an unwanted edge, `cuda/`
reaching into `omp/` say, is visible at the use site instead of buried in a bare
file name.

Nothing above is generated. Build trees are configured out of tree, into
`../compute-acceleration-build/<preset>`, so the checkout holds only
sources and every configuration can be deleted without touching them.

The split is by *toolchain*, not by feature: `src/cpu` holds everything a plain
C++ compiler builds, `src/omp` what it builds given `-fopenmp` and a runtime to
link, `src/cuda` what NVCC builds. Each layer is independently switchable, which
is what lets `ACCEL_CUDA=OFF` produce a complete, testable build
on a machine with no toolkit — most laptops — and lets the ThreadSanitizer pass
drop `src/omp` without touching anything else.

Every task exists in as many layers as it has meaningful implementations:

| Task | Serial baseline | Serial optimised | OpenMP | CUDA |
|------|-----------------|------------------|--------|------|
| GEMM | `GemmIjk` | `GemmIkj`, `GemmTiled`, `GemmIjkBColMajor`, `GemmTiledBColMajor` | `GemmIkjOmp`, `GemmTiledBColMajorOmp` | `GemmTiledKernel` |
| Sum reduction | `SumSequential` | `SumBlocked` | `SumOmp` | `BlockSumReduceKernel` |

## Design decisions

### Google C++ style, with two departures

Naming, file layout, header guards and comment form follow the
[Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html):
`PascalCase` functions, `snake_case` variables, `kPascalCase` constants,
lowercase `.h`/`.cc` file names, `ACCEL_<PATH>_<FILE>_H_` include guards, `//`
comments, no using-directives. `.clang-format` and `.clang-tidy` enforce what
tooling can; [CONTRIBUTING.md](../CONTRIBUTING.md) states the rest.

Two rules are deliberately not followed, and both are load-bearing rather than
oversights:

- **Exceptions.** The guide bans them, on the grounds that retrofitting
  exception-safety onto an existing codebase costs more than it returns. That
  reasoning does not apply to code written exception-safe from the start, and
  here the error-path design *is* the exception: `CUDA_CHECK` throws so that
  `DeviceBuffer` and `CudaEvent` release their handles while unwinding. The
  banned alternative — `exit(1)` from inside a check macro — leaks every device
  allocation on every error path, which is precisely the bug this project was
  built to not have.
- **Unsigned index types.** The guide prefers signed integers and warns against
  `std::size_t`, while acknowledging that container sizes are unsigned by
  standard mandate. Every index here is a container index, and the benchmark
  shape (`4096 x 2048`) needs 24 bits for a single offset, so `int` is not
  merely unidiomatic but wrong. Extents are widened to `std::size_t` once,
  after validation, and every conversion is explicit — which is what
  `-Wconversion -Wsign-conversion` in the warning set is there to hold.


### Kernels named for the axis they vary

Every host GEMM computes the same product, so the function name is the only
place the interesting difference can live: the loop nesting order (`ijk`,
`ikj`), whether the loops block (`tiled`), and which operand layout is expected
(`b_col_major`). A shared name such as `matrix_mult` would say nothing about
which nest or storage order produced a number. Naming each kernel for its
distinguishing axis keeps a benchmark table readable without opening the
source.

### One flat `accel` namespace

The directory layout separates host from device code; the namespace does not.
A nested `accel::cuda` would collide with libcu++'s `::cuda`: inside
`namespace accel`, an unqualified `cuda::std::foo` would resolve to the
inner namespace and fail to compile, with a diagnostic that points nowhere
useful. `accel::` for everything, with `accel::bench` and
`accel::test` for support code that is not part of the library.

### A hand-rolled `Span` instead of `std::span`

The project pins C++17 for host translation units and for NVCC's device pass, so
`std::span` is unavailable. A 60-line view is cheaper than fragmenting the
codebase across two language levels, and the alternative — raw pointer plus
separate length — is what lets extent bugs through in the first place.

`Span` carries a length, which is what makes a single up-front
`ValidateGemmOperands()` call sufficient. The kernels then index without
per-element bounds checks, keeping the inner loops branch-free.

### Validate once, at the boundary

Every public entry point validates extents before touching memory and throws
`std::invalid_argument` on a violation. Nothing is validated inside a loop nest.
This is the only way to get both a diagnosable API and an inner loop a
vectoriser will accept.

`assert` is not an option here: it compiles out under `-DNDEBUG` — that is, in
exactly the optimised builds the benchmarks run in.

### RAII for every device resource

`DeviceBuffer` and `CudaEvent` own their handles and release them on scope exit,
however that exit is reached. `CUDA_CHECK` throws `CudaError` rather than
calling `exit(1)`: an `exit` from inside an error-check macro bypasses every
destructor and leaks device memory and events on every error path, whereas
throwing lets those owners unwind.

### Only the winning serial kernels get an OpenMP counterpart

`GemmIjk` is 37.9x slower than `GemmIkj`. Threading it would have produced a
headline speedup that was mostly the loop reordering, and the OpenMP row of the
results table would then have measured the wrong thing. The parallel kernels
derive from `GemmIkj` and `GemmTiledBColMajor` so that the speedup column
is attributable to threads alone.

The same reasoning is why `SumBlocked` exists. Comparing `SumOmp` against
`SumSequential` credits threads with 6.1x on an M5 — but 5.9x of that is
available single-threaded, from breaking one dependency chain into eight. A
reduction table without `SumBlocked` in it is a table that attributes
instruction-level parallelism to thread-level parallelism.

### Splitting M, not K

Both OpenMP GEMM kernels distribute rows of C. Each thread then owns a disjoint
slice of the output, so the accumulating `C += A * B` contract survives with no
atomics, no locks, and no reduction over C — and, because every element is
accumulated by one thread in a fixed order, the result is bit-identical to the
serial kernel and independent of thread count. Splitting the contraction would
have needed a reduction over the whole of C and made the result depend on the
schedule.

`SumOmp` cannot have that property: a scalar reduction is combined in an order
the runtime chooses. That is why the tests compare it with a relative tolerance
and `SumBlocked` with exact equality.

### Accumulating CPU kernels, assigning CUDA kernel

The five host variants compute `C += A * B` (BLAS `beta = 1`); the device kernel
computes `C = A * B`. That asymmetry is deliberate and both halves are pinned by
tests. The host benchmarks reuse one output buffer across repetitions and rely on
accumulation; the device kernel writes each output exactly once from a register
accumulator, and reading C first would cost a global load per element for
nothing.

### No external test framework

`tests/support/test_harness.h` is about a hundred lines and provides
registration, per-case failure isolation, and a summary. The project has to stay
buildable with nothing but a compiler and CMake — no network, no vendored
dependency, no `FetchContent`.

The reference GEMM in `tests/support/reference_gemm.h` is deliberately naive
and accumulates in `double`, and shares no code with any optimised path. A bug
in the shared blocking or indexing logic cannot hide by being present in both
the kernel and its own reference.

### Tolerance, not equality

Every kernel sums the same `k` terms in a different order, and floating-point
addition is not associative, so results agree to within accumulated rounding
rather than bit-for-bit. Comparisons are *relative*
(`MaxRelativeError()` normalises by `max(|a|, |b|)`), which is what makes one
tolerance work across outputs spanning several orders of magnitude.

The exception is the all-ones CUDA GEMM check in `bench_cuda_gemm_tiled`: summing
`k` ones is exact in `float` for `k < 2^24`, so that one is an equality check and
any deviation is a real indexing or synchronisation bug.

## Extending

- **A new CPU variant** — add the declaration to `src/cpu/cpu_gemm.h` and the
  definition to `src/cpu/cpu_gemm.cc`, then a case in
  `tests/cpu_gemm_test.cc` comparing against `ReferenceGemm`. Add it to an
  existing driver rather than writing a new one if it belongs in an existing
  comparison.
- **A new CUDA kernel** — header and source under `src/cuda/`, registered in
  the `accel_cuda` target. Give it a `launch*` wrapper that validates geometry
  and derives shared-memory size, so call sites cannot desynchronise the two.
  Add a case to `tests/cuda_kernels_test.cu`; that is what `compute-sanitizer`
  runs.
- **A new OpenMP kernel** — header and source under `src/omp/`, and a case in
  `tests/omp_kernels_test.cc` that runs it at every count in `kThreadCounts`.
  Parallelise the *fastest* serial kernel, not the naive one, or the speedup
  column measures the optimisation the serial variant already had.
- **A new benchmark** — one `accel_add_host_benchmark` or
  `accel_add_cuda_benchmark` line in `CMakeLists.txt`, and an entry in
  `run_benchmark` inside `scripts/bench.sh`. Declare the driver's option
  names to `CliOptions` so a typo is an error rather than a silently ignored
  flag. Every driver must verify its own output and return non-zero on mismatch.
