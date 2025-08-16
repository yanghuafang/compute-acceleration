# Sanitizers

Two separate toolchains are needed to cover this project, and neither one sees
what the other does.

| Layer | Tool | Catches |
|-------|------|---------|
| Host C++ | compiler sanitizers (ASan / UBSan / TSan) | heap and stack overflows, use-after-free, signed overflow, invalid casts, data races |
| Device code | `compute-sanitizer` | out-of-bounds device accesses, shared-memory races, uninitialised reads, invalid barriers |

The compiler sanitizers instrument host translation units only. They see
nothing inside a `__global__` function — not one out-of-bounds shared-memory
write. `compute-sanitizer` is a separate NVIDIA binary that runs an ordinary,
uninstrumented build under a virtual machine and inspects device memory traffic.

## Running everything

```bash
scripts/sanitize.sh
```

That runs three passes: `address+undefined`, then `thread`, then
`compute-sanitizer` with each of its four tools. Passes that cannot run — no
CUDA toolkit, no GPU, no `compute-sanitizer` in `PATH` — are reported as skipped
and do not fail the run. To make a missing device pass visible instead:

```bash
scripts/sanitize.sh --mode device   # non-zero if nothing could run
```

Only the test suites are instrumented, never the benchmarks. The default
4096x2048x4096 shape takes minutes uninstrumented and roughly 20x that under
ASan, and it exercises no code path that the small test shapes miss — which is
the reason the tests use prime-ish extents like 37x53x41 rather than convenient
powers of two.

## Host passes

### AddressSanitizer + UndefinedBehaviorSanitizer

```bash
cmake --workflow --preset asan-ubsan          # what CI runs, in one command
```

Or a step at a time, which is what `scripts/build-accel.sh --sanitizer
address+undefined` resolves to:

```bash
cmake --preset asan-ubsan
cmake --build --preset asan-ubsan
ctest --preset asan-ubsan                     # asan-ubsan-macos on a Mac
```

The `asan-ubsan` preset implies two things:

- **CUDA targets are disabled.** An ASan host process reports false positives
  from the CUDA driver's own allocations, so a combined build is noise.
- **`RelWithDebInfo` replaces `Release`.** `-O0` makes an already-slow
  instrumented run slower still, while `-O2` with frame pointers and debug info
  keeps stack traces readable. `cmake/Sanitizers.cmake` forces `-g` regardless
  of build type for the same reason.

UBSan is configured with `-fno-sanitize-recover=all`, and the test preset sets
`halt_on_error=1`. Without them UBSan prints a diagnostic and *continues*, so a
scripted run exits zero while having found real defects. With them, the first
report is the exit status.

The test preset also sets `detect_leaks` explicitly per platform — `1` on Linux,
`0` on macOS, where LeakSanitizer is unsupported and would abort every test.
There are two presets rather than one because a preset's environment cannot
branch; `ctest --list-presets` shows only the one valid for your platform.

### ThreadSanitizer

```bash
cmake --workflow --preset tsan
```

TSan and ASan cannot share a process — both runtimes claim the same
shadow-memory ranges — so `cmake/Sanitizers.cmake` rejects the combination with
a configure-time error rather than letting it fail at runtime. They are two
presets, two build directories, and two CI jobs for that reason.

⚠️ **The thread pass does not cover `src/omp`.** `scripts/build-accel.sh` passes
`--no-openmp` whenever `--sanitizer thread` is given, and says so in its output.

TSan reasons about happens-before edges it can observe. The barrier at the end
of an `omp parallel for` is a real edge, but it lives inside the OpenMP runtime,
and TSan sees it only if that runtime was built with `LIBOMP_TSAN_SUPPORT` —
which ships the `libarcher` OMPT tool. Neither Homebrew's `libomp` nor a
distribution `libgomp` is built that way, so *every* worker write is reported as
racing the master's read after the region:

```
WARNING: ThreadSanitizer: data race
  Read of size 4 by main thread:      MaxRelativeError(...)
  Previous write of size 4 by T1:     GemmIkjOmp (.omp_outlined)
```

Suppressing that would mean suppressing `called_from_lib:libomp`, which covers
every parallel region in the project — a green pass that checks nothing. The
honest option is to leave the OpenMP targets out and say so.

To get real coverage, build LLVM's OpenMP runtime with `LIBOMP_TSAN_SUPPORT=ON`
and preload the resulting `libarcher`, then configure with
`-DACCEL_OPENMP=ON -DACCEL_TSAN=ON` directly
rather than through `scripts/build-accel.sh`.

Until then the OpenMP kernels are checked three other ways: ASan and UBSan do
cover them (the address+undefined pass leaves OpenMP enabled), `test_omp_kernels`
runs every case at 1, 2, 3 and 8 threads, and the row-split design makes the
results bit-identical across thread counts — a test that fails the moment two
threads touch the same output element.

### Runtime options

`scripts/sanitize.sh` exports these unless you have already set them:

| Variable | Value |
|----------|-------|
| `ASAN_OPTIONS` | `abort_on_error=0:strict_string_checks=1:detect_stack_use_after_return=1` (plus `detect_leaks=1` on Linux) |
| `UBSAN_OPTIONS` | `print_stacktrace=1:halt_on_error=1` |
| `TSAN_OPTIONS` | `halt_on_error=1:second_deadlock_stack=1` |

> **Note**
> `detect_leaks` is Linux-only. Apple's ASan ships without LeakSanitizer, and
> requesting it on macOS makes the runtime abort before `main` with
> *"detect_leaks is not supported on this platform"* — which reads exactly like
> a test failure and is not one. Leak detection therefore only happens on Linux;
> the RAII wrappers in `src/cuda/` are what make that acceptable.

## Device pass

```bash
scripts/sanitize.sh --mode device
```

`compute-sanitizer` runs one tool per invocation, and they catch disjoint classes
of bug, so the script runs all four against `test_cuda_kernels`:

| Tool | What it finds | Why it matters here |
|------|---------------|---------------------|
| `memcheck` | out-of-bounds and misaligned device accesses, leaked device allocations | The boundary-masking arithmetic in `GemmTiledKernel` and the grid-stride bound in `BlockSumReduceKernel` are exactly where an off-by-one lands |
| `racecheck` | shared-memory hazards | The tiled GEMM needs a *second* `__syncthreads()` after the inner product so no warp overwrites a tile another warp is still reading; drop it and only `racecheck` notices |
| `initcheck` | reads of device memory never written | Catches a partials buffer or an output tile that a masked-off branch left untouched |
| `synccheck` | invalid or divergent barrier use | `BlockSumReduceKernel` keeps its `__syncthreads()` outside the `if (tid < step)` for this reason — a barrier inside is undefined behaviour, not merely slow |

Run manually against one tool:

```bash
compute-sanitizer --tool racecheck --error-exitcode 1 \
  ../compute-acceleration-build/compute-sanitizer/bin/test_cuda_kernels
```

`--error-exitcode 1` is not optional in a script. By default
`compute-sanitizer` exits zero even when its report describes a fault, so a
scripted run comes back green on a real bug.

The build for this pass is `RelWithDebInfo` and the CUDA targets compile with
`-lineinfo` (set in `CMakeLists.txt`), which is what makes reports point at
source lines instead of raw PC offsets. It does not change generated code.

## What CI covers, and what it cannot

`.github/workflows/sanitizers.yml` runs the `asan-ubsan` and `tsan` presets as
separate jobs on ubuntu-24.04 (Clang 20) and macos-15, on every push and once a
week. Two properties of that matrix are worth knowing:

- **LeakSanitizer rides along with ASan on Linux, but not on macOS/arm64.** The
  Ubuntu jobs therefore check strictly more than a local run on a Mac, and a
  leak introduced on macOS will first be reported by CI.
- **The device pass never runs.** `compute-sanitizer` requires a physical
  NVIDIA GPU and no GitHub-hosted runner has one, so `memcheck`, `racecheck`,
  `initcheck` and `synccheck` are covered by local runs only. A green CI badge
  says nothing about device-side memory safety.

⚠️ Treat `scripts/sanitize.sh --mode device` as a required step before
any change to `src/cuda/` or `src/cuda/` is considered done. CI
cannot do it for you.

## Why the tests exist at all

The suites in `tests/` are small and fast primarily so that the passes above are
cheap enough to run routinely. Shapes are non-square and not multiples of the
tile size (37x53x41, 1x7x96, 96x7x1) so that tail handling and any m/n/k
transposition is exercised; the CUDA suite adds an aligned 64x32x96 case, a
minimal 1x1x1 case, and a `--count` that is not a multiple of
`blocks * threads`.

The CUDA suite exits successfully with a notice when no device is visible. A
green run on a machine without a GPU therefore means "the host code is correct",
not "the kernels were verified". Use `scripts/run-tests.sh --require-cuda` where
that distinction has to be enforced.
