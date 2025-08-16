# `compute-acceleration`: one kernel, four backends

[![Lint](https://github.com/yanghuafang/compute-acceleration/actions/workflows/lint.yml/badge.svg)](https://github.com/yanghuafang/compute-acceleration/actions/workflows/lint.yml)
[![Build](https://github.com/yanghuafang/compute-acceleration/actions/workflows/build.yml/badge.svg)](https://github.com/yanghuafang/compute-acceleration/actions/workflows/build.yml)
[![CUDA](https://github.com/yanghuafang/compute-acceleration/actions/workflows/cuda.yml/badge.svg)](https://github.com/yanghuafang/compute-acceleration/actions/workflows/cuda.yml)
[![Sanitizers](https://github.com/yanghuafang/compute-acceleration/actions/workflows/sanitizers.yml/badge.svg)](https://github.com/yanghuafang/compute-acceleration/actions/workflows/sanitizers.yml)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![API reference](https://img.shields.io/badge/docs-API%20reference-blue)](https://yanghuafang.github.io/compute-acceleration/)

Two numeric kernels — a GEMM and a sum reduction — implemented four ways each:
a serial baseline, a serial variant tuned for memory access order, an OpenMP
variant, and a CUDA one. All measured against the same shape, so the four rows
are comparable.

The short answers, on an Apple M5. For GEMM, reordering three loops is worth
**37.9x**; cache blocking 2.6x; transposing an operand 4.9x; stacking those two
less than either. OpenMP adds **4.7x** on ten cores, and the GPU kernel is
another 93x on top of the best CPU result.

For the reduction, the interesting result is that eight independent accumulator
lanes on **one** core reach 45.5 GiB/s where the textbook loop reaches 7.7 —
and it takes all ten cores under OpenMP to edge past that. Instruction-level
parallelism, not threads, is what the naive loop was leaving on the table.
Numbers and analysis in [docs/Benchmarks.md](docs/Benchmarks.md).

## Quick start

```bash
git clone git@github.com:yanghuafang/compute-acceleration.git
cd compute-acceleration/scripts

# macOS:        ./install-deps-macos.sh
# Ubuntu: ./install-deps-ubuntu.sh

./build-accel.sh    # Release; CUDA and OpenMP targets if the toolchains are found
./run-tests.sh      # ctest, every suite
./bench.sh          # the published shape, a few minutes
```

Nothing is fetched and nothing is vendored — a C++17 compiler and CMake 3.25 are
the whole hard dependency list. Both accelerator toolchains are optional and
detected at configure time: without them the build is smaller but complete and
testable, and `./run-tests.sh` still passes. Pass `--require-openmp` or
`--require-cuda` when you want a missing one to fail instead.

```bash
scripts/bench.sh --quick     # small shapes, seconds
scripts/sanitize.sh             # ASan+UBSan, TSan, compute-sanitizer
```

## What is here

**Five CPU GEMM variants**, all computing `C += A * B` at
`A(4096x2048) * B(2048x4096)`:

| Kernel | Idea | `M5/clang` |
|--------|------|-----------:|
| `GemmIjk` | textbook `i -> j -> k`, row-major B — the baseline | 76.371 s |
| `GemmTiled` | cache blocking, row-major B | 28.895 s |
| `GemmIjkBColMajor` | transposed B, so both inner operands are unit-stride | 15.556 s |
| `GemmTiledBColMajor` | blocking *and* a transposed B | 5.267 s |
| `GemmIkj` | loop reorder to a scalar-times-row AXPY; auto-vectorises | **2.013 s** |

**Two OpenMP GEMM kernels**, both splitting rows of C so each thread owns a
disjoint slice of the output — bit-identical to the serial result at any thread
count:

| Kernel | Idea | `M5/clang`, 10 threads |
|--------|------|-----------------------:|
| `GemmIkjOmp` | `GemmIkj` with the M axis distributed | 172.3 GFLOP/s |
| `GemmTiledBColMajorOmp` | blocked, distributed by row block | 59.7 GFLOP/s |

**Four sum reductions** over 256 MiB of floats, reported as effective read
bandwidth because a reduction is memory-bound:

| Kernel | Idea | `M5/clang` |
|--------|------|-----------:|
| `SumSequential` | one accumulator — one dependency chain | 7.7 GiB/s |
| `SumBlocked` | eight independent lanes, still one thread | **45.5 GiB/s** |
| `SumOmp` | OpenMP `reduction(+ : ...)`, 10 threads | 47.4 GiB/s |
| `BlockSumReduceKernel` | grid-stride load into a shared-memory tree | see [docs/Benchmarks.md](docs/Benchmarks.md) |

**Two CUDA kernels**:

- `GemmTiledKernel` — 32x32 shared-memory tiling, 21.758 ms on an RTX 3090 Ti
  (3158 GFLOP/s, ~8% of the card's FP32 peak).
- `BlockSumReduceKernel` — grid-stride load into a shared-memory tree
  reduction, one partial sum per block, final add on the host.

## Layout

```
src/                  The only include root; headers sit beside their sources
  core/               Span, GemmShape, MatrixUtils, Stopwatch, CliOptions
  cpu/  omp/  cuda/   Kernels, split by which toolchain builds them:
                      plain C++, C++ with -fopenmp, and NVCC
benchmarks/           One driver per experiment; each verifies its own result
tests/                One suite per layer, plus a ~100-line harness
cmake/                Sanitizer flag wiring
scripts/              build-accel, run-tests, bench, sanitize, format, tidy, docs
docs/                 Install, Testing, Benchmarks, Sanitizers, Architecture
```

[docs/Architecture.md](docs/Architecture.md) covers the design decisions — why
`Span` is hand-rolled, why validation happens once at the boundary, why the CPU
kernels accumulate and the CUDA one assigns, and why there is no external test
framework.

## Building

Every named configuration is a preset in [`CMakePresets.json`](CMakePresets.json),
so CMake needs no wrapper at all:

```bash
cmake --workflow --preset release      # configure, build and test in one step
cmake --workflow --preset asan-ubsan   # exactly what the sanitizer CI job runs
cmake --list-presets                   # every configuration, with descriptions
```

`scripts/build-accel.sh` resolves its flags to one of those presets and applies
anything genuinely ad-hoc on top, so a developer's shorthand and CI's command
cannot describe different builds:

```bash
scripts/build-accel.sh --type Debug
scripts/build-accel.sh --arch 86 --jobs 16          # pin the SM, don't probe for it
scripts/build-accel.sh --no-cuda                    # host targets only
scripts/build-accel.sh --no-openmp                  # serial host targets only
scripts/build-accel.sh --sanitizer address+undefined
scripts/build-accel.sh --help
```

| CMake option | Default | Effect |
|--------------|---------|--------|
| `ACCEL_CUDA` | `ON` | Build CUDA targets when a toolkit is found |
| `ACCEL_OPENMP` | `ON` | Build OpenMP targets when a runtime is found |
| `ACCEL_TESTS` | `ON` | Build the test suites |
| `ACCEL_BENCHMARKS` | `ON` | Build the benchmark drivers |
| `ACCEL_WERROR` | `ON` | `-Werror` / `/WX`; `--no-werror` to iterate |
| `ACCEL_ASAN` | `OFF` | AddressSanitizer on host code |
| `ACCEL_UBSAN` | `OFF` | UndefinedBehaviorSanitizer on host code |
| `ACCEL_TSAN` | `OFF` | ThreadSanitizer on host code |

`CMAKE_CUDA_ARCHITECTURES` defaults to `native`, which compiles only for the
installed GPUs. Set it explicitly (`--arch 86`) to build a portable binary or to
build on a machine with no GPU.

Builds are **out of tree**: every preset configures into a sibling of the
checkout, `../compute-acceleration-build/<preset>`, so a Release tree and
an instrumented one coexist without putting object files where `grep` and the
editor index have to walk them. Binaries land in `<build-dir>/bin`.

Set `ACCEL_BUILD_DIR` to relocate every configuration at once — onto a
faster disk, say — or pass `--build-dir` to place a single one explicitly.

## Running the benchmarks

Each driver takes `--key=value` options and rejects anything it does not
recognise, so a mistyped flag is an error rather than a silently ignored one:

```bash
BIN=../compute-acceleration-build/release/bin

${BIN}/bench_cpu_gemm_layouts --m=1024 --k=512 --n=1024 --tile=32
${BIN}/bench_omp_gemm_scaling --threads=8 --iters=5
${BIN}/bench_omp_sum_reduction --count=268435456
${BIN}/bench_cuda_gemm_tiled --iters=500
${BIN}/bench_cuda_sum_reduction --count=268435456 --blocks=1024
${BIN}/bench_cpu_gemm_row_major --help
```

`scripts/bench.sh` takes `--cpu-only`, `--omp-only` and `--cuda-only`
to run one backend at a time; `--cpu-only` means the single-threaded drivers,
since OpenMP has its own flag.

Every driver checks its own result against a reference or an exact expected
value and returns non-zero on a mismatch, so the benchmarks double as an
end-to-end test.

## Testing and sanitizers

```bash
scripts/run-tests.sh                    # builds with --werror, then ctest
scripts/run-tests.sh --require-cuda     # fail if the CUDA suite was not built
scripts/run-tests.sh --require-openmp   # fail if the OpenMP suite was not built
scripts/sanitize.sh               # every sanitizer pass in turn
```

The CUDA suite exits successfully with a notice when no device is visible, so a
green run on a machine without a GPU means "the host code is correct", not "the
kernels were verified" — `--require-cuda` is how you make that distinction
enforceable.

Host code is covered by the compiler sanitizers; device code by
`compute-sanitizer`, which is a separate tool and sees inside kernels where ASan
does not. Both, and why the tests are deliberately small and use prime-ish
extents, are in [docs/Sanitizers.md](docs/Sanitizers.md).

⚠️ **ThreadSanitizer does not cover the OpenMP kernels.** TSan cannot see the
barrier at the end of a parallel region unless the OpenMP runtime carries
`LIBOMP_TSAN_SUPPORT` annotations, and the runtimes Homebrew and the
distributions ship do not — so every worker write reports as a race. The thread
pass therefore builds with `--no-openmp`, and says so. ASan and UBSan do cover
them; [docs/Sanitizers.md](docs/Sanitizers.md) explains how to get real TSan
coverage.

## Continuous integration

Five workflows, split by how long they take and what they can break rather than
by topic, so that a formatting slip is not reported behind a twenty-minute CUDA
compile:

| Workflow | Runners | What it establishes |
|----------|---------|---------------------|
| `lint.yml` | ubuntu-24.04 | clang-format and clang-tidy; configure-only, so it answers in under a minute |
| `docs.yml` | ubuntu-24.04 | Doxygen with warnings fatal; publishes to Pages from `main` only |
| `build.yml` | ubuntu-24.04 (GCC 13 and Clang 20), macos-15 | `ACCEL_WERROR` build, ctest with `--require-openmp`, a `--smoke` benchmark pass |
| `cuda.yml` | ubuntu-24.04 + `nvcc` | `src/cuda/` and the `.cuh` headers compile and link for `sm_86` |
| `sanitizers.yml` | ubuntu-24.04 and macos-15 | ASan+UBSan and TSan as separate jobs, plus a weekly scheduled run |

No workflow contains a compiler flag, a build type or a `-D` of any kind: each
step names a preset, so `cmake --workflow --preset asan-ubsan` reproduces a
failing sanitizer job exactly. Runner images and the Clang major are pinned for
the same reason — an unpinned clang-format eventually disagrees with the one in
your editor about a file nobody edited.

The matrix is about compilers rather than distributions. Any one developer
builds with exactly one of Homebrew LLVM, GCC and mainline Clang, and the three
disagree about `-Wconversion` in particular — `ACCEL_WERROR` in CI is where that
surfaces. They also disagree about OpenMP: GCC has libgomp built in, AppleClang
compiles OpenMP but ships no runtime, so macOS uses the Homebrew keg and every
job then asserts `--require-openmp` rather than letting a missing runtime turn
into a quietly smaller test run.

Branch protection names the matrix legs directly. Adding a leg therefore means
adding it to the required set as well, and removing one means dropping it —
otherwise the new leg is ungated, or the old name is required and never
reports.

**Note:** no hosted runner has a GPU. The CUDA job compiles the device code and
runs the suite, which self-skips, so the kernels are proven to *build* but never
to *execute*; `compute-sanitizer` cannot run there at all. Device correctness
still requires `scripts/run-tests.sh --require-cuda` and
`scripts/sanitize.sh --mode device` on a machine with an NVIDIA card.

Every runner is a GA image. Beta images are deliberately not in the matrix:
they fail for reasons that belong to the image rather than to the code, and a
job that is allowed to fail is one nobody reads.

## Style

Google style, 80 columns, enforced by `.clang-format`:

```bash
scripts/format.sh           # format in place
scripts/format.sh --check   # report offenders, change nothing, exit non-zero
```

Includes are regrouped by `.clang-format` into the order the style guide
states: the implementation's own header, C system headers, the C++ standard
library, other libraries, then this project. Two settings there are
load-bearing and carry their reasons in the file — `DerivePointerAlignment`,
whose Google default flipped in LLVM 21, and `IncludeIsMainSourceRegex`,
without which clang-format does not treat a `.cu` as a source file and sorts
its own header into the middle of the project block.

## License

MIT — see [LICENSE](LICENSE).
