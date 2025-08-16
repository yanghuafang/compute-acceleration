# Testing

Six suites, all built from `tests/` and registered with CTest. They are
deliberately small: the sanitizer passes rerun every one of them, and a suite
that takes minutes uninstrumented takes an afternoon under ASan.

| Suite | Covers | Built when |
|-------|--------|-----------|
| `test_cpu_gemm` | the five serial GEMM variants, extents, aliasing contracts | always |
| `test_matrix_utils` | fill, transpose, and the tolerance comparison | always |
| `test_cli_options` | `--key=value` parsing and unknown-option rejection | always |
| `test_reduction` | `SumSequential` and `SumBlocked`, including lane tails | always |
| `test_omp_kernels` | the OpenMP kernels at 1, 2, 3 and 8 threads | an OpenMP runtime is found |
| `test_cuda_kernels` | the device kernels and the RAII owners | a CUDA toolkit is found |

## Running

```bash
cd scripts
./run-tests.sh                    # build the release preset, run every suite
./run-tests.sh --no-build         # use the existing build directory
./run-tests.sh --require-openmp   # fail if the OpenMP suite was not built
./run-tests.sh --require-cuda     # fail if the CUDA suite was not built
./run-tests.sh --preset release-hostonly   # skip the `device` label
./run-tests.sh --no-build -- -R omp   # extra arguments go to ctest
```

Or without the wrapper — `ctest --preset release` runs the same suites with the
same environment, and `cmake --workflow --preset release` covers configure,
build and test in one command.

Every test carries a label: `host` on all of them, `openmp` on the suite that
needs a runtime, and `device` on the CUDA suite. `ctest -L openmp` selects; the
`release-hostonly` preset excludes `device`, which is how CI states that no
hosted runner can run those rather than letting the suite self-skip.

⚠️ **A green run can mean less than it looks.** An optional toolchain CMake
failed to detect produces a smaller, still-passing test run — six suites become
four and nothing says so. That is exactly when a passing check is most
misleading, which is what the `--require-*` guards are for. CI passes them.

The CUDA suite has a second version of the same trap: it exits successfully with
a notice when no device is visible, so it is green on a machine with the toolkit
and no GPU. `--require-cuda` establishes that it was *built*, not that a kernel
ran.

## No external test framework

`tests/support/test_harness.h` is about a hundred lines: a Registry, four
`CHECK_*` macros, and a `RunAll()` that reports a tally. The reasoning is in
[Architecture.md](Architecture.md#no-external-test-framework) — briefly, a
project whose point is that it has no dependencies should not acquire one to
test itself, and a GoogleTest fetch is larger than the code under test.

`tests/support/reference_gemm.h` holds the independent double-precision GEMM
every kernel is checked against, plus the fixed LCG that generates operands. It
is a deliberately naive `i -> j -> k` loop: a reference that shared an
optimisation with the kernel under test would agree with it for the wrong reason.

## What the shapes are chosen for

Extents are prime-ish and non-square — `37x53x41`, `1x7x96`, `96x7x1` — so that
tail handling runs on every kernel and an m/n/k transposition cannot pass. None
is a multiple of the 64-element default tile, so every blocked kernel takes its
clamped-tail path. The CUDA suite adds an aligned `64x32x96` case, a minimal
`1x1x1`, and a `--count` that is not a multiple of `blocks * threads`.

Operands come from a fixed LCG rather than a ramp: a ramp makes every row of A a
scalar multiple of every other, which masks exactly the transposition and
row/column mix-ups these tests exist to catch.

## Thread counts

`test_omp_kernels` runs each case at 1, 2, 3 and 8 threads. A parallel kernel
that is correct on one thread and wrong on eight is the common failure, and a
suite that never varies the count cannot see it. Eight is more threads than the
`37`-row test shape can use, so the runtime's clamping and the empty-slice path
are exercised too.

Two assertions there are worth knowing about. The OpenMP GEMM results must be
**bit-identical** across thread counts and to the serial kernel — the row split
gives each thread a disjoint slice of C, so any drift means two threads are
touching the same output element. `SumOmp` deliberately does not have that
property, because the runtime chooses the combine order, so it is compared with
a relative tolerance instead.

## Benchmarks as tests

Every benchmark driver verifies its own output against a reference or an exact
expected value and returns non-zero on a mismatch, so `bench.sh` is an
end-to-end check as well as a measurement:

```bash
./bench.sh --quick     # small shapes, seconds
./bench.sh --smoke     # the same thing, under the name CI uses
```

## Sanitizers

[Sanitizers.md](Sanitizers.md) covers those, including the one gap: the thread
pass cannot see through an unannotated OpenMP runtime, so it drops those targets
rather than reporting false races.

## CI

`.github/workflows/build.yml` runs, on ubuntu-24.04 (GCC 13 and Clang 20) and
macos-15: a build of the `release` preset, `run-tests.sh --preset
release-hostonly --require-openmp`, and `bench.sh --smoke`.

`format.sh` and `tidy.sh` live in `.github/workflows/lint.yml`, on one
platform: their findings do not depend on the host, and both answer in under a
minute. Neither needs a compiled tree — clang-tidy reads the compile database,
which CMake writes at generate time — so a formatting slip is reported in
seconds rather than behind a macOS build.

`docs.sh --strict` runs in `.github/workflows/docs.yml`, which gates pull
requests and publishes from `main`. It is not also in `lint.yml`: that would
build the same site twice on every push to `main`.

That the format check runs on one platform is now economy rather than necessity.
clang-format's layout moves between majors, and the runners do not ship the same
one by default, so the major is pinned (`ACCEL_CLANG_VERSION` in
`scripts/build-env.sh`) and both the install scripts and CI honour it.

`.github/workflows/cuda.yml` compiles the CUDA layer with `nvcc` for `sm_86`. No
hosted runner has a GPU, so that job proves `src/cuda` still builds and links;
it cannot run a kernel, and the `cuda-ci-hostonly` test preset says so by
excluding the `device` label. `.github/workflows/sanitizers.yml` runs the host
sanitizer passes, one job per sanitizer.
