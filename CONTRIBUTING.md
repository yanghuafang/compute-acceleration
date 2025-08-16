# Contributing to `compute-acceleration`

Thanks for your interest. This is a **measurement project**: two numeric kernels
implemented across four backends, with published numbers. Contributions of all
sizes are welcome — a new kernel variant, a platform's results, clearer docs, a
fix to an inner loop.

Because the point of the repository is that its numbers are *attributable*,
a change that improves a measurement matters less than a change that makes
clear which variable produced it.

## Ways to contribute

- **Add a kernel variant** — see [Adding a kernel](#adding-a-kernel).
- **Report numbers from your hardware** — the tables in
  [docs/Benchmarks.md](docs/Benchmarks.md) name the machine each column came
  from; a new column with its own machine row is welcome.
- **Improve documentation** in `docs/`, or add comments that explain *why*.
- **Fix a bug** — [docs/Architecture.md](docs/Architecture.md) maps `src/` to
  responsibilities.

## Prerequisites

Targets **macOS** (Homebrew) and **Ubuntu**. You need CMake 3.25+ and a
C++17 compiler; an OpenMP runtime, a CUDA toolkit, clang-format, clang-tidy,
doxygen and graphviz enable the rest.

```bash
cd scripts
./install-deps-macos.sh     # macOS
./install-deps-ubuntu.sh    # Ubuntu
```

See [docs/Install.md](docs/Install.md) for what each dependency turns on, the
CUDA-on-Ubuntu recipe, and the manual CMake options.

## Build

```bash
cd scripts
./build-accel.sh
```

Output goes to `../../compute-acceleration-build/release/bin` — a sibling of the
repo, kept out of the source tree. The directory is named for the preset that
built it; `cmake --list-presets` shows the rest.

## Test

Every change must keep the suites green. From `scripts/`:

```bash
./run-tests.sh --require-openmp   # add --require-cuda on a machine with a GPU
./bench.sh --smoke                # drivers verify their own output
./sanitize.sh --mode host         # ASan+UBSan, then TSan
```

⚠️ **Use the `--require-*` guards.** A toolchain CMake failed to detect produces
a smaller, still-green test run: six suites become four and nothing says so. CI
passes them for that reason, and so should you.

On a machine with an NVIDIA GPU, also run `./sanitize.sh --mode device` before
calling a change to `src/cuda/` done. CI cannot: no hosted runner has a GPU, so
`compute-sanitizer` never runs there. See [docs/Sanitizers.md](docs/Sanitizers.md).

## Adding a kernel

1. Header and source under the layer that builds it — `src/cpu/`, `src/omp/` or
   `src/cuda/`. `src/` is the only include root, so includes name their layer:
   `#include "core/span.h"`.
2. Name it for the axis it varies — the loop nesting order (`ijk`, `ikj`),
   whether it blocks (`tiled`), which operand layout it expects (`bColMajor`).
   A shared name such as `matmul` would leave a results table unattributable.
3. Register it in `CMakeLists.txt` and add a case to the matching suite in
   `tests/`, checked against `ReferenceGemm` or an exact expected value.
4. For a parallel variant, derive it from the **fastest** serial kernel. Threading
   the naive one reports a speedup that is mostly the optimisation the serial
   variant already had.
5. Add it to an existing benchmark driver if it belongs in an existing
   comparison; a new driver is for a new question.

## Coding style

- **C++17**, formatted with `clang-format` using the repo's
  [`.clang-format`](.clang-format) (Google base style, 2-space indent, 80-column
  limit).
- Three scripts enforce this, and CI runs all of them — run them before
  committing:

  ```bash
  ./scripts/format.sh   # clang-format + strip trailing whitespace
  ./scripts/tidy.sh     # clang-tidy; --fix applies what it can
  ./scripts/docs.sh     # Doxygen API reference; --open shows it
  ```

- The clang-tidy check list in [`.clang-tidy`](.clang-tidy) is curated, not the
  full upstream set. Each disabled family carries the reason it was disabled — if
  a check would help, re-argue it there rather than silencing findings case by
  case.
- This project follows the
  [Google C++ Style Guide](https://google.github.io/styleguide/cppguide.html).
  In practice that means:

  | Thing | Form | Example |
  |-------|------|---------|
  | File names | lowercase with underscores; `.h` and `.cc` | `cpu_gemm.h`, `cpu_gemm.cc` |
  | Types | `PascalCase` | `GemmShape`, `DeviceBuffer` |
  | Functions | `PascalCase` | `GemmIkj()`, `MaxRelativeError()` |
  | Accessors | may be named like variables | `shape.is_valid()`, `shape.a_elements()` |
  | Variables, parameters | `snake_case` | `tile_size`, `a_row` |
  | Class members | `snake_case_` | `data_`, `size_` |
  | Constants | `kPascalCase` | `kDefaultTileSize`, `kSumLanes` |
  | Namespaces | lowercase | `accel`, `accel::bench` |
  | Macros | `SHOUTING_CASE` | `CUDA_CHECK`, `TEST_CASE` |

- Headers carry `#define` guards named `ACCEL_<PATH>_<FILE>_H_`, not
  `#pragma once`. Every file opens with the two-line licence header.
- Comments use `//`, never `///` or `/** */`. The first sentence is the
  summary — Doxygen picks it up through `JAVADOC_AUTOBRIEF`, so no `@brief`
  tag is needed, and no other Doxygen tag is used. Comments explain **intent
  and trade-offs**, not what the code already says. Reserve them for
  architecture, algorithms, non-obvious numerics, and how to use an interface.
- **No using-directives.** `using namespace accel;` is out; a `.cc` may carry
  using-declarations (`using accel::GemmIkj;`) or a namespace alias.
- Two deliberate departures from the guide, both documented where they apply:
  exceptions are used rather than error codes — see below — and `std::size_t`
  carries index arithmetic, because the benchmark shape overflows a 32-bit
  index and the guide's preference for signed types is explicitly softened for
  container indices.
- **Exceptions are used here, which the Google guide forbids.** The rule exists
  for codebases whose existing code is not exception-safe; this one was written
  exception-safe from the first commit, and the whole error-path design depends
  on it. `CUDA_CHECK` throws so that `DeviceBuffer` and `CudaEvent` release
  their handles while unwinding — the alternative, `exit(1)` from inside a check
  macro, leaks every device allocation on every error path. Validate extents
  once at a public entry point and throw `std::invalid_argument`; never inside a
  loop nest. `assert` is not an option either — it compiles out under
  `-DNDEBUG`, which is exactly the build the benchmarks run in.
- Prefer small, focused changes — **one idea per pull request**.

## A number is a claim

If a change touches a published figure, say which machine produced it and re-run
the driver rather than scaling an old number. The tables carry a `Source` column
for exactly this. Timings measured on hardware not available for a re-run belong
in the [Provenance](docs/Benchmarks.md#provenance) section, marked as such.

## Commit & pull request

- Write commit messages as **Conventional Commits** — `type(scope): description`,
  lowercase, imperative, no trailing period, 50 characters or fewer. The types in
  use are `feat`, `fix`, `refactor`, `perf`, `docs`, `build`, `ci` and `test`.
- Second line blank; the body is a bulleted list wrapped at 72 columns that
  explains **why**, not what the diff already shows.
- Keep unrelated changes out of the same commit.
- Before opening a PR: run the tests, the smoke benchmarks and the host
  sanitizer passes, and update any affected docs.

## License

By contributing, you agree that your contributions are licensed under the
project's [MIT License](LICENSE).
