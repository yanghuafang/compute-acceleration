# Install

`compute-acceleration` depends on CMake 3.25+ and a C++17 compiler. Two
features set that floor: `CMAKE_CUDA_ARCHITECTURES=native` (3.24), which
compiles only for the GPUs actually installed, and the `workflowPresets` in
`CMakePresets.json` (3.25), which are what let a CI failure be reproduced with
one local command. Everything else here builds under 3.18, and every supported
platform ships 3.28 or newer. Everything else is optional and detected at
configure time: without the CUDA toolkit or an OpenMP runtime the build is
smaller but complete and testable.

## Dependencies

```bash
cd scripts
./install-deps-macos.sh     # macOS
./install-deps-ubuntu.sh    # Ubuntu
```

Each platform takes its whole LLVM toolchain from one place — the distro
archive on Ubuntu, the `llvm` keg on macOS — so `clang`, `clang-format` and
`clang-tidy` are always the same major. That matters for `tidy.sh`: a
clang-tidy of a different major than the compiler that produced the compile
database resolves headers against the wrong resource directory, and the real
error arrives buried under thousands of diagnostics.

On macOS the keg also supplies what Xcode does not: `clang-format`,
`clang-tidy`, and an OpenMP runtime. `build-env.sh` puts it on `PATH`; set
`ACCEL_NO_LLVM_PATH=1` to keep your own toolchain in front.

Both scripts take `--build-only` to install just cmake, ninja and a compiler,
skipping the lint and docs tooling.

| Package | Needed for | Absent means |
|---------|-----------|--------------|
| `cmake`, a C++17 compiler | everything | nothing builds |
| OpenMP runtime | the `src/omp` layer | `bench_omp_*` and `test_omp_kernels` are not built |
| CUDA toolkit | the `src/cuda` layer | `bench_cuda_*` and `test_cuda_kernels` are not built |
| `clang-format`, `clang-tidy` | `format.sh`, `tidy.sh` | those scripts exit 1 with an install hint |
| `doxygen`, `graphviz` | `docs.sh` | that script exits 1 with an install hint |

⚠️ **macOS ships no OpenMP runtime.** AppleClang compiles OpenMP but Xcode
carries no `libomp`, so a clean Mac produces a serial-only build — quietly, since
a missing optional toolchain is not an error. `install-deps-macos.sh` installs
Homebrew's, and CMake finds it through `brew --prefix libomp`. Pass
`--require-openmp` to `run-tests.sh` when you want the absence to fail instead.

**macOS has no CUDA toolkit at all.** NVIDIA stopped shipping one after CUDA
10.2, so the device targets are always absent there. Build them on Linux with an
NVIDIA GPU, or on a machine with the toolkit and `--arch` set — see below.

### CUDA on Ubuntu

Not installed by `install-deps-ubuntu.sh`: the toolkit is a multi-gigabyte
download most machines running that script do not need.

```bash
curl -fsSLO https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get install -y --no-install-recommends cuda-nvcc-13-3 cuda-cudart-dev-13-3
export PATH=/usr/local/cuda-13.3/bin:$PATH
```

This project calls into `cudart` and nothing else, so `cuda-nvcc` and
`cuda-cudart-dev` are enough; the full `cuda-toolkit` metapackage pulls in
several gigabytes of libraries it never links.

## Build

```bash
cd scripts
./build-accel.sh
```

Builds are **out of tree**. Every configuration lands under a sibling of the
checkout — `../compute-acceleration-build/<preset>` — so a Release
tree and an instrumented one coexist without putting object files where `grep`
and the editor index have to walk them. Binaries are in `<build-dir>/bin`.

Set `ACCEL_BUILD_DIR` to relocate every configuration at once, or pass
`--build-dir` to place a single one explicitly.

### build-accel.sh options

| Option | Effect |
|--------|--------|
| `-t, --type TYPE` | `Release` (default), `Debug`, `RelWithDebInfo`, `MinSizeRel` |
| `-s, --sanitizer NAME` | `address`, `undefined`, `address+undefined`, `thread`. Implies `--no-cuda` and `RelWithDebInfo`; `thread` also implies `--no-openmp` |
| `--arch LIST` | `CMAKE_CUDA_ARCHITECTURES`, e.g. `86`. Required on a machine with no GPU, where the `native` default cannot resolve |
| `--no-cuda`, `--no-openmp` | Skip a layer even when its toolchain is present |
| `-p, --preset NAME` | Use a preset from `CMakePresets.json` directly; overrides `--type` and `--sanitizer` |
| `--werror`, `--no-werror` | Warnings as errors. On by default, from the presets |
| `--clean` | Delete the build directory before configuring |

### Presets

Every named configuration lives in `CMakePresets.json`, which is what the script
above resolves its flags to and what CI runs unmodified. Using CMake directly is
therefore not a fallback — it is the same build:

```bash
cmake --list-presets                   # every configuration, with descriptions
cmake --workflow --preset release      # configure, build and test in one step
cmake --workflow --preset asan-ubsan   # exactly what the sanitizer job runs
```

A single step at a time, when that is more useful than the whole workflow:

```bash
cmake --preset release
cmake --build --preset release
ctest --preset release-hostonly        # skips the `device` label
```

Presets are also what an IDE reads: VS Code's CMake Tools and CLion both offer
them directly, so the build a contributor selects there is the CI build.

An ad-hoc `-D` still works on top of a preset when a one-off is genuinely what
you want — `cmake --preset release -DACCEL_BENCHMARKS=OFF`.

| CMake option | Default | Effect |
|--------------|---------|--------|
| `ACCEL_CUDA` | `ON` | Build CUDA targets when a toolkit is found |
| `ACCEL_OPENMP` | `ON` | Build OpenMP targets when a runtime is found |
| `ACCEL_TESTS` | `ON` | Build the test suites |
| `ACCEL_BENCHMARKS` | `ON` | Build the benchmark drivers |
| `ACCEL_WERROR` | `OFF` | `-Werror` / `/WX` |
| `ACCEL_ASAN` | `OFF` | AddressSanitizer on host code |
| `ACCEL_UBSAN` | `OFF` | UndefinedBehaviorSanitizer on host code |
| `ACCEL_TSAN` | `OFF` | ThreadSanitizer on host code |

`ACCEL_WERROR` is opt-in rather than default for the reason a learner cloning
this repository should get a build and not a wall of errors: a compiler upgrade
can introduce a new warning at any time. CI turns it on.
