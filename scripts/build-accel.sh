#!/bin/bash

# build-accel.sh — configure and build compute-acceleration.
#
# Every named configuration lives in ../CMakePresets.json, not here. This script
# resolves its flags to a preset, then applies anything genuinely ad-hoc as a
# -D override on top. That split is the point: CI runs a preset with no
# overrides at all, so `cmake --workflow --preset <name>` reproduces a CI
# failure exactly, and a flag that only ever appears in a developer's shell
# cannot silently diverge from what CI built.
#
# Builds are out of tree. Every configuration lands under a sibling of the
# checkout — ../compute-acceleration-build/<preset> by default — which
# ACCEL_BUILD_DIR relocates wholesale and --build-dir overrides one at a time.
#
# Modes:
#   (default)          Release, with CUDA and OpenMP if the toolchains are found
#   --sanitizer NAME   instrumented host build; implies --no-cuda, and
#                      `thread` additionally implies --no-openmp
#
# Usage:
#   ./build-accel.sh
#   ./build-accel.sh --sanitizer address+undefined
#   ./build-accel.sh --arch 86
#   ./build-accel.sh --preset cuda-ci        # name a preset directly

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

usage() {
  cat <<'USAGE'
Usage: build-accel.sh [options]

Options:
  -p, --preset NAME     Configure preset to use; overrides --type/--sanitizer.
                        `cmake --list-presets` shows them all.
  -t, --type TYPE       Release | Debug | RelWithDebInfo | MinSizeRel
                        (default: Release)
  -b, --build-dir DIR   Build directory
                        (default: ../<checkout>-build/<preset>)
  -s, --sanitizer NAME  none | address | undefined | address+undefined | thread
                        (default: none). Implies --no-cuda and RelWithDebInfo;
                        `thread` additionally implies --no-openmp.
  -j, --jobs N          Parallel build jobs (default: detected core count)
      --arch LIST       CMAKE_CUDA_ARCHITECTURES, e.g. 86 (default: native)
      --no-cuda         Skip CUDA targets even when a toolkit is present
      --no-openmp       Skip OpenMP targets even when a runtime is present
      --no-tests        Skip the test suites
      --no-benchmarks   Skip the benchmark drivers
      --werror          Treat compiler warnings as errors (the preset default)
      --no-werror       Allow warnings; for iterating on a half-done change
      --clean           Delete the build directory before configuring
  -h, --help            Show this message

Every option below --preset is an override applied on top of the preset. CI
passes none of them; see ../CMakePresets.json for what it does run.

See ./sanitize.sh to build and run under every sanitizer in turn.
USAGE
}

preset=
build_type=Release
build_dir=
sanitizer=none
jobs="$(accel_detect_jobs)"
cuda_arch=
do_clean=false
# Empty means "leave the preset's value alone"; only an explicit flag overrides.
overrides=()

add_override() {
  overrides+=("-D$1")
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--preset)
      preset="${2:?--preset requires a value}"
      shift 2
      ;;
    -t|--type)
      build_type="${2:?--type requires a value}"
      shift 2
      ;;
    -b|--build-dir)
      build_dir="${2:?--build-dir requires a value}"
      shift 2
      ;;
    -s|--sanitizer)
      sanitizer="${2:?--sanitizer requires a value}"
      shift 2
      ;;
    -j|--jobs)
      jobs="${2:?--jobs requires a value}"
      shift 2
      ;;
    --arch)
      cuda_arch="${2:?--arch requires a value}"
      shift 2
      ;;
    --no-cuda)
      add_override ACCEL_CUDA=OFF
      shift
      ;;
    --no-openmp)
      add_override ACCEL_OPENMP=OFF
      shift
      ;;
    --no-tests)
      add_override ACCEL_TESTS=OFF
      shift
      ;;
    --no-benchmarks)
      add_override ACCEL_BENCHMARKS=OFF
      shift
      ;;
    --werror)
      add_override ACCEL_WERROR=ON
      shift
      ;;
    --no-werror)
      add_override ACCEL_WERROR=OFF
      shift
      ;;
    --clean)
      do_clean=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

accel_require_positive_int "${jobs}" '--jobs' || exit 1

# cmake runs from the repo root below, so a relative --build-dir has to be
# resolved against the caller's directory first.
if [[ -n "${build_dir}" && "${build_dir}" != /* ]]; then
  build_dir="$(pwd)/${build_dir}"
fi

# --type and --sanitizer are spellings of a preset. Resolving them here rather
# than assembling -D flags keeps CMakePresets.json the only place that says what
# a named configuration is.
if [[ -z "${preset}" ]]; then
  case "${sanitizer}" in
    none)
      case "${build_type}" in
        Release)        preset=release ;;
        Debug)          preset=debug ;;
        RelWithDebInfo) preset=relwithdebinfo ;;
        MinSizeRel)     preset=minsizerel ;;
        *)
          echo "Invalid --type '${build_type}'" >&2
          exit 1
          ;;
      esac
      ;;
    address+undefined) preset=asan-ubsan ;;
    thread)            preset=tsan ;;
    # No preset of their own: CI never runs a single sanitizer, so one exists
    # only when someone is bisecting which of the two is firing.
    address)
      preset=asan-ubsan
      add_override ACCEL_UBSAN=OFF
      ;;
    undefined)
      preset=asan-ubsan
      add_override ACCEL_ASAN=OFF
      ;;
    *)
      echo "Invalid --sanitizer '${sanitizer}'" >&2
      exit 1
      ;;
  esac

  if [[ "${sanitizer}" != none && "${build_type}" != Release ]]; then
    # The sanitizer presets pin RelWithDebInfo; honour an explicit --type anyway.
    add_override "CMAKE_BUILD_TYPE=${build_type}"
  fi
fi

if [[ -n "${cuda_arch}" ]]; then
  add_override "CMAKE_CUDA_ARCHITECTURES=${cuda_arch}"
fi

# Must match `binaryDir` in ../CMakePresets.json. Derived rather than queried
# because --clean has to know the path before CMake is ever invoked.
if [[ -z "${build_dir}" ]]; then
  build_dir="$(accel_preset_build_dir "${preset}")"
fi

if [[ "${do_clean}" == true && -d "${build_dir}" ]]; then
  echo "Removing ${build_dir}"
  rm -rf -- "${build_dir}"
fi

if ! accel_have cmake; then
  echo "cmake not found. See docs/Install.md." >&2
  echo "  macOS:  brew install cmake" >&2
  echo "  Ubuntu: sudo apt install cmake" >&2
  exit 1
fi

# The presets name Ninja, so it is a hard dependency now rather than an
# opportunistic upgrade. Both install-deps scripts provide it.
if ! accel_have ninja; then
  echo "ninja not found; the presets in CMakePresets.json require it." >&2
  echo "  macOS:  brew install ninja" >&2
  echo "  Ubuntu: sudo apt install ninja-build" >&2
  exit 1
fi

# CMake cannot change the generator of an existing build tree. Reusing one that
# was configured differently would mean the preset name no longer describes what
# is in the directory, so say so and name the remedy instead.
if [[ -f "${build_dir}/CMakeCache.txt" ]]; then
  cached_generator="$(sed -n 's/^CMAKE_GENERATOR:INTERNAL=//p' \
                      "${build_dir}/CMakeCache.txt")"
  if [[ -n "${cached_generator}" && "${cached_generator}" != Ninja ]]; then
    echo "${build_dir} was configured with ${cached_generator}, but the" >&2
    echo "'${preset}' preset builds with Ninja, and CMake cannot switch the" >&2
    echo "generator of an existing tree." >&2
    echo "  ./build-accel.sh --preset ${preset} --clean" >&2
    exit 1
  fi
fi

configure_args=(--preset "${preset}" -B "${build_dir}")
configure_args+=(${overrides[@]+"${overrides[@]}"})

echo "Configuring preset '${preset}' in ${build_dir}"
if [[ ${#overrides[@]} -gt 0 ]]; then
  echo "Overrides: ${overrides[*]}"
fi
# From the repo root: `cmake --preset` reads CMakePresets.json out of the
# current directory, so the same command has to work from a shell, from
# scripts/, and from CI.
(cd "${ACCEL_ROOT}" && cmake "${configure_args[@]}")

echo "Building with ${jobs} jobs"
cmake --build "${build_dir}" --parallel "${jobs}"

echo "Build complete: ${build_dir}/bin"
