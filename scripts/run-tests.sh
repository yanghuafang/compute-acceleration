#!/bin/bash

# run-tests.sh — build and run the CTest suites.
#
# The suites, their labels and the sanitizer environment live in
# ../CMakePresets.json; this script adds the two things a preset cannot express.
#
# First, the --require-* guards. The CUDA suite exits successfully with a notice
# when no device is visible, so a green run on a machine without a GPU means
# "the host code is correct", not "the kernels were verified". A toolchain CMake
# failed to detect produces the same shape of quiet success: a smaller,
# still-green test run, which is exactly when a passing check is most
# misleading. Pass them in CI.
#
# Second, the build step, so that one command covers configure, build and test.
#
# Modes:
#   (default)   build the preset's configuration, then run its suites
#   --no-build  use the existing build directory as-is
#
# Usage:
#   ./run-tests.sh
#   ./run-tests.sh --require-openmp
#   ./run-tests.sh --preset release-hostonly     # skip the device suite
#   ./run-tests.sh --no-build -- -R omp          # extra args go to ctest

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

usage() {
  cat <<'USAGE'
Usage: run-tests.sh [options] [-- <extra ctest args>]

Options:
  -p, --preset NAME     Test preset to run (default: release).
                        `ctest --list-presets` shows them all;
                        release-hostonly excludes the device suite.
  -b, --build-dir DIR   Run against this directory instead of the preset's own
      --no-build        Use the existing build directory as-is
      --require-cuda    Fail if the CUDA test target is absent
      --require-openmp  Fail if the OpenMP test target is absent
  -h, --help            Show this message
USAGE
}

test_preset=release
build_dir=
do_build=true
require_cuda=false
require_openmp=false
ctest_args=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    -p|--preset)
      test_preset="${2:?--preset requires a value}"
      shift 2
      ;;
    -b|--build-dir)
      build_dir="${2:?--build-dir requires a value}"
      shift 2
      ;;
    --no-build)
      do_build=false
      shift
      ;;
    --require-cuda)
      require_cuda=true
      shift
      ;;
    --require-openmp)
      require_openmp=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      ctest_args=("$@")
      break
      ;;
    *)
      echo "Unknown option: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

if [[ -n "${build_dir}" && "${build_dir}" != /* ]]; then
  build_dir="$(pwd)/${build_dir}"
fi

# Which configure preset a test preset belongs to. Mirrors the `configurePreset`
# fields in ../CMakePresets.json, and is needed only to know what to build:
# `ctest --preset` locates the directory by itself.
configure_preset_for() {
  case "$1" in
    release|release-hostonly)    echo release ;;
    cuda-ci|cuda-ci-hostonly)    echo cuda-ci ;;
    debug)                       echo debug ;;
    asan-ubsan|asan-ubsan-macos) echo asan-ubsan ;;
    tsan)                        echo tsan ;;
    *)                           echo '' ;;
  esac
}

configure_preset="$(configure_preset_for "${test_preset}")"
if [[ -z "${configure_preset}" && -z "${build_dir}" ]]; then
  echo "Don't know which configure preset '${test_preset}' belongs to." >&2
  echo "Add it to configure_preset_for in this script, or point at a tree" >&2
  echo "directly with --build-dir." >&2
  exit 1
fi

if [[ "${do_build}" == true ]]; then
  build_args=(--preset "${configure_preset}")
  if [[ -n "${build_dir}" ]]; then
    build_args+=(--build-dir "${build_dir}")
  fi
  "${script_dir}/build-accel.sh" "${build_args[@]}"
fi

if ! accel_have ctest; then
  echo "ctest not found. It ships with CMake; see docs/Install.md." >&2
  exit 1
fi

# --build-dir addresses a tree directly; otherwise the preset carries both the
# directory and the label filter and sanitizer environment that go with it.
if [[ -n "${build_dir}" ]]; then
  selector=(--test-dir "${build_dir}" --output-on-failure)
  where="${build_dir}"
else
  selector=(--preset "${test_preset}")
  where="preset '${test_preset}'"
fi

# ctest --preset, like cmake --preset, reads CMakePresets.json from the current
# directory, so every invocation below runs from the repo root.
run_ctest() {
  (cd "${ACCEL_ROOT}" && ctest "${selector[@]}" "$@")
}

# The guards ask what was *built*; the run above asks what should *execute*.
# Those differ under release-hostonly, whose label filter hides the device suite
# from `ctest -N` — so --require-cuda has to query the tree, not the preset, or
# it would fail on exactly the CI job it exists to protect.
if [[ -n "${build_dir}" ]]; then
  listing_dir="${build_dir}"
else
  listing_dir="$(accel_preset_build_dir "${configure_preset}")"
fi

# Distinguishing "the target is absent" from "ctest could not run" matters: with
# stderr discarded and only a grep to go on, a build directory ctest cannot read
# reports identically to one that genuinely lacks the target, and the operator
# goes looking for a missing toolchain instead of a broken tree.
require_target() {
  local target="$1" flag="$2" listing
  if ! listing="$(ctest --test-dir "${listing_dir}" -N 2>&1)"; then
    echo "${flag} given but ctest could not read ${listing_dir}:" >&2
    echo "${listing}" >&2
    exit 1
  fi
  if ! grep -q "${target}" <<<"${listing}"; then
    echo "${flag} given but ${target} was not built." >&2
    echo "Configured targets:" >&2
    grep -E '^\s+Test\s+#' <<<"${listing}" >&2 || echo "  (none)" >&2
    exit 1
  fi
}

# `if` rather than `cond && cmd`: under `set -e` an and-list whose test fails
# returns non-zero, and as the last statement of a script that would exit 1.
if [[ "${require_cuda}" == true ]]; then
  require_target test_cuda_kernels --require-cuda
fi
if [[ "${require_openmp}" == true ]]; then
  require_target test_omp_kernels --require-openmp
fi

echo "Running tests: ${where}"
run_ctest ${ctest_args[@]+"${ctest_args[@]}"}
echo "All tests passed."
