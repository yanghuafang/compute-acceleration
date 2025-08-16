#!/bin/bash

# sanitize.sh — build and run the test suites under every sanitizer.
#
# Two toolchains are involved and neither covers the other:
#
#   * Host C++ is instrumented by the compiler (ASan / UBSan / TSan). These
#     catch heap overflows, use-after-free, signed overflow and data races in
#     the CPU kernels and in the RAII wrappers' host-side logic.
#   * Device code is checked by NVIDIA's compute-sanitizer, a separate binary
#     that runs an uninstrumented build under a virtual machine. The compiler
#     sanitizers see nothing inside a kernel.
#
# Host sanitizer builds therefore disable the CUDA targets outright: an ASan
# host process also reports false positives from the driver's own allocations.
#
# One further asymmetry: the address+undefined pass covers the OpenMP kernels,
# but the thread pass does not. TSan cannot see an OpenMP barrier unless the
# runtime carries LIBOMP_TSAN_SUPPORT annotations, and neither Homebrew's
# libomp nor a distribution libgomp does, so every worker write reports as a
# race. build-accel.sh drops the OpenMP targets for that pass; see
# docs/Sanitizers.md for how to get real coverage.
#
# Only the test suites are run, never the benchmarks: the default
# 4096x2048x4096 shape takes minutes uninstrumented and roughly 20x that under
# ASan, and it exercises no code path the small test shapes miss.
#
# A pass that cannot run — no CUDA toolkit, no GPU, no compute-sanitizer — is
# reported as skipped and does not fail the run. Use --mode device to make its
# absence visible instead.
#
# Modes:
#   host    address+undefined, then thread
#   device  compute-sanitizer with memcheck, racecheck, initcheck, synccheck
#   all     both (default)
#
# Each pass is a preset in ../CMakePresets.json, so a failure here reproduces
# with one command -- `cmake --workflow --preset tsan`. What this script adds is
# running them all in turn and aggregating the result; what it deliberately does
# not add is any flag of its own, since a pass configured differently here than
# in CI would defeat the point of running it.
#
# Usage:
#   ./sanitize.sh
#   ./sanitize.sh --mode host --keep-going
#   ./sanitize.sh --mode device --arch 86

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

usage() {
  cat <<'USAGE'
Usage: sanitize.sh [options]

Options:
  -m, --mode MODE     host | device | all   (default: all)
  -j, --jobs N        Parallel build jobs (default: detected core count)
      --keep-going    Run every pass even after one fails
      --arch LIST     CMAKE_CUDA_ARCHITECTURES for the device pass
  -h, --help          Show this message

Exit status is non-zero if any pass reported a problem.
USAGE
}

mode=all
jobs="$(accel_detect_jobs)"
keep_going=false
cuda_arch=

while [[ $# -gt 0 ]]; do
  case "$1" in
    -m|--mode)
      mode="${2:?--mode requires a value}"
      shift 2
      ;;
    -j|--jobs)
      jobs="${2:?--jobs requires a value}"
      shift 2
      ;;
    --keep-going)
      keep_going=true
      shift
      ;;
    --arch)
      cuda_arch="${2:?--arch requires a value}"
      shift 2
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

case "${mode}" in
  host|device|all) ;;
  *)
    echo "Invalid --mode '${mode}'" >&2
    exit 1
    ;;
esac

accel_require_positive_int "${jobs}" '--jobs' || exit 1

# ASAN_OPTIONS, UBSAN_OPTIONS, TSAN_OPTIONS and OMP_NUM_THREADS are set by the
# test presets in ../CMakePresets.json, so that a developer running
# `ctest --preset tsan` gets the same runtime configuration as CI without going
# through this script. detect_leaks in particular is stated per platform there
# rather than left to the sanitizer's own default.

executed=0
failures=0
skipped=0

note_failure() {
  echo "$1" >&2
  failures=$((failures + 1))
  if [[ "${keep_going}" != true ]]; then
    echo "${failures} sanitizer pass(es) reported problems." >&2
    exit 1
  fi
}

# $1 is the configure preset, $2 the test preset that goes with it. Build and
# test are kept as separate steps rather than one `cmake --workflow` so that a
# failure still says which of the two it was.
run_host_pass() {
  local configure_preset="$1" test_preset="$2"

  executed=$((executed + 1))
  echo "Host pass: ${configure_preset}"

  if ! "${script_dir}/build-accel.sh" \
      --preset "${configure_preset}" \
      --jobs "${jobs}"; then
    note_failure "${configure_preset}: build failed"
    return 0
  fi

  if ! "${script_dir}/run-tests.sh" --no-build --preset "${test_preset}"; then
    note_failure "${configure_preset}: tests reported a problem"
    return 0
  fi
  echo "Host pass clean: ${configure_preset}"
}

# One tool per invocation, because compute-sanitizer runs exactly one at a time
# and they catch disjoint classes of bug:
#   memcheck   out-of-bounds and misaligned device accesses, leaked allocations
#   racecheck  shared-memory hazards — the __syncthreads() the tiled GEMM needs
#   initcheck  reads of device memory that was never written
#   synccheck  invalid or divergent barrier use
device_tools='memcheck racecheck initcheck synccheck'

run_device_pass() {
  local build_dir="${ACCEL_BUILD_DIR}/compute-sanitizer"

  if ! accel_have compute-sanitizer; then
    echo "compute-sanitizer not in PATH — skipping device pass."
    skipped=$((skipped + 1))
    return 0
  fi

  local build_args=(
    --preset relwithdebinfo
    --build-dir "${build_dir}"
    --no-benchmarks
    --jobs "${jobs}"
  )
  if [[ -n "${cuda_arch}" ]]; then
    build_args+=(--arch "${cuda_arch}")
  fi

  echo "Device pass: building ${build_dir}"
  if ! "${script_dir}/build-accel.sh" "${build_args[@]}"; then
    note_failure "compute-sanitizer: build failed"
    return 0
  fi

  local binary="${build_dir}/bin/test_cuda_kernels"
  if [[ ! -x "${binary}" ]]; then
    echo "test_cuda_kernels was not built (no CUDA toolkit?) — skipping device pass."
    skipped=$((skipped + 1))
    return 0
  fi

  local tool
  for tool in ${device_tools}; do
    executed=$((executed + 1))
    echo "Device pass: compute-sanitizer --tool ${tool}"
    # --error-exitcode makes a detected fault a non-zero status; without it
    # compute-sanitizer exits 0 and only the report text says otherwise, which
    # is how a scripted run comes back green on a real bug.
    if ! compute-sanitizer \
        --tool "${tool}" \
        --error-exitcode 1 \
        --launch-timeout 120 \
        "${binary}"; then
      note_failure "compute-sanitizer --tool ${tool} reported a problem"
      continue
    fi
    echo "Device pass clean: ${tool}"
  done
}

if [[ "${mode}" == host || "${mode}" == all ]]; then
  # The ASan+UBSan test preset differs by platform: LeakSanitizer is supported
  # on Linux and unsupported on macOS/arm64, where detect_leaks=1 aborts every
  # test. CMakePresets.json states both explicitly; this picks the right one.
  if [[ "$(uname -s)" == Darwin ]]; then
    run_host_pass asan-ubsan asan-ubsan-macos
  else
    run_host_pass asan-ubsan asan-ubsan
  fi
  run_host_pass tsan tsan
fi

if [[ "${mode}" == device || "${mode}" == all ]]; then
  run_device_pass
fi

printf '\n'
if [[ "${skipped}" -gt 0 ]]; then
  echo "${skipped} pass(es) skipped."
fi
if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} of ${executed} sanitizer passes reported problems." >&2
  exit 1
fi
if [[ "${executed}" -eq 0 ]]; then
  echo "No sanitizer passes ran." >&2
  exit 1
fi
echo "${executed} sanitizer passes clean."
