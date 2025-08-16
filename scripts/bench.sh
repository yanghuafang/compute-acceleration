#!/bin/bash

# bench.sh — run the benchmark drivers and check what they report.
#
# Every driver verifies its own result against a reference or an exact expected
# value and exits non-zero on a mismatch, so a green run here is a correctness
# result as well as a timing one. A driver whose binary is absent is skipped
# rather than fatal: CUDA and OpenMP targets legitimately do not exist on a
# machine without those toolchains.
#
# Modes:
#   (default)     every backend, at the published shape — several minutes
#   --quick       small shapes; a smoke test, not a measurement
#   --smoke       alias for --quick, for the CI step name
#   --cpu-only    only the single-threaded CPU drivers
#   --omp-only    only the OpenMP drivers
#   --cuda-only   only the CUDA drivers
#
# Usage:
#   ./bench.sh
#   ./bench.sh --quick
#   ./bench.sh --omp-only --m 2048 --k 1024 --n 2048

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

usage() {
  cat <<'USAGE'
Usage: bench.sh [options]

Options:
  -b, --build-dir DIR   Build directory
                        (default: ../<checkout>-build/release)
      --no-build        Use the existing build directory as-is
      --cpu-only        Run only the single-threaded CPU drivers
      --omp-only        Run only the OpenMP drivers
      --cuda-only       Run only the CUDA drivers
      --quick, --smoke  Small shapes: a smoke test, not a measurement
      --m N --k N --n N Override the GEMM extents
  -h, --help            Show this message

Note: the default 4096x2048x4096 shape takes several minutes on
bench_cpu_gemm_row_major, whose naive kernel is the slow baseline by design.
USAGE
}

build_dir="${ACCEL_BUILD_DIR}/release"
do_build=true
run_cpu=true
run_omp=true
run_cuda=true
quick=false
gemm_m=
gemm_k=
gemm_n=

while [[ $# -gt 0 ]]; do
  case "$1" in
    -b|--build-dir)
      build_dir="${2:?--build-dir requires a value}"
      shift 2
      ;;
    --no-build)
      do_build=false
      shift
      ;;
    --cpu-only)
      run_omp=false; run_cuda=false
      shift
      ;;
    --omp-only)
      run_cpu=false; run_cuda=false
      shift
      ;;
    --cuda-only)
      run_cpu=false; run_omp=false
      shift
      ;;
    --quick|--smoke)
      gemm_m=512; gemm_k=256; gemm_n=512; quick=true
      shift
      ;;
    --m)
      gemm_m="${2:?--m requires a value}"
      shift 2
      ;;
    --k)
      gemm_k="${2:?--k requires a value}"
      shift 2
      ;;
    --n)
      gemm_n="${2:?--n requires a value}"
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

if [[ "${do_build}" == true ]]; then
  "${script_dir}/build-accel.sh" --build-dir "${build_dir}"
fi

bin_dir="${build_dir}/bin"
if [[ ! -d "${bin_dir}" ]]; then
  echo "No binaries found in ${bin_dir}." >&2
  echo "Run ./build-accel.sh first, or drop --no-build." >&2
  exit 1
fi

shape_args=()
[[ -n "${gemm_m}" ]] && shape_args+=("--m=${gemm_m}")
[[ -n "${gemm_k}" ]] && shape_args+=("--k=${gemm_k}")
[[ -n "${gemm_n}" ]] && shape_args+=("--n=${gemm_n}")

failures=0
executed=0

run_benchmark() {
  local name="$1"
  shift
  local binary="${bin_dir}/${name}"
  if [[ ! -x "${binary}" ]]; then
    echo "Skipping ${name} (not built)."
    return 0
  fi
  executed=$((executed + 1))
  echo "Running ${name}"
  if ! "${binary}" "$@"; then
    echo "${name} reported a failure." >&2
    failures=$((failures + 1))
  fi
  printf '\n'
}

if [[ "${run_cpu}" == true ]]; then
  run_benchmark bench_cpu_gemm_row_major ${shape_args[@]+"${shape_args[@]}"}
  run_benchmark bench_cpu_gemm_layouts ${shape_args[@]+"${shape_args[@]}"}
fi

if [[ "${run_omp}" == true ]]; then
  run_benchmark bench_omp_gemm_scaling ${shape_args[@]+"${shape_args[@]}"}
  # No shape args: the reduction is sized by --count, whose default is chosen
  # to exceed last-level cache rather than to match the GEMM shape.
  if [[ "${quick}" == true ]]; then
    run_benchmark bench_omp_sum_reduction --count=1048576 --iters=2
  else
    run_benchmark bench_omp_sum_reduction
  fi
fi

if [[ "${run_cuda}" == true ]]; then
  run_benchmark bench_cuda_gemm_tiled ${shape_args[@]+"${shape_args[@]}"}
  run_benchmark bench_cuda_sum_reduction
fi

if [[ "${executed}" -eq 0 ]]; then
  echo "No benchmarks were executed." >&2
  exit 1
fi
if [[ "${failures}" -gt 0 ]]; then
  echo "${failures} of ${executed} benchmarks failed verification." >&2
  exit 1
fi
echo "${executed} benchmarks passed verification."
