#!/bin/bash

# build-env.sh — shared paths and helpers. Sourced, never executed.
#
# No `set -euo pipefail` here: shell options are not scoped to a file, so
# setting them in something sourced changes the caller's shell too.
#
# Portability target is bash 3.2, which is what /bin/bash still is on macOS.
# That rules out associative arrays and `${var,,}`, and means a possibly-empty
# array must expand as `${a[@]+"${a[@]}"}` under `set -u`.

accel_script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ACCEL_ROOT="$(cd "${accel_script_dir}/.." && pwd)"
export ACCEL_SCRIPT_DIR="${accel_script_dir}"
export ACCEL_ROOT

# Build trees live beside the checkout, never inside it: object files in the
# source tree slow every find, grep and editor index that walks it. The name
# derives from the checkout's own, so two clones do not share one build root.
# Export ACCEL_BUILD_DIR to override.
if [[ -z "${ACCEL_BUILD_DIR:-}" ]]; then
  accel_parent="$(dirname "${ACCEL_ROOT}")"
  accel_name="$(basename "${ACCEL_ROOT}")"
  ACCEL_BUILD_DIR="${accel_parent}/${accel_name}-build"
  unset accel_parent accel_name
fi
export ACCEL_BUILD_DIR

accel_have() {
  command -v "$1" >/dev/null 2>&1
}

# clang-format's layout and clang-tidy's check names both move between majors,
# so a --check failure on one machine and not another is usually version skew
# rather than a real finding. Pinning one major here is what makes CI, the
# install scripts and a contributor's editor agree. Override to test a new one.
export ACCEL_CLANG_VERSION="${ACCEL_CLANG_VERSION:-20}"

# Prefer the version-suffixed binary, which is how the distro packages ship;
# fall back to the bare name, which is what the Homebrew keg provides.
accel_tool() {
  if accel_have "$1-${ACCEL_CLANG_VERSION}"; then
    echo "$1-${ACCEL_CLANG_VERSION}"
  else
    echo "$1"
  fi
}

# Build directory for a configure preset. Must match `binaryDir` in
# CMakePresets.json; derived rather than queried because callers need the path
# before CMake has run, and parsing JSON in bash 3.2 without jq is not worth it.
accel_preset_build_dir() {
  echo "${ACCEL_BUILD_DIR}/$1"
}

# macOS: put the Homebrew llvm keg first, so clang, clang++, clang-format and
# clang-tidy all come from one toolchain. Xcode ships neither clang-format nor
# clang-tidy nor an OpenMP runtime, and Homebrew keeps the keg unlinked, so
# without this the lint scripts find nothing and the build silently loses
# OpenMP. Set ACCEL_NO_LLVM_PATH=1 to keep your own toolchain in front.
if [[ "$(uname -s)" == Darwin && -z "${ACCEL_NO_LLVM_PATH:-}" ]] && accel_have brew; then
  # The versioned keg first, so macOS runs the same major as Ubuntu CI; plain
  # `llvm` second, for a checkout set up before the pin existed.
  for accel_keg in "llvm@${ACCEL_CLANG_VERSION}" llvm; do
    accel_llvm_prefix="$(brew --prefix "${accel_keg}" 2>/dev/null)"
    if [[ -x "${accel_llvm_prefix}/bin/clang" ]]; then
      export PATH="${accel_llvm_prefix}/bin:${PATH}"
      break
    fi
  done
  unset accel_keg accel_llvm_prefix

  # CMake's FindOpenMP does not look inside a keg-only Homebrew prefix on its
  # own, so a build with the keg's clang reports "OpenMP runtime not found" and
  # quietly drops the OMP targets. OpenMP_ROOT is the documented hint.
  if [[ -z "${OpenMP_ROOT:-}" ]]; then
    accel_omp_prefix="$(brew --prefix libomp 2>/dev/null)"
    if [[ -d "${accel_omp_prefix}" ]]; then
      export OpenMP_ROOT="${accel_omp_prefix}"
    fi
    unset accel_omp_prefix
  fi
fi

# Validates that $1 is a positive decimal integer, reporting $2 as the context.
accel_require_positive_int() {
  case "$1" in
    '' | *[!0-9]*)
      echo "$2 must be a positive integer, got '$1'" >&2
      return 1
      ;;
  esac
  if [[ "$1" -le 0 ]]; then
    echo "$2 must be a positive integer, got '$1'" >&2
    return 1
  fi
}

# `nproc` does not exist on macOS, and a wrong answer only costs build time.
accel_detect_jobs() {
  if accel_have nproc; then
    nproc
  elif accel_have sysctl; then
    sysctl -n hw.ncpu 2>/dev/null || echo 4
  else
    echo 4
  fi
}
