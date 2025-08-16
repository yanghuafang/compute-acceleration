#!/bin/bash

# install-deps-macos.sh — install build and lint dependencies.
#
# The llvm keg supplies clang, clang++, clang-format and clang-tidy. Xcode's
# toolchain is deliberately not used for these: it ships no clang-format or
# clang-tidy at all, and no OpenMP runtime, so a build against it is quietly
# smaller. build-env.sh puts the keg on PATH, since Homebrew keeps it keg-only.
#
# NVIDIA has shipped no macOS CUDA toolkit since 10.2, so the device targets are
# always absent here. The build is complete and testable without them.
#
# Usage:
#   ./install-deps-macos.sh
#   ./install-deps-macos.sh --build-only   # skip the lint and docs tooling

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

build_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --build-only)
      build_only=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--build-only]" >&2
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if ! command -v brew >/dev/null 2>&1; then
  echo "Homebrew not found. Install it from https://brew.sh, then re-run." >&2
  exit 1
fi

# The versioned keg, so macOS and Ubuntu CI run the same clang-format major.
llvm_keg="llvm@${ACCEL_CLANG_VERSION}"

packages=(cmake ninja "${llvm_keg}" libomp)
if [[ "${build_only}" != true ]]; then
  packages+=(doxygen graphviz)
fi

brew install "${packages[@]}"

echo
echo "Done. $("$(brew --prefix "${llvm_keg}")/bin/clang" --version | sed -n '1p')"
