#!/bin/bash

# install-deps-ubuntu.sh — install build and lint dependencies.
#
# Everything comes from the distro archive, so clang, clang-format and
# clang-tidy are all the same major and agree about resource directories. A
# clang-tidy older than the compiler that produced the compile database cannot
# resolve that compiler's headers, which is why the versions are not mixed.
#
# The major is pinned (ACCEL_CLANG_VERSION, default 20) because clang-format's
# layout moves between majors: an unpinned formatter means CI and a
# contributor's editor eventually disagree about a file nobody edited. Noble
# does not carry that major, so apt.llvm.org is added when the archive cannot
# supply it. Failing is deliberate if neither can: silently installing whatever
# major is to hand is the exact drift the pin exists to prevent.
#
# CUDA is not installed here; docs/Install.md has the recipe.
#
# Usage:
#   ./install-deps-ubuntu.sh
#   ./install-deps-ubuntu.sh --build-only   # skip the lint and docs tooling
#   ACCEL_CLANG_VERSION=21 ./install-deps-ubuntu.sh   # try another major

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

v="${ACCEL_CLANG_VERSION}"

# libclang-rt-*-dev carries the ASan/UBSan/TSan runtimes clang links against;
# without it the sanitizer builds fail at link time, not configure time. It sits
# in the build set rather than the lint set because a sanitizer build is a
# build: putting it behind --build-only would make every sanitizer job install
# Doxygen and graphviz to get at one runtime library.
packages=(build-essential cmake ninja-build "clang-${v}" "libomp-${v}-dev"
          "libclang-rt-${v}-dev")
if [[ "${build_only}" != true ]]; then
  packages+=("clang-format-${v}" "clang-tidy-${v}" doxygen graphviz)
fi

sudo apt-get update

# apt.llvm.org, and only when the archive cannot supply the pinned major. Added
# by hand rather than by piping llvm.sh into a shell, so that what is trusted is
# visible in this file: one key, one source line, one release.
if ! apt-cache policy "clang-${v}" | grep -q 'Candidate: [^(]'; then
  codename="$(. /etc/os-release && echo "${VERSION_CODENAME}")"
  echo "clang-${v} is not in the archive; adding apt.llvm.org for ${codename}."
  sudo install -d -m 0755 /etc/apt/keyrings
  curl -fsSL https://apt.llvm.org/llvm-snapshot.gpg.key \
    | sudo gpg --dearmor -o /etc/apt/keyrings/llvm.gpg
  echo "deb [signed-by=/etc/apt/keyrings/llvm.gpg]" \
       "http://apt.llvm.org/${codename}/ llvm-toolchain-${codename}-${v} main" \
    | sudo tee /etc/apt/sources.list.d/llvm-${v}.list >/dev/null
  sudo apt-get update
fi

if ! apt-cache policy "clang-${v}" | grep -q 'Candidate: [^(]'; then
  echo "clang-${v} is unavailable from the archive and from apt.llvm.org." >&2
  echo "Set ACCEL_CLANG_VERSION to a major this release carries, and change" >&2
  echo "the pin in scripts/build-env.sh to match so CI agrees." >&2
  exit 1
fi

sudo apt-get install -y --no-install-recommends "${packages[@]}"

echo
echo "Done. $("clang-${v}" --version | sed -n '1p')"
