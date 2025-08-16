#!/bin/bash

# docs.sh — build the Doxygen API reference from the headers under src/.
#
# Doxygen exits 0 after warning about a broken reference, an unknown command, or
# a parameter name that no longer matches the signature. A check that gated on
# its exit status would therefore pass while the site rots, so this gates on the
# warning log the Doxyfile writes instead.
#
# Output lands beside the build tree, not in the checkout: ../<checkout>-build/
# docs/html/index.html.
#
# Modes:
#   (default)   build; report any warning but still produce a readable site
#   --strict    build; exit 1 if Doxygen wrote any warning. What CI runs.
#   --open      build, then open the result in the default browser
#
# --strict is not the default because a half-finished refactor warns constantly,
# and a docs build that refuses to produce output is one nobody runs locally.
# The warnings are printed either way; only the exit status differs.
#
# Usage:
#   ./docs.sh
#   ./docs.sh --open
#   ./docs.sh --strict

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

open_after=false
strict=false
usage='Usage: docs.sh [--open] [--strict]'
while [[ $# -gt 0 ]]; do
  case "$1" in
    --open)
      open_after=true
      shift
      ;;
    --strict)
      strict=true
      shift
      ;;
    -h|--help)
      echo "${usage}" >&2
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "${usage}" >&2
      exit 1
      ;;
  esac
done

if ! accel_have doxygen; then
  echo "doxygen not found. See docs/Install.md." >&2
  echo "  macOS:  brew install doxygen graphviz" >&2
  echo "  Ubuntu: sudo apt install doxygen graphviz" >&2
  exit 1
fi

if ! accel_have dot; then
  echo "Warning: graphviz 'dot' not found; the Doxyfile asks for collaboration" >&2
  echo "diagrams and Doxygen will warn about every one it cannot draw." >&2
fi

echo "Using $(command -v doxygen) — $(doxygen --version)"

out_dir="${ACCEL_BUILD_DIR}/docs"
log_file="${out_dir}/doxygen-warnings.log"
mkdir -p "${out_dir}"
: >"${log_file}"

# The Doxyfile reads both paths from here rather than hardcoding a sibling
# directory, since the build root follows the checkout's own name.
export ACCEL_DOCS_DIR="${out_dir}"

# Run from the repo root: every path in the Doxyfile is relative to it, so that
# the same file works from a shell, from scripts/, and from CI.
(cd "${ACCEL_ROOT}" && doxygen docs/doxygen/Doxyfile)

if [[ -s "${log_file}" ]]; then
  echo "Doxygen reported warnings:" >&2
  cat "${log_file}" >&2
  echo >&2
  echo "Fix the comments above, or adjust docs/doxygen/Doxyfile." >&2
  if [[ "${strict}" == true ]]; then
    exit 1
  fi
  echo "Continuing anyway; pass --strict to make these fatal." >&2
fi

index="${out_dir}/html/index.html"
echo "Documentation written to ${index}"

if [[ "${open_after}" == true ]]; then
  if [[ "$(uname -s)" == Darwin ]]; then
    open "${index}"
  else
    xdg-open "${index}" >/dev/null 2>&1 || echo "Open ${index} manually." >&2
  fi
fi
