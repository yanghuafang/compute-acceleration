#!/bin/bash

# format.sh — apply the repo's source formatting rules in place.
#
# Two passes over the same file set:
#   1. clang-format, using the .clang-format at the repo root (Google style,
#      2-space indent, 80 columns).
#   2. strip trailing whitespace, which clang-format leaves inside comments.
#
# Modes:
#   (default)   rewrite files in place
#   --check     report what would change and exit 1 without writing; for CI
#
# Usage:
#   ./format.sh
#   ./format.sh --check

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

check_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)
      check_only=true
      shift
      ;;
    -h|--help)
      echo "Usage: $0 [--check]" >&2
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Usage: $0 [--check]" >&2
      exit 1
      ;;
  esac
done

clang_format="$(accel_tool clang-format)"
if ! accel_have "${clang_format}"; then
  echo "${clang_format} not found. See docs/Install.md." >&2
  echo "  macOS:  brew install llvm@${ACCEL_CLANG_VERSION}" >&2
  echo "  Ubuntu: sudo apt install clang-format-${ACCEL_CLANG_VERSION}" >&2
  exit 1
fi

# Print the binary and version: different clang-format majors disagree on
# layout, so a --check failure on one machine and not another is almost always
# version skew rather than a real formatting slip. Saying which one ran turns
# that from a puzzle into a one-line diagnosis. CI pins 20 for the same reason.
clang_format_version="$("${clang_format}" --version)"
echo "Using $(command -v "${clang_format}") — ${clang_format_version}"
case "${clang_format_version}" in
  *" ${ACCEL_CLANG_VERSION}."*) ;;
  *) echo "Warning: this .clang-format is maintained against clang-format" \
          "${ACCEL_CLANG_VERSION}, which is what CI runs and what install-deps-*" \
          "pin. Another major may report differences that are not real." >&2 ;;
esac

# GNU sed takes -i; BSD sed (macOS) requires an explicit empty suffix.
if sed --version >/dev/null 2>&1; then
  sed_inplace=(sed -i)
else
  sed_inplace=(sed -i '')
fi

roots=("${ACCEL_ROOT}/src" "${ACCEL_ROOT}/tests" "${ACCEL_ROOT}/benchmarks")

list_sources() {
  find "${roots[@]}" -type f \
    \( -name '*.h' -o -name '*.cc' -o -name '*.cu' -o -name '*.cuh' \) -print
}

status=0

if [[ "${check_only}" == true ]]; then
  while IFS= read -r file; do
    if ! "${clang_format}" --style=file "$file" | diff -q - "$file" >/dev/null 2>&1; then
      echo "needs clang-format: ${file#"${ACCEL_ROOT}/"}"
      status=1
    fi
    if grep -qE '[[:blank:]]+$' "$file"; then
      echo "trailing whitespace: ${file#"${ACCEL_ROOT}/"}"
      status=1
    fi
  done < <(list_sources | sort)

  if [[ "${status}" -eq 0 ]]; then
    echo "All files are formatted."
  fi
  exit "${status}"
fi

formatted=0
stripped=0
while IFS= read -r file; do
  "${clang_format}" --style=file -i "$file"
  formatted=$((formatted + 1))
  # After clang-format, so this pass has the final say.
  if grep -qE '[[:blank:]]+$' "$file"; then
    "${sed_inplace[@]}" -E 's/[[:blank:]]+$//' "$file"
    stripped=$((stripped + 1))
  fi
done < <(list_sources | sort)

echo "clang-format applied to ${formatted} file(s); trailing whitespace stripped from ${stripped}."
