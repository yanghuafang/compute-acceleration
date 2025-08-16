#!/bin/bash

# tidy.sh — run clang-tidy over this project's hand-written host sources.
#
# Complements format.sh: clang-format fixes layout, clang-tidy fixes constructs
# (redundant declarations, missing const, implicit conversions, and so on). The
# check list lives in ../.clang-tidy, with the reasoning for what is
# deliberately disabled.
#
# Needs a compile database — CMakeLists.txt sets CMAKE_EXPORT_COMPILE_COMMANDS,
# so ./build-accel.sh produces one in the build directory. Run a build first if
# the file is missing.
#
# Scope: .cc translation units only. clang-tidy cannot parse an nvcc command
# line — --generate-code, -forward-unknown-to-host-compiler and --options-file
# are not clang flags — so a build with CUDA enabled puts entries in the
# database that clang-tidy would choke on. src/cuda is therefore not covered
# here; compute-sanitizer and the CUDA test suite are what check it. See
# docs/Sanitizers.md.
#
# Modes:
#   (default)   report findings; exit 1 if any
#   --fix       apply clang-tidy's automatic fixes in place, then re-format
#
# Usage:
#   ./tidy.sh
#   ./tidy.sh --fix
#   ./tidy.sh ../src/cpu/cpu_gemm.cc   # limit to specific files

set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
source "${script_dir}/build-env.sh" || exit 1

build_dir="${ACCEL_BUILD_DIR}/release"
fix=false
files=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --fix)
      fix=true
      shift
      ;;
    -b|--build-dir)
      build_dir="${2:?--build-dir requires a value}"
      shift 2
      ;;
    -h|--help)
      echo "Usage: $0 [--fix] [--build-dir DIR] [file...]" >&2
      exit 0
      ;;
    *)
      files+=("$1")
      shift
      ;;
  esac
done

clang_tidy="$(accel_tool clang-tidy)"
if ! accel_have "${clang_tidy}"; then
  echo "${clang_tidy} not found. See docs/Install.md." >&2
  echo "  macOS:  brew install llvm@${ACCEL_CLANG_VERSION}" >&2
  echo "  Ubuntu: sudo apt install clang-tidy-${ACCEL_CLANG_VERSION}" >&2
  exit 1
fi

# Name the binary: check names move between majors, so a finding that appears
# on one machine and not another is usually version skew.
echo "Using $(command -v "${clang_tidy}") — $("${clang_tidy}" --version | sed -n '1p')"

compile_db="${build_dir}/compile_commands.json"
if [[ ! -f "${compile_db}" ]]; then
  echo "No compile database at ${compile_db}." >&2
  echo "Run ./build-accel.sh first." >&2
  exit 1
fi

# clang-tidy from Homebrew LLVM does not know the macOS SDK location, so the
# standard library headers would not resolve and every file would fail to parse.
extra_args=()
if [[ "$(uname -s)" == Darwin ]]; then
  sdk_path="$(xcrun --show-sdk-path 2>/dev/null)" || sdk_path=
  if [[ -n "${sdk_path}" ]]; then
    extra_args+=("--extra-arg=-isysroot${sdk_path}")
  fi
fi

if [[ ${#files[@]} -eq 0 ]]; then
  while IFS= read -r f; do
    files+=("$f")
  done < <(find "${ACCEL_ROOT}/src" "${ACCEL_ROOT}/tests" "${ACCEL_ROOT}/benchmarks" \
             -type f -name '*.cc' | sort)
fi

# The quietest way for this check to pass while analyzing nothing: a file the
# compile database does not mention. clang-tidy's response is version- and
# batch-dependent — it may print "Skipping <file>. Compile command not found.",
# guess a command from a sibling entry, or do nothing at all and exit 0. Only
# the first is detectable after the fact, so check membership up front instead.
#
# This is the state the tree is in whenever a source is added without
# re-running CMake, i.e. exactly when a green check is most misleading.
unanalyzable=()
for file in "${files[@]}"; do
  if [[ ! -f "${file}" ]]; then
    unanalyzable+=("${file} — no such file")
    continue
  fi
  abs_file="$(cd "$(dirname "${file}")" && pwd)/$(basename "${file}")"
  # The surrounding quotes keep the match exact, so foo.cc cannot match
  # barfoo.cc.
  if ! grep -qF "\"${abs_file}\"" "${compile_db}"; then
    unanalyzable+=("${abs_file} — not in the compile database")
  fi
done

if [[ ${#unanalyzable[@]} -gt 0 ]]; then
  echo "clang-tidy cannot analyze ${#unanalyzable[@]} of ${#files[@]} file(s):" >&2
  printf '  %s\n' "${unanalyzable[@]}" >&2
  echo "Re-run ./build-accel.sh to refresh ${compile_db}." >&2
  exit 1
fi

tidy_args=(-p "${build_dir}" ${extra_args[@]+"${extra_args[@]}"} --quiet)
if [[ "${fix}" == true ]]; then
  tidy_args+=(--fix --fix-errors)
fi

work_dir="$(mktemp -d "${TMPDIR:-/tmp}/accel-tidy-XXXXXX")"
trap 'rm -rf "${work_dir}"' EXIT

# clang-tidy splits its output across both streams, and the two carry different
# meanings:
#
#   stdout   the findings — one block per diagnostic, empty when there are none
#   stderr   progress and tallies, mixed with the reasons a file could not be
#            checked at all: "Compile command not found", "Error while
#            processing", "unable to handle compilation"
#
# Redirect them separately rather than discarding stderr, so a broken run is
# distinguishable from a clean one. Pass/fail then comes from three independent
# signals: findings, trouble on stderr, and clang-tidy's own exit status.
echo "Running clang-tidy over ${#files[@]} file(s)..."
# `|| tidy_status=$?` rather than reading $? afterwards: a non-zero clang-tidy
# is an expected outcome here, and set -e would abort before it could be read.
tidy_status=0
"${clang_tidy}" "${tidy_args[@]}" "${files[@]}" \
  >"${work_dir}/stdout" 2>"${work_dir}/stderr" || tidy_status=$?

findings=0
if [[ -s "${work_dir}/stdout" ]]; then
  cat "${work_dir}/stdout"
  findings=1
fi

broken=0

# Everything on stderr except the known-benign progress and tally lines is worth
# showing; a narrow filter means an unfamiliar message surfaces rather than
# being swallowed. `|| true` because grep exits 1 when it prints nothing, which
# is the ordinary outcome on a clean run.
grep -vE '^(\[[0-9]+/[0-9]+\] Processing file |[0-9]+ warnings? generated\.|Suppressed [0-9]+ warnings? )' \
  "${work_dir}/stderr" >"${work_dir}/stderr-notable" || true

if [[ -s "${work_dir}/stderr-notable" ]]; then
  echo "clang-tidy wrote to stderr:" >&2
  cat "${work_dir}/stderr-notable" >&2
fi

if grep -qE 'Compile command not found|Error while processing|Error while trying to load a compilation database|unable to handle compilation' \
     "${work_dir}/stderr"; then
  echo "clang-tidy could not analyze one or more files (see stderr above)." >&2
  broken=1
fi

if [[ "${tidy_status}" -ne 0 ]]; then
  echo "clang-tidy exited ${tidy_status}." >&2
  broken=1
fi

if [[ "${fix}" == true ]]; then
  # clang-tidy's rewrites do not respect .clang-format line breaking.
  "${script_dir}/format.sh" >/dev/null
  echo "Applied fixes and re-formatted."
  exit "${broken}"
fi

if [[ "${findings}" -eq 0 && "${broken}" -eq 0 ]]; then
  echo "clang-tidy: no findings (${#files[@]} file(s) analyzed)."
  exit 0
fi
exit 1
