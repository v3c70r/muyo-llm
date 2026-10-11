#!/usr/bin/env bash
#
# Local sanity checks for muyo-llm.
#
# There is no CI. These run on the development machine before a commit, mirroring Muyo's
# scripts/sanity.sh so a change is judged by the same standard in both repositories.
#
# Usage:
#   scripts/sanity.sh              fast: format of changed lines + a Debug build and the smoke test
#   scripts/sanity.sh format       clang-format on the lines this branch changed
#   scripts/sanity.sh build        configure, build, and run the M0 smoke test
#   scripts/sanity.sh sanitize     build + run the smoke test under ASan/UBSan
#   scripts/sanity.sh tidy         clang-tidy on the changed files (report only)
#   scripts/sanity.sh all          everything above
#
# Environment:
#   BASE=<ref>          comparison ref for the format check (default: origin/main)
#   MUYO_SOURCE_DIR     Muyo checkout to build against (default: thirdparty/Muyo)
#
# The last line is always 'sanity: OK' or 'sanity: FAILED', and the exit status matches it.

set -uo pipefail

ROOT="$(git rev-parse --show-toplevel)"
if [ -z "$ROOT" ] || ! cd "$ROOT"; then
    printf '\033[31msanity: FAILED (cannot enter the repository root)\033[0m\n'
    exit 1
fi
BASE="${BASE:-origin/main}"
JOBS="$(nproc 2>/dev/null || echo 4)"
MUYO_SOURCE_DIR="${MUYO_SOURCE_DIR:-$ROOT/thirdparty/Muyo}"
# Every op the engine can currently run, checked in both build configurations. One line to
# extend when an op lands; the alternative is a list that drifts from main.cpp.
MODES="--smoke --add --matvec --rmsnorm --rope --swiglu --getrows --argmax --softmax --attention --layer"

SRC_GLOBS=('src/*.cpp' 'src/*.h' 'src/**/*.cpp' 'src/**/*.h')

bold() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
ok() { printf '\033[32m%s\033[0m\n' "$*"; }
bad() { printf '\033[31m%s\033[0m\n' "$*"; }

fail=0

cmd_format() {
    bold "format: lines changed vs $BASE"
    local diff tool out
    diff="$(git diff -U0 --no-color "$BASE" -- "${SRC_GLOBS[@]}")"
    if [ -z "$diff" ]; then
        ok "no changed C++ files"
        return
    fi
    tool="$(command -v clang-format-diff || command -v clang-format-diff-18 || true)"
    if [ -z "$tool" ]; then
        bad "clang-format-diff not found; skipping"
        return
    fi
    out="$(printf '%s' "$diff" | "$tool" -p1)"
    if [ -n "$out" ]; then
        bad "format issues:"
        printf '%s\n' "$out"
        fail=1
    else
        ok "format OK"
    fi
}

cmd_build() {
    bold "build: Debug + smoke"
    cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug -DMUYO_SOURCE_DIR="$MUYO_SOURCE_DIR" >/dev/null || { bad "configure FAILED"; fail=1; return; }
    cmake --build build --target muyo_llm -j"$JOBS" >/dev/null || { bad "build FAILED"; fail=1; return; }
    ( cd build && for mode in $MODES; do ./muyo_llm "$mode" || exit 1; done ) || { bad "modes FAILED"; fail=1; return; }
    ok "build OK"
}

cmd_sanitize() {
    bold "sanitize: ASan + UBSan"
    cmake -S . -B build-san -DCMAKE_BUILD_TYPE=Sanitize -DMUYO_SOURCE_DIR="$MUYO_SOURCE_DIR" >/dev/null || { bad "configure FAILED"; fail=1; return; }
    cmake --build build-san --target muyo_llm -j"$JOBS" >/dev/null || { bad "build FAILED"; fail=1; return; }
    local san_env="LSAN_OPTIONS=suppressions=$ROOT/san.supp:print_suppressions=1 UBSAN_OPTIONS=print_stacktrace=1"
    (
        cd build-san
        for mode in $MODES; do
            env $san_env ./muyo_llm "$mode" || exit 1
        done
    ) || { bad "modes FAILED under sanitizers"; fail=1; return; }
    ok "sanitize OK"
}

cmd_tidy() {
    bold "tidy: changed files (report only)"
    if ! command -v clang-tidy >/dev/null; then
        bad "clang-tidy not found; skipping"
        return
    fi
    local files
    files="$(git diff --name-only "$BASE" -- "${SRC_GLOBS[@]}")"
    if [ -z "$files" ]; then
        ok "no changed C++ files"
        return
    fi
    # Report only: the config is Muyo's borrowed one and has never been clean here either.
    printf '%s\n' "$files" | xargs -r clang-tidy -p build 2>&1 | tail -40
    ok "tidy reported (not enforced)"
}

case "${1:-default}" in
    format) cmd_format ;;
    build) cmd_build ;;
    sanitize) cmd_sanitize ;;
    tidy) cmd_tidy ;;
    all) cmd_format; cmd_build; cmd_sanitize; cmd_tidy ;;
    default) cmd_format; cmd_build ;;
    *)
        printf 'usage: %s [format|build|sanitize|tidy|all]\n' "$0" >&2
        exit 2
        ;;
esac

if [ "$fail" -eq 0 ]; then
    printf '\033[32msanity: OK\033[0m\n'
    exit 0
fi
printf '\033[31msanity: FAILED\033[0m\n'
exit 1
