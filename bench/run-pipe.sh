#!/usr/bin/env bash
# Owned CDNN pipes. Shapes are the 900 MHz scoreboard rows.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
XRE="${XRE:-/usr/local/xpu-4.33.0}"
XTDK="${XTDK_DIR:-/usr/local/xtdk}"
OUTB="${XPU_BUILD:-/usr/local/xtdk-build}"
export PATH="$XRE/bin:$PATH"
export LD_LIBRARY_PATH="$XRE/lib64${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

which=""
SET=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mhz) SET+=(--mhz "$2"); shift 2 ;;
    *) which="$1"; shift ;;
  esac
done

if [[ ${#SET[@]} -gt 0 ]]; then
  "$REPO/clock/set-mhz" "${SET[1]}"
  trap '"$REPO/clock/set-mhz" 900' EXIT
fi

build_one() {
  local src="$1"
  local base
  base="$(basename "$src" .xpu)"
  if [[ ! -f "$OUTB/${base}.bin" ]]; then
    XTDK_DIR="$XTDK" bash "$REPO/setup/compile-xpu.sh" "$src" "$OUTB"
  fi
}

run_one() {
  local name="$1" dtype="$2" m="$3" n="$4" k="$5"
  build_one "$ROOT/kernels/${name}.xpu"
  local sym
  sym="$(cat "$OUTB/${name}.entry")"
  echo "=== $name $dtype ${m}x${n}x${k} $sym ==="
  "$ROOT/k200_pipe_bench" "$OUTB/${name}.bin" "$sym" "$dtype" "$m" "$n" "$k" \
    --loops 20 --warmup 2 --grid 4,8
}

[[ -x "$ROOT/k200_pipe_bench" ]] || bash "$ROOT/compile.sh"

case "${which:-all}" in
  int8) run_one cdnn_mac_int8_pipe2 int8 2064384 512 4096 ;;
  int16) run_one cdnn_mac_int16_pipe2 int16 1032192 512 4096 ;;
  fp16) run_one cdnn_mac_fp16_pipe2 fp16 786432 512 4096 ;;
  fp32) run_one cdnn_mac_fp32_pipe fp32 516096 512 4096 ;;
  all)
    run_one cdnn_mac_int8_pipe2 int8 2064384 512 4096
    run_one cdnn_mac_int16_pipe2 int16 1032192 512 4096
    run_one cdnn_mac_fp16_pipe2 fp16 786432 512 4096
    run_one cdnn_mac_fp32_pipe fp32 516096 512 4096
    ;;
  *) echo "usage: $0 [int8|int16|fp16|fp32|all] [--mhz N]" >&2; exit 2 ;;
esac
