#!/usr/bin/env bash
# One library profile, with xpu_smi sampling. Optional --mhz sets the core PLL
# and restores 900 unless --keep-mhz.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
XRE="${XRE:-/usr/local/xpu-4.33.0}"
XDNN="${XDNN:-$REPO/third_party/xdnn}"
export PATH="$XRE/bin:$XRE/tools:$PATH"
export LD_LIBRARY_PATH="$XRE/lib64:$XDNN/so${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
XPU_SMI="${XPU_SMI:-$XRE/bin/xpu_smi}"
PLL="${PLL:-$REPO/clock/k200-pll}"

PROFILE="${1:-}"
if [[ -z "$PROFILE" ]]; then
  echo "usage: $0 <int8|int16|fp16|fp32|int32|int4>[-vintage|-modern] [--mhz N] [--keep-mhz]" >&2
  exit 2
fi
shift
SET_MHZ=""
KEEP=0
BENCH_ARGS=()
while [[ $# -gt 0 ]]; do
  case "$1" in
    --mhz|--set-mhz) SET_MHZ="$2"; shift 2 ;;
    --keep-mhz) KEEP=1; shift ;;
    --) shift; BENCH_ARGS+=("$@"); break ;;
    *) BENCH_ARGS+=("$1"); shift ;;
  esac
done

if [[ ! -x "$ROOT/k200_gemm_bench" ]]; then
  bash "$ROOT/compile.sh"
fi

word_for() {
  case "$1" in
    900) echo 0x66c1 ;;
    1000) echo 0x6781 ;;
    1050) echo 0x67e1 ;;
    1100) echo 0x6841 ;;
    1150) echo 0x68a1 ;;
    1200) echo 0x6901 ;;
    1250) echo 0x6961 ;;
    *) echo "" ;;
  esac
}

if [[ -n "$SET_MHZ" ]]; then
  WORD="$(word_for "$SET_MHZ")"
  if [[ -z "$WORD" ]]; then
    echo "refusing $SET_MHZ MHz (1300 faulted). Use 900-1250 in steps of 50." >&2
    exit 2
  fi
  "$PLL" set "$WORD"
  sleep 3
fi

STAMP="$(date +%Y%m%d-%H%M%S)"
OUT="$ROOT/out/${PROFILE}-${STAMP}"
mkdir -p "$OUT"
"$XPU_SMI" | tee "$OUT/smi_pre.txt" || true

cleanup() {
  if [[ -n "$SET_MHZ" && "$KEEP" -eq 0 ]]; then
    echo "restoring 900 MHz"
    "$PLL" set 0x66c1 || true
  fi
}
trap cleanup EXIT

python3 "$ROOT/monitor.py" --smi "$XPU_SMI" --interval 0.25 \
  --out "$OUT/smi.csv" --summary "$OUT/smi.summary.txt" &
MON_PID=$!
sleep 0.4
set +e
if [[ "$PROFILE" == int4* ]]; then
  XTDK_DIR="${XTDK_DIR:-/usr/local/xtdk}" bash "$REPO/setup/compile-xpu.sh" \
    "$ROOT/kernels/cdnn_mac_int4_pipe.xpu" "${XPU_BUILD:-/usr/local/xtdk-build}"
  "$ROOT/k200_int4_bench" "${XPU_BUILD:-/usr/local/xtdk-build}/cdnn_mac_int4_pipe.bin" \
    "${BENCH_ARGS[@]}" | tee "$OUT/bench.txt"
else
  "$ROOT/k200_gemm_bench" "$PROFILE" "${BENCH_ARGS[@]}" | tee "$OUT/bench.txt"
fi
RC=${PIPESTATUS[0]}
set -e
kill "$MON_PID" 2>/dev/null || true
wait "$MON_PID" 2>/dev/null || true
"$XPU_SMI" | tee "$OUT/smi_post.txt" || true
echo "logdir=$OUT rc=$RC"
exit "$RC"
