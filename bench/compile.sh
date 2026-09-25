#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
REPO="$(cd "$ROOT/.." && pwd)"
XRE="${XRE:-/usr/local/xpu-4.33.0}"
XDNN="${XDNN:-$REPO/third_party/xdnn}"
libdir="$XRE/lib64"
[[ -d "$libdir" ]] || libdir="$XRE/lib"

g++ -O2 -std=c++14 "$ROOT/k200_gemm_bench.cpp" \
  -I"$XRE/include" -I"$XDNN/include" \
  -L"$libdir" -L"$XDNN/so" \
  -lxpuapi -lxpurt \
  -Wl,-rpath,"$libdir" -Wl,-rpath,"$XDNN/so" \
  -o "$ROOT/k200_gemm_bench"

g++ -O2 -std=c++14 "$ROOT/k200_int4_bench.cpp" \
  -I"$XRE/include" -L"$libdir" -lxpurt \
  -Wl,-rpath,"$libdir" \
  -o "$ROOT/k200_int4_bench"

g++ -O2 -std=c++14 "$ROOT/k200_pipe_bench.cpp" \
  -I"$XRE/include" -L"$libdir" -lxpurt \
  -Wl,-rpath,"$libdir" \
  -o "$ROOT/k200_pipe_bench"

g++ -O2 -std=c++14 "$REPO/setup/k200-smoke.cpp" \
  -I"$XRE/include" -I"$XDNN/include" \
  -L"$libdir" -L"$XDNN/so" \
  -lxpuapi -lxpurt \
  -Wl,-rpath,"$libdir" -Wl,-rpath,"$XDNN/so" \
  -o "$REPO/setup/k200-smoke"

g++ -O2 -std=c++14 "$REPO/examples/vvadd_host.cpp" \
  -I"$XRE/include" -L"$libdir" -lxpurt \
  -Wl,-rpath,"$libdir" \
  -o "$REPO/examples/vvadd_host"

echo "built gemm, int4, pipe, smoke, vvadd_host"
