#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
XRE="${XRE:-/usr/local/xpu-4.33.0}"
g++ -O2 -std=c++14 "$ROOT/k200-pll.cpp" \
  -I"$XRE/include" -L"$XRE/lib64" -lxpurt \
  -Wl,-rpath,"$XRE/lib64" \
  -o "$ROOT/k200-pll"
echo "built $ROOT/k200-pll"
