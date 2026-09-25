#!/usr/bin/env bash
# Download the two vendor archives the container install hashes.
# XRE is the Ubuntu 20.04 / 4.33 userspace. XTDK comes out of xpu_sdk_v2.0.0.61,
# which is the public package that contains the aarch64 clang-10.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
DEST="${1:-$ROOT/.ci-vendor}"
mkdir -p "$DEST"

XRE_URL="https://klx-sdk-release-public.su.bcebos.com/xre/release/4.33.0.1/xre-ubuntu_2004_x86_64.tar.gz"
XRE_SHA=fef220c80fae1a6cc5cb4df5509c345686e29820fac65a46686ad8eb7d8697f5
SDK_URL="https://paddle-wheel.bj.bcebos.com/kunlun/xpu_sdk_v2.0.0.61.tar.gz"
SDK_SHA=f67da36385e0f436a7f0c27135be5f39e6d8ddac0af658b474843b1a85438e18

grab() {
  local url="$1" sha="$2" out="$3"
  if [[ -f "$out" ]] && echo "$sha  $out" | sha256sum -c -; then
    echo "have $out"
    return
  fi
  echo "fetch $url"
  curl -fL --retry 3 -o "$out.partial" "$url"
  mv "$out.partial" "$out"
  echo "$sha  $out" | sha256sum -c -
}

grab "$XRE_URL" "$XRE_SHA" "$DEST/xre-ubuntu_2004_x86_64_4.33.tar.gz"
grab "$SDK_URL" "$SDK_SHA" "$DEST/xpu_sdk_v2.0.0.61.tar.gz"
echo "DEST=$DEST"
