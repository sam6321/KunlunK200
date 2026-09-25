#!/usr/bin/env bash
# Same checks as a card-less install: unpack, compile host tools, compile one .xpu.
# Does not load a kernel module and does not open /dev/xpu.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
VENDOR="${CI_VENDOR:-$ROOT/.ci-vendor}"
bash "$ROOT/docker/fetch-ci.sh" "$VENDOR"

docker build -t k200-setup -f "$ROOT/docker/Dockerfile" "$ROOT"
docker run --rm \
  -v "$VENDOR:/vendor:ro" \
  -v "$ROOT:/src" \
  -e XRE_TARBALL=/vendor/xre-ubuntu_2004_x86_64_4.33.tar.gz \
  -e XPU_SDK_TARBALL=/vendor/xpu_sdk_v2.0.0.61.tar.gz \
  k200-setup bash /src/setup/install.sh --prefix /opt/k200 --skip-module

echo "=== refuse when kunlun looks loaded ==="
set +e
docker run --rm \
  -v "$ROOT:/src" \
  -e KUNLUN_SYSFS=/tmp/kunlun-loaded \
  k200-setup bash -c 'set -e; mkdir -p /tmp/kunlun-loaded; bash /src/setup/install.sh --prefix /opt/k200'
rc=$?
set -e
if [[ "$rc" -eq 0 ]]; then
  echo "expected non-zero when kunlun is present" >&2
  exit 1
fi
echo "ci checks passed"
