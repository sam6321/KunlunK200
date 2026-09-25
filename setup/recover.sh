#!/usr/bin/env bash
# After a 714, stop launching and run this. soft_rst needs device id 0.
# The first modprobe often returns 810; wait and retry.
set -euo pipefail
XRE="${XRE:-/usr/local/xpu-4.33.0}"
RST="$XRE/tools/kunlun1/soft_rst"
if [[ ! -x "$RST" ]]; then
  echo "missing $RST" >&2
  exit 1
fi
echo "soft_rst 0"
sudo "$RST" 0 || true
echo "rmmod kunlun"
sudo rmmod kunlun || true
sleep 2
echo "modprobe kunlun"
set +e
sudo modprobe kunlun kl1_dma_direct=1 kl1_bounce_pipe=1 kl1_p2p_stub=0
rc=$?
set -e
if [[ "$rc" -ne 0 ]]; then
  echo "modprobe returned $rc; waiting 8s and retrying"
  sleep 8
  sudo modprobe kunlun kl1_dma_direct=1 kl1_bounce_pipe=1 kl1_p2p_stub=0
fi
"$XRE/bin/xpu_smi" | head -20 || true
