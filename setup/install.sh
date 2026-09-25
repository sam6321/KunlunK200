#!/usr/bin/env bash
# Install K200 userspace, and optionally the Mobius kunlun.ko.
# Refuses to build or load the module when one is already loaded.
set -euo pipefail

PREFIX=/usr/local
SKIP_MODULE=0
SKIP_XTDK=0
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
MOBIUS_REV=884178c135d0e8fa768a1e501dea0f8d3aaa899d
XRE_SHA=fef220c80fae1a6cc5cb4df5509c345686e29820fac65a46686ad8eb7d8697f5
XTDK_SHA=514e43964556729cc44b56f55a67023d0ecedaa447139b731d6e6b1ce1574f89
XPU_SDK_SHA=f67da36385e0f436a7f0c27135be5f39e6d8ddac0af658b474843b1a85438e18
KUNLUN_SYSFS="${KUNLUN_SYSFS:-/sys/module/kunlun}"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prefix) PREFIX="$2"; shift 2 ;;
    --skip-module) SKIP_MODULE=1; shift ;;
    --skip-xtdk) SKIP_XTDK=1; shift ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

if [[ -e "$KUNLUN_SYSFS" && "$SKIP_MODULE" -eq 0 ]]; then
  echo "kunlun is already loaded ($KUNLUN_SYSFS)." >&2
  echo "Refusing to build or install the module. Re-run with --skip-module," >&2
  echo "or unload it yourself. This script will not rmmod a running driver." >&2
  exit 1
fi

sha256() { sha256sum "$1" | awk '{print $1}'; }

need_tarball() {
  local var="$1" sha="$2" name="$3"
  local path="${!var:-}"
  if [[ -z "$path" || ! -f "$path" ]]; then
    echo "Set $var to $name" >&2
    echo "  expected sha256 $sha" >&2
    echo "  XRE: https://klx-sdk-release-public.su.bcebos.com/xre/release/4.33.0.1/xre-ubuntu_2004_x86_64.tar.gz" >&2
    echo "  XTDK: XPU_SDK_TARBALL=xpu_sdk_v2.0.0.61.tar.gz" >&2
    echo "        https://paddle-wheel.bj.bcebos.com/kunlun/xpu_sdk_v2.0.0.61.tar.gz" >&2
    exit 1
  fi
  local got
  got="$(sha256 "$path")"
  if [[ "$got" != "$sha" ]]; then
    echo "$var sha256 $got != $sha" >&2
    exit 1
  fi
}

if [[ "$(id -u)" -eq 0 ]]; then SUDO=""; else SUDO="sudo"; fi

echo "==> packages"
$SUDO apt-get update
$SUDO apt-get install -y build-essential gcc-12 g++-12 make git pciutils \
  python3 python3-venv python3-pip cmake pkg-config libnuma-dev \
  "linux-headers-$(uname -r)" qemu-user-static gcc-aarch64-linux-gnu

echo "==> hugepages (2 GiB)"
echo 'vm.nr_hugepages = 1024' | $SUDO tee /etc/sysctl.d/99-k200-hugepages.conf >/dev/null
$SUDO sysctl --system >/dev/null || true

if [[ -f /etc/default/grub ]] && ! grep -q 'iommu=pt' /etc/default/grub; then
  echo "==> adding iommu=pt to GRUB (reboot before the card is used for DMA)"
  $SUDO sed -i 's/^GRUB_CMDLINE_LINUX_DEFAULT="/GRUB_CMDLINE_LINUX_DEFAULT="iommu=pt /' /etc/default/grub
  $SUDO update-grub || true
fi

echo "==> udev"
$SUDO mkdir -p /etc/udev/rules.d
$SUDO cp "$ROOT/setup/99-kunlun.rules" /etc/udev/rules.d/99-kunlun.rules
if command -v udevadm >/dev/null 2>&1; then
  $SUDO udevadm control --reload-rules || true
fi

need_tarball XRE_TARBALL "$XRE_SHA" "xre-ubuntu_2004_x86_64_4.33.tar.gz"
echo "==> XRE userspace -> $PREFIX/xpu-4.33.0"
tmp="$(mktemp -d)"
tar -xzf "$XRE_TARBALL" -C "$tmp"
src="$(find "$tmp" -maxdepth 2 -type d -name 'xre-ubuntu_2004_x86_64' | head -1)"
$SUDO mkdir -p "$PREFIX/xpu-4.33.0"
$SUDO cp -a "$src/bin" "$src/include" "$PREFIX/xpu-4.33.0/"
if [[ -d "$src/lib64" ]]; then $SUDO cp -a "$src/lib64" "$PREFIX/xpu-4.33.0/"; fi
if [[ -d "$src/lib" ]]; then $SUDO cp -a "$src/lib" "$PREFIX/xpu-4.33.0/"; fi
# This tarball ships shared objects in so/, not lib64.
if [[ -d "$src/so" && ! -e "$PREFIX/xpu-4.33.0/lib64/libxpurt.so" ]]; then
  $SUDO mkdir -p "$PREFIX/xpu-4.33.0/lib64"
  $SUDO cp -a "$src/so/." "$PREFIX/xpu-4.33.0/lib64/"
fi
if [[ -d "$src/tools" ]]; then $SUDO cp -a "$src/tools" "$PREFIX/xpu-4.33.0/"; fi
rm -rf "$tmp"
echo "$PREFIX/xpu-4.33.0/lib64" | $SUDO tee /etc/ld.so.conf.d/xpu-4.33.conf >/dev/null
$SUDO ldconfig || true

$SUDO tee /etc/profile.d/k200-xpu.sh >/dev/null <<EOF
export XPU_XRE=$PREFIX/xpu-4.33.0
export XTDK=$PREFIX/xtdk
export PATH="\$XPU_XRE/bin:\$XPU_XRE/tools:\$XPU_XRE/tools/kunlun1:\$PATH"
export LD_LIBRARY_PATH="\$XPU_XRE/lib64${LD_LIBRARY_PATH:+:\$LD_LIBRARY_PATH}"
EOF

if [[ "$SKIP_XTDK" -eq 0 ]]; then
  if [[ -n "${XPU_SDK_TARBALL:-}" ]]; then
    need_tarball XPU_SDK_TARBALL "$XPU_SDK_SHA" "xpu_sdk_v2.0.0.61.tar.gz"
    XTDK_TARBALL="$XPU_SDK_TARBALL"
  else
    need_tarball XTDK_TARBALL "$XTDK_SHA" "an XTDK tarball with bin/clang-10"
  fi
  echo "==> XTDK -> $PREFIX/xtdk"
  tmp="$(mktemp -d)"
  tar -xzf "$XTDK_TARBALL" -C "$tmp"
  $SUDO rm -rf "$PREFIX/xtdk"
  $SUDO mkdir -p "$PREFIX/xtdk"
  if [[ -e "$tmp/bin/clang-10" ]]; then
    $SUDO cp -a "$tmp/." "$PREFIX/xtdk/"
  else
    inner="$(find "$tmp" -maxdepth 5 -type f -name clang-10 | head -1)"
    if [[ -z "$inner" ]]; then
      echo "XTDK tarball has no bin/clang-10." >&2
      exit 1
    fi
    $SUDO cp -a "$(dirname "$(dirname "$inner")")/." "$PREFIX/xtdk/"
  fi
  $SUDO chmod a+x "$PREFIX/xtdk/bin/"*
  rm -rf "$tmp"
  if [[ ! -x "$PREFIX/xtdk/bin/clang-10" ]]; then
    echo "clang-10 did not land in $PREFIX/xtdk/bin" >&2
    exit 1
  fi
fi

echo "==> host tools"
export XRE="$PREFIX/xpu-4.33.0"
export XDNN="$ROOT/third_party/xdnn"
bash "$ROOT/clock/compile.sh"
bash "$ROOT/bench/compile.sh"
$SUDO cp "$ROOT/clock/k200-pll" "$ROOT/setup/k200-smoke" "$PREFIX/xpu-4.33.0/bin/" 2>/dev/null || \
  $SUDO cp "$ROOT/clock/k200-pll" "$PREFIX/xpu-4.33.0/bin/"
if [[ -x "$ROOT/setup/k200-smoke" ]]; then
  $SUDO cp "$ROOT/setup/k200-smoke" "$PREFIX/xpu-4.33.0/bin/"
fi

if [[ "$SKIP_XTDK" -eq 0 ]]; then
  echo "==> example .xpu"
  XTDK_DIR="$PREFIX/xtdk" bash "$ROOT/setup/compile-xpu.sh" \
    "$ROOT/examples/simd_vvadd.xpu" "$PREFIX/xtdk-build"
fi

if [[ "$SKIP_MODULE" -eq 1 ]]; then
  echo "skipped kernel module (--skip-module)"
  echo "DONE prefix=$PREFIX"
  exit 0
fi

echo "==> Mobius kunlun.ko @ $MOBIUS_REV"
src_dir="${K200_DRIVER_SRC:-$HOME/k200-src/kunlun_k200}"
if [[ ! -d "$src_dir/.git" ]]; then
  mkdir -p "$(dirname "$src_dir")"
  git clone https://github.com/mobius/kunlun_k200.git "$src_dir"
fi
git -C "$src_dir" fetch --depth 1 origin "$MOBIUS_REV" || git -C "$src_dir" fetch origin
git -C "$src_dir" checkout "$MOBIUS_REV"
make -C "$src_dir/kunlun-driver" modules CC=gcc-12
echo 'options kunlun kl1_p2p_stub=0 kl1_dma_direct=1 kl1_bounce_pipe=1' | \
  $SUDO tee /etc/modprobe.d/kunlun-kl1.conf >/dev/null
$SUDO mkdir -p "/lib/modules/$(uname -r)/updates/dkms"
$SUDO cp "$src_dir/kunlun-driver/kunlun.ko" "/lib/modules/$(uname -r)/updates/dkms/kunlun.ko"
$SUDO depmod -a
$SUDO modprobe kunlun kl1_dma_direct=1 kl1_bounce_pipe=1 kl1_p2p_stub=0 || {
  echo "modprobe returned non-zero; wait 8s and retry (810 is common on the first try)"
  sleep 8
  $SUDO modprobe kunlun kl1_dma_direct=1 kl1_bounce_pipe=1 kl1_p2p_stub=0
}
echo "DONE prefix=$PREFIX"
