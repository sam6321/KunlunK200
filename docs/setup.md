# Setup

Host is Ubuntu 22.04, kernel 6.8, gcc-12. See [hardware](hardware.md) for BIOS, power, and the blower before the card goes in the slot.

## Install

Put the two tarballs where the script can hash them. Checksums are in [THIRD_PARTY.md](../THIRD_PARTY.md).

- XRE 4.33, Ubuntu 20.04 build: [xre-ubuntu_2004_x86_64.tar.gz](https://klx-sdk-release-public.su.bcebos.com/xre/release/4.33.0.1/xre-ubuntu_2004_x86_64.tar.gz). The CentOS zip and XRE 5 are not this card.
- XTDK: [xpu_sdk_v2.0.0.61.tar.gz](https://paddle-wheel.bj.bcebos.com/kunlun/xpu_sdk_v2.0.0.61.tar.gz), passed as `XPU_SDK_TARBALL`. `clang-10` in it is aarch64, so compiles run under `qemu-aarch64`.

```bash
sudo XRE_TARBALL=/path/to/xre-ubuntu_2004_x86_64.tar.gz \
     XPU_SDK_TARBALL=/path/to/xpu_sdk_v2.0.0.61.tar.gz \
     bash setup/install.sh
```

What that does:

1. If `kunlun` is already loaded, it **stops**. It will not `rmmod` a running driver. Pass `--skip-module` to install userspace only.
2. Installs build tools, gcc-12, qemu-user, and an aarch64 sysroot.
3. Reserves 2 GiB of hugepages and adds `iommu=pt` if it is missing. Reboot once so the command line is live.
4. Installs a udev rule so `/dev/xpu*` is mode `0666`.
5. Unpacks **XRE userspace** to `/usr/local/xpu-4.33.0`. The DKMS driver inside that tarball is not installed.
6. Unpacks XTDK to `/usr/local/xtdk` unless `--skip-xtdk`.
7. Builds `k200-pll`, the benches, `k200-smoke`, and the `simd_vvadd` example.
8. Clones [kunlun_k200](https://github.com/mobius/kunlun_k200) at `884178c`, builds `kunlun.ko` with gcc-12, and modprobes it with `kl1_dma_direct=1` and `kl1_bounce_pipe=1`.

```bash
/usr/local/xpu-4.33.0/bin/k200-smoke
```

That copies 4 MB to each die and back, then a 32³ `fc`. Both dies must pass.

`--prefix` changes `/usr/local`. `--skip-module` is how a container or a machine that already has the driver installs the compiler and the benches.

## Environment

`/etc/profile.d/k200-xpu.sh` puts XRE on `PATH` and `LD_LIBRARY_PATH`. Open a new shell after install. The XDNN library used by the GEMM bench is `third_party/xdnn/so`, added by `bench/compile.sh`, not by that profile.

## Compiling a .xpu

XTDK’s clang is an aarch64 binary. `setup/compile-xpu.sh` runs it with `qemu-aarch64-static`:

```bash
bash setup/compile-xpu.sh examples/simd_vvadd.xpu /usr/local/xtdk-build
examples/vvadd_host /usr/local/xtdk-build/simd_vvadd.bin \
  "$(cat /usr/local/xtdk-build/simd_vvadd.entry)"
```

The script prints the mangled entry name into a `.entry` file. Pass that string to the host launcher. `-mcpu=xpu2` objects are a different architecture. Do not launch them.
