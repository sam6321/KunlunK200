# Third-party code

MIT in `LICENSE` covers the files written for this repository. It does not cover the pieces below.

## XDNN (`third_party/xdnn/`)

Kunlunxin XDNN for Ubuntu x86_64, package date 20220919.

| File | Bytes | SHA256 |
|------|------:|--------|
| Upstream `xdnn-ubuntu_x86_64.tar.gz` (20220919), unpacked to `include/` and `so/` | 16434613 | `d5ba7b20a1217dd78c7f438aec7ad74c83400c9b241f0b1f725b3037ed72a922` |
| `third_party/xdnn/so/libxpuapi.so` | 60728800 | `e97dc7d2ce0f34df8d6584d1e9eb49e22344ce55c2cd2dcc9ff0e9ca88bff0f5` |

`libxpuapi.so` exports vintage `fc_int8` / `fc_int16` and `xdnn::fc`. The GEMM bench links this file.

## XRE 4.33 and XTDK

Downloaded by `setup/install.sh` and by `docker/fetch-ci.sh`. URLs are in [docs/links.md](docs/links.md).

| Package | Bytes | SHA256 |
|---------|------:|--------|
| `xre-ubuntu_2004_x86_64.tar.gz` (4.33) | 52955921 | `fef220c80fae1a6cc5cb4df5509c345686e29820fac65a46686ad8eb7d8697f5` |
| `xpu_sdk_v2.0.0.61.tar.gz` (contains aarch64 `clang-10`) | 398051742 | `f67da36385e0f436a7f0c27135be5f39e6d8ddac0af658b474843b1a85438e18` |

XRE userspace is installed. The DKMS driver inside that tarball is not used. `kunlun.ko` is built from https://github.com/mobius/kunlun_k200 at `884178c135d0e8fa768a1e501dea0f8d3aaa899d`.
