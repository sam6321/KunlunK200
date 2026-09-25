# Benchmarks

SI TOPS is `2 · M · N · K / 1e12 / seconds`. Everyday clock is 900 MHz. To run at another core clock:

```bash
bash bench/profiles/int8.sh --mhz 1000
bash bench/run-pipe.sh int8 --mhz 1150
```

`--mhz` sets PLL0–3 and restores 900 when the run exits. `clock/set-mhz` does the same without a benchmark. 1300 is refused. HBM is left at 1000 MHz (`0x13CB`).

`bench/run.sh` samples `xpu_smi` through `bench/monitor.py` for the length of the launch. Logs land in `bench/out/`.

## Library GEMMs

`bench/k200_gemm_bench` links `third_party/xdnn/so/libxpuapi.so`. Unsiffixed `int8` and `int16` are the vintage `fc_int8` / `fc_int16` entry points. Unsiffixed `fp16` and `fp32` are modern `xdnn::fc`. Add `-vintage` or `-modern` to force the other one.

| Profile | M | N | K | What |
|---------|--:|--:|--:|------|
| `int8` | 65408 | 512 | 4096 | vintage `fc_int8`, both dies |
| `int16` | 32704 | 512 | 4096 | vintage `fc_int16` |
| `fp16` | 32704 | 512 | 4096 | `xdnn::fc<float16>` |
| `fp32` | 16384 | 512 | 4096 | `xdnn::fc<float>` |

```bash
bash bench/profiles/int8.sh
bash bench/profiles/int16.sh
bash bench/profiles/fp16.sh
bash bench/profiles/fp32.sh
```

The library bench does not read C back. A number counts only when `xpu_smi` still shows both dies in state `N` afterwards.

`fc` INT4 returns `NOT_IMPLEMENT`. INT4 is `bench/k200_int4_bench` plus `bench/kernels/cdnn_mac_int4_pipe.xpu`:

```bash
bash bench/run.sh int4 --loops 20 --warmup 2
```

## Owned pipes

`bench/run-pipe.sh` compiles the `.xpu` if needed and checks C against K (ones input).

| Command | Shape at 900 MHz | TOPS |
|---------|------------------|-------------------:|
| `run-pipe.sh int8` | 2064384×512×4096 | 212.0 |
| `run-pipe.sh int16` | 1032192×512×4096 | 52.8 |
| `run-pipe.sh fp16` | 786432×512×4096 | 52.3 |
| `run-pipe.sh fp32` | 516096×512×4096 | 34.3 |

These pipes are `bench/kernels/cdnn_mac_*_pipe*.xpu`. They include `bench/kernels/cdnn_ref.h`. Library numbers above are `bench/k200_gemm_bench` against `third_party/xdnn/so/libxpuapi.so`.

Build only, no card:

```bash
bash bench/compile.sh          # host binaries, needs XRE unpacked
bash setup/compile-xpu.sh bench/kernels/cdnn_mac_int8_pipe2.xpu /tmp/xpu-build
```
