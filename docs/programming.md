# Programming

KL1 is 4 clusters × 8 cores per die. The GEMM pipes in `bench/kernels/` use one core per CDNN engine:

| Core | Engine |
|-----:|--------|
| 0 | DMA in, port 0 |
| 1 | DMA in, port 1 |
| 2 | MAC |
| 3 | shuffle 0 |
| 4 | shuffle 1 |
| 5 | elementwise |
| 6 | reshape |
| 7 | DMA out |

INT8 keeps B in the 2 MB L1W and reads A once, split across the two DMA ports. INT16 B at K=4096, N=512 is 4 MB, so those pipes chunk N to 256 and read A twice. FP16 is that INT16 pipe with DMA quantization. FP32 is the same with a float source and `max=1`.

`xdnn::fc` implements FP32, FP16, INT8, and INT16. BF16, TF32, and INT4 return `NOT_IMPLEMENT`. Native `mm_int4` does run. It does not beat the INT8 MAC rate.

## Faults

A bad launch returns 714 and further launches return 713. Stop, then run `setup/recover.sh`.

Do not build with `-mcpu=xpu2`. Do not use cluster `GM2LM_ASYNC`, or cluster `GM2SM` / `LM2SM`.

`xfence_unlock` in the XTDK header is two statements without a `do { } while (0)` wrapper. A brace-less `if` runs the unlock on every core, including cores that do not own that engine, and that faults the die. Always brace it. `bench/kernels/cdnn_ref.h` says the same at the top.

## Small example

`examples/simd_vvadd.xpu` is a cluster SIMD add, not a GEMM. Build and check it with the commands in [setup](setup.md). That is the path for a new `.xpu`: compile with `setup/compile-xpu.sh`, load the blob, `xpu_create_cl_func` or `xpu_create_sd_func`, then `xpu_launch_async`.

## Clock

Core frequency is `M × 25 MHz / (P × 2^S)` with S=1, P=3, so `M = Fout_MHz × 6 / 25`. The word is `(P<<13)|(M<<3)|S`. `clock/set-mhz` only knows the steps from 900 to 1250. `freq static` in the vendor tool only knows 500–900. The PLL write is not sticky across `rmmod`. See [benchmarks](benchmarks.md).
