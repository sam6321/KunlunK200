# Glossary

Short definitions for this card. Later pages use the names and link back here.

## The card

**K200.** Baidu / Kunlunxin inference card, PCI `1d22:3684`. Two dies, 8 GB HBM each. The listings that say R200 or P800 are a different chip.

**Kunlun / Kunlunxin.** The chip family and the company name on later software. First-generation silicon is what this repo targets.

**XPU.** The processor on the card. Software often says “XPU” for one die. This card shows up as `/dev/xpu0` and `/dev/xpu1`.

**XPU1 / KL1.** The first-generation architecture. K200 is KL1. Compiler flag `-mcpu=xpu` is this chip.

**PD.** Processing die. One K200 has two PDs. A benchmark that says “both PDs” is using both dies on the one card.

**KL2 / XPU2 / R200 / KL3 / P800.** Later Kunlun parts. Their wheels, XRE 5, and `-mcpu=xpu2` binaries do not belong on a K200. A program that compiles as XPU2 must not be launched here.

## Software

**XRE.** The userspace runtime (version 4.33 on this card): `xpu_smi`, `libxpurt.so`, headers. It is not the kernel module.

**XPURT.** The runtime library inside XRE (`libxpurt.so`). The Paddle wheel ships an older copy; put 4.33 first on `LD_LIBRARY_PATH`.

**kunlun.ko.** The kernel module. Build it from [kunlun_k200](https://github.com/mobius/kunlun_k200) for Ubuntu 22.04 / kernel 6.8. The DKMS tree inside the XRE tarball is not the one `setup/install.sh` uses.

**XDNN.** The kernel library. `libxpuapi.so` is the file the GEMM bench links. Public `xdnn::fc` covers INT8, INT16, FP16, and FP32. INT4, BF16, and TF32 return `NOT_IMPLEMENT`.

**XTDK.** The device compiler. `clang-10` in that tree emits XPU code. On an x86 host it runs under `qemu-aarch64`.

**.xpu.** A device source file, compiled to a small `KERNEL` blob and launched with `xpu_launch_async`.

**XTCL.** A graph compiler in the vendor SDK. Not required for the benches here.

**XCCL.** Collective communication for more than one card. Not used in this repo.

**Paddle / Paddle-Lite.** The framework that actually runs models on this card. Inference uses `Config.enable_xpu()`. The wheel is `paddlepaddle_xpu` 2.6.1 for CPython 3.10.

## The processor

**CDNN.** The tensor engine on each die: DMA, shuffle, MAC, elementwise, reshape, DMA out. Four clusters, eight cores, each core tied to one engine in the fast GEMM schedule.

**Cluster.** The scalar / SIMD side, separate from CDNN. Four clusters per die. A small add like `simd_vvadd` runs here, not on the MAC.

**Core.** One of eight threads in a cluster. In a GEMM pipe, core 0 is one DMA port, core 2 is the MAC, and so on.

**MAC.** The multiply-accumulate array. It has INT8 and INT16 datapaths. There is no FP32 multiplier and no INT4 datapath that runs faster than INT8.

**DMA / DS / EW / RS.** Engines around the MAC. DMA moves HBM ↔ on-chip SRAM. DS (shuffle) lays a tile into the MAC’s input SRAM. EW is elementwise (activation, scale). RS reshapes. DMA out writes the result back to HBM.

**L1D / L1W / L1E.** On-chip SRAM for the MAC: activations, weights, partials. About 512 KB, 2 MB, and 256 KB per cluster.

**L2.** A smaller staging SRAM between DMA and the L1s (128 KB banks).

**HBM.** The 8 GB external memory on each die. Its PLL is separate from the core clock and stays at 1000 MHz in these notes.

**PLL.** The clock generator. Core PLL0–3 are what `xpu_smi` reports as frequency. `clock/set-mhz` writes those four only.

## Numbers

**TOPS.** Here, `2 · M · N · K / 1e12 / seconds` for a GEMM. The 2 counts a multiply and an add.

**INT8 / INT16.** Native MAC widths. At 900 MHz the whole card is about 230 TOPS INT8 and 58 TOPS INT16 when the GEMM is big enough to fill it.

**INT4.** Packed 4-bit values. `mm_int4` runs, but it issues at the INT8 rate. It is not twice INT8. `xdnn::fc` does not implement it.

**FP16 / FP32.** Storage formats. On this chip a “float” GEMM is an integer MAC plus a scale. The fast FP32 path quantizes to INT16. A true 32-bit product is several INT16 passes and is slower, not faster.

## The host

**EPS-12V.** The card’s 8-pin power plug. It is the CPU pinout (4×12 V, 4×ground), not a GPU PCIe 8-pin. Forcing a GPU plug in puts 12 V on ground.

**BAR / Above 4G.** The card needs a 64-bit PCI memory window. The BIOS option is usually “Above 4G decoding”. Without it the card may not enumerate, especially next to a GPU.

**IOMMU.** Address translation for device DMA. `iommu=pt` on the kernel command line is the setting used here.

**Hugepage.** 2 MB pages the runtime allocates from. This setup reserves 1024 of them (2 GiB).

**714.** The device fault code after an illegal launch. Stop launching. `setup/recover.sh` resets die 0 and reloads the module. More launches while it is faulted return 713.

**810.** A common return from the first `modprobe` after that reset. Wait about 8 seconds and modprobe again.