# Performance

Measured on one K200, firmware `0001.0008.0016`, XRE 4.33, blower on the sink, HBM left at 1000 MHz. Dies stayed in state `N` for every number below. SI TOPS is `2·M·N·K/1e12/seconds`.

The slide at 1 GHz is 256 INT8, 64 INT16, 16 INT32 for the whole card. At 900 MHz that is 230 / 58 / 14. INT4 is not on the slide. A 2×INT8 guess (461 at 900 MHz) is false: packed INT4 issues at the INT8 rate.

There is no FP32 multiplier. `fc<float>` and the float `conv2d` path quantize with a max and issue one INT16 MAC. Four INT16 partials (a real 32-bit product) measured 9.9 TOPS and cannot pass the INT16 array. The fast FP32 path is the quantized one.

## 900 MHz

N=512, K=4096. Owned pipes gold-check `C = K` on an input of ones.

| Pipe | Shape | TOPS | Library at a comparable launch |
|------|-------|-----:|-------------------------------|
| INT4 `cdnn_mac_int4_pipe` | 4057088×512×4096 | 220.2 | none |
| INT8 pipe2 | 2064384×512×4096 | 212.0 | 228.1 `fc_int8` |
| INT16 pipe2 | 1032192×512×4096 | 52.8 | 57.4 `fc_int16` |
| FP16 pipe2 | 786432×512×4096 | 52.3 | 36.8 `fc<float16>` |
| FP32 pipe | 516096×512×4096 | 34.3 | 24.9 `fc<float>` |

INT8 and INT16 are short of the library (about 93% and 92%). FP16 and FP32 are ahead, because those pipes stay on a packed INT16 MAC while `fc<float16>` does not. INT4 has no library GEMM.

Library shapes at 900 MHz, both dies, for the compare suite (smaller M than the pipes): INT8 228-class on the large pipe shape above; the compare-suite library row at 1000 MHz is in the next table.

## Core clock

Owned pipes below are 1048576×512×4096 except FP32, which is 516096×512×4096 (a float A at M=1048576 is 16 GB and did not allocate). A blank is a fault, not a slow result.

| MHz | word | INT4 | INT8 | INT16 | FP16 | FP32 |
|----:|------|-----:|-----:|------:|-----:|-----:|
| 1000 | `0x6781` | 243.5 | 232.9 | 58.6 | 58.6 | 36.1 |
| 1050 | `0x67E1` | 255.6 | 235.5 | 61.4 | 61.5 | 36.8 |
| 1100 | `0x6841` | 267.7 | 234.8 | 64.2 | 64.2 | 37.3 |
| 1150 | `0x68A1` | 279.8 | 234.7 | 66.0 | 66.0 | 37.8 |
| 1200 | `0x6901` | 291.4 | 236.1 | 67.1 | 67.1 | 38.2 |
| 1250 | `0x6961` | 300.9 | 238.5 | 67.5 | fault | fault |

Library, vintage / modern, on the smaller compare shapes. 1200 left die 1 in state `E` on every call, so 1250 was not run.

| MHz | INT8 65408 | INT16 32704 | FP16 32704 | FP32 16384 |
|----:|------------|-------------|------------|------------|
| 1000 | 240.1 / 238.6 | 62.0 / 61.7 | 40.4 / 39.7 | 25.4 / 25.3 |
| 1050 | 251.7 / 250.1 | 65.0 / 64.8 | 42.4 / 41.6 | 25.8 / 25.7 |
| 1100 | 263.4 / 261.6 | 68.1 / 67.8 | 44.4 / 43.6 | 26.0 / 26.1 |
| 1150 | 274.9 / 273.0 | 71.1 / 70.8 | 46.4 / 45.5 | 26.2 / 26.2 |
| 1200 | die 1 fault | die 1 fault | die 1 fault | die 1 fault |

INT4, INT16, FP16, and the library INT8/INT16 calls still turn core clock into TOPS. The owned INT8 pipe and both FP32 paths go flat above 1 GHz: HBM was not overclocked, and the step time is no longer the MAC. Owned INT4/INT8/INT16 stayed gold through 1250 MHz. FP16 and FP32 stayed clean through 1200. Every library call faulted at 1200. 1300 faulted on the first INT8 warmup and was not retried.

## Gaps

INT8 pipe2 is 212 against library 228 at 900 MHz, and the library keeps scaling with clock while pipe2 stays near 233–238. Above 1 GHz that pipe is waiting on HBM. INT16 stays about 5 TOPS behind at every clean clock. A full K=4096, N=512 INT16 B is 4 MB and L1W is 2 MB, so the INT16 pipe chunks N and reads A twice.

FP16 pipe2 stays on a packed INT16 MAC, so its ceiling is the INT16 library number. FP32 uses `max=1`, which is exact for ones and for values already in [-1, 1]. Other magnitudes saturate. `fc<float>` finds the real max first.

## ResNet50

A fused float ResNet50 (`enable_xpu()`) did one forward in **0.68 ms** on die 0 after the image was already on the device, about 1470 forwards per second. That is about 6 TFLOP/s for ~4.1 billion operations, a fraction of one die’s INT16 peak. Batch-1 ResNet is launch and memory around many small convolutions. An INT8 model would be faster. It would not be 4× this frame rate. INT4 would not add another factor of two on the MAC. See [paddle](paddle.md).
