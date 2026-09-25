# Troubleshooting

## 714

An illegal launch faults the die. Further launches return 713 and the die shows state `E` in `xpu_smi`. Stop launching.

```bash
bash setup/recover.sh
```

That runs `soft_rst 0` (device id 0, not 1), then `rmmod` and `modprobe`. The first `modprobe` often returns **810**. The script waits 8 seconds and retries. The PLL returns to 900 MHz across the reload.

Do not keep poking 1300 MHz. That word (`0x69C1`) faulted on the first INT8 warmup.

## The card does not enumerate

See [hardware](hardware.md). EPS seated, Above 4G on, and the 12 V rail not sagging under another GPU. Secure Boot off or the module will not load.

## modprobe 810

Wait and retry once. Repeating `rmmod` while `/proc/modules` says `Unloading` is how the module gets stuck. Reboot instead of looping.

## Smoke test fails `fc`

`LD_LIBRARY_PATH` must list `/usr/local/xpu-4.33.0/lib64` and `third_party/xdnn/so`. `k200-smoke` is linked with an rpath to both when `bench/compile.sh` built it. A shell that still has a newer `libxpurt` ahead of 4.33 will load the wrong runtime.

## Paddle scores are flat

NumPy 2. Install `numpy<2` and run again. A uniform 0.001 vector printed at four decimal places looks like zeros.

## Paddle imports, then the card faults

The wheel’s bundled `libxpurt.so.4.31` must not win over 4.33. Check the log line `XPURT ... libxpurt.so`. It should be the 4.33 path.

## Fans ignore the curve

`k200-fancontrol --status`. If there is no `it8688` hwmon node, install the [it87](https://github.com/frankcrawford/it87) module. `PWM=` must be the header the blower is plugged into. BIOS Smart Fan takes the header back unless the daemon keeps rewriting the enable bit. The script does that while it is running.

## Do not

- OTA or OTP the firmware.
- Install XRE 5, or a `paddlepaddle_xpu` wheel built for XPU2 / R200 / P800.
- Launch a `.xpu` that was compiled `-mcpu=xpu2`.
- Treat 1300 MHz as a setting.
