# Hardware

The card is a Kunlun K200, PCI `1d22:3684`, two dies, 8 GB HBM each. Confirm that ID when it arrives. Listings for R200 or P800 are a later chip.

Names are defined in the [glossary](glossary.md).

## Bill of materials

Listings go stale. The part name is what matters. These are the ones that were actually used.

| Part | What arrived | Listing |
|------|----------------|---------|
| Card | Kunlun K200. This listing had reviews; the others did not. | [1005012299475660](https://www.aliexpress.com/item/1005012299475660.html) |
| Blower | Delta **BFB1012VH**, 97×94×33 mm, 12 V 1.80 A, 4-pin PWM. The unit is the VH, about 4500 RPM, not the EH (~5300 RPM) that listing text sometimes claims. Start PWM low. | [1005011744504516](https://www.aliexpress.com/item/1005011744504516.html) |
| Blower power | 1-to-5 Molex plus PWM splitter. 12 V from a Molex lead on the PSU. PWM from a motherboard fan header. Plug the blower into the hub’s **CPU** port so that tach is the one the board reads. | [1005004992202048](https://www.aliexpress.com/item/1005004992202048.html) |
| Card power | Not a listing. One-piece **EPS-12V** from the PSU’s CPU cable, or a cable that really rewires PCIe 8-pin onto EPS. | |
| Shroud | Printed from [cooling/k200_shroud.scad](../cooling/k200_shroud.scad). Two M2 holes, 36 mm apart, on the 8-pin end. | |

Do not buy a “GPU 8-pin” and force it into the socket. EPS is 4×12 V and 4×ground. A PCIe 8-pin is 3×12 V and 5×ground, and the rows are swapped relative to the latch. A keying-only adapter puts 12 V on ground.

The shroud around the socket is a tunnel with a notch over the centre latch. A desktop 4+4 CPU cable is two housings and often will not seat. Use a single 8-pin EPS housing. Do not file the shroud.

The 8-pin is required to enumerate. The slot’s 75 W is not the card’s 12 V feed. Do not power it from the slot alone.

If the card shares a PSU with a high-draw GPU, a sag on that 12 V rail at power-on can keep the K200 from showing up. An unpowered second GPU also frees the PCI memory window the K200 needs.

## Cooling

The stock sink is not enough once a full INT8 GEMM is running. Idle is about 40 °C and 40 W with a blower on the fins. An INT8 or INT4 load is on the order of 150 W.

`cooling/k200_shroud.scad` is a straight duct from a 97 mm blower onto the fin stack at the 8-pin end. Print the flange, screw it on, plug the EPS lead through the opening, then seat the blower. Point the intake away from the neighbouring slot.

`kunlun.ko` does not expose a hardware-monitor temperature, so lm-sensors `fancontrol` cannot follow the card. [cooling/k200-fancontrol](../cooling/k200-fancontrol) reads the hotter die from `xpu_smi -m` and writes a motherboard PWM register.

The curve shipped in [cooling/k200-fancontrol.conf](../cooling/k200-fancontrol.conf) is 52 °C at duty 90 through 78 °C at duty 255. If `xpu_smi` fails it holds duty 220. It rewrites the enable bit every cycle because BIOS Smart Fan takes the header back, and it restores BIOS auto on exit.

```bash
sudo install -m 0755 cooling/k200-fancontrol /usr/local/sbin/k200-fancontrol
sudo cp cooling/k200-fancontrol.conf /etc/k200-fancontrol.conf
sudo cp cooling/k200-fancontrol.service /etc/systemd/system/
sudo systemctl enable --now k200-fancontrol
k200-fancontrol --status
```

`PWM=` in the conf is the header index. Match it with `--status`. The example conf uses `PWM=3`. Boards with an IT8688E Super I/O need the [it87](https://github.com/frankcrawford/it87) module; Ubuntu’s in-tree driver does not expose `pwm` nodes for that chip. The service modprobes `it87`.

## BIOS and OS

- **Above 4G decoding** on, so the 64-bit BAR is assigned.
- **Secure Boot** off. The module is not signed.
- IOMMU may already be on. Add `iommu=pt` to the kernel command line (`setup/install.sh` does this and then you reboot).
- Ubuntu **22.04**, kernel **6.8**. The module is built with **gcc-12**, the same major the 6.8 kernel used. Ubuntu 24.04’s newer kernel and Python 3.12 are a different port.
- Reserve 1024 × 2 MB hugepages (2 GiB). The install script writes that sysctl.

## Boot failures

The card is missing from `lspci`: reseat the EPS plug, confirm it is EPS and not a GPU 8-pin, and check Above 4G if another GPU is installed. A sagging shared 12 V rail looks the same.

`lspci` shows `1d22:3684` but there is no `/dev/xpu*`: the module is not loaded, or the first `modprobe` returned 810. See [troubleshooting](troubleshooting.md).
