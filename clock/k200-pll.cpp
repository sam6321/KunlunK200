// Dump SYSCON PLLs, or program PLL0–3 with a raw P/M/S word.
// Sequence matches kunlun kl1 static_pll_set (xpu_monitor.c). Does not touch HBM PLL4/5.
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <xpu/runtime.h>
#include <xpu/runtime_ex.h>

static const uint64_t SYSCON = 0x20040000ULL;

static const uint32_t CORE_PMS[4] = {0x0000, 0x000C, 0x0018, 0x2000};
static const uint32_t CORE_BYP[4] = {0x0004, 0x0010, 0x001C, 0x2004};
static const uint32_t CORE_RST[4] = {0x0008, 0x0014, 0x0020, 0x2008};
static const uint32_t HBM_PMS[2] = {0x200C, 0x2018};

static int rr(uint64_t addr, uint32_t *v) {
    int r = xpu_rr(addr, v);
    if (r != 0)
        std::fprintf(stderr, "xpu_rr(0x%llx) failed %d\n", (unsigned long long)addr, r);
    return r;
}

static int rw(uint64_t addr, uint32_t v) {
    int r = xpu_rw(addr, v);
    if (r != 0)
        std::fprintf(stderr, "xpu_rw(0x%llx, 0x%x) failed %d\n",
                     (unsigned long long)addr, v, r);
    return r;
}

static uint32_t core_mhz(uint32_t pll) {
    uint32_t s = pll & 7u;
    uint32_t m = (pll >> 3) & 0x3FFu;
    uint32_t p = (pll >> 13) & 0x3Fu;
    uint32_t den = p << s;
    return den ? (m * 25u) / den : 0;
}

static uint32_t hbm_mhz(uint32_t pll) {
    uint32_t q = (pll >> 3) & 7u;
    uint32_t f = (pll >> 6) & 0x1FFu;
    uint32_t r = (pll >> 15) & 0x3Fu;
    uint32_t den = (r + 1u) << q;
    return den ? (25u * (f + 1u)) / den : 0;
}

static void print_core(int i, uint32_t pll) {
    uint32_t s = pll & 7u;
    uint32_t m = (pll >> 3) & 0x3FFu;
    uint32_t p = (pll >> 13) & 0x3Fu;
    std::printf("  PLL%d  raw=0x%08x  S=%u M=%u P=%u  %u MHz\n",
                i, pll, s, m, p, core_mhz(pll));
}

static int dump(void) {
    uint32_t src = 0;
    if (rr(SYSCON + 0x100, &src))
        return 1;
    std::printf("source_sel @0x100 = 0x%x\n", src);

    for (int i = 0; i < 4; i++) {
        uint32_t pll = 0, byp = 0, rst = 0;
        if (rr(SYSCON + CORE_PMS[i], &pll) ||
            rr(SYSCON + CORE_BYP[i], &byp) ||
            rr(SYSCON + CORE_RST[i], &rst))
            return 1;
        print_core(i, pll);
        std::printf("         bypass=0x%x rst=0x%x\n", byp, rst);
    }
    for (int i = 0; i < 2; i++) {
        uint32_t pll = 0;
        if (rr(SYSCON + HBM_PMS[i], &pll))
            return 1;
        std::printf("  PLL%d  raw=0x%08x  (HBM) %u MHz\n",
                    4 + i, pll, hbm_mhz(pll));
    }
    return 0;
}

// Same sequence as static_pll_set(): refclk, bypass, rst, P/M/S, un-rst, un-bypass, PLL.
static int static_pll_set_word(uint32_t word) {
    std::printf("programming PLL0-3 with 0x%x (%u MHz)\n", word, core_mhz(word));
    for (int i = 0; i < 4; i++) {
        uint32_t val = 0;
        if (rw(SYSCON + 0x100, 0))
            return 1;
        if (rr(SYSCON + CORE_BYP[i], &val))
            return 1;
        if (rw(SYSCON + CORE_BYP[i], val | 0x4u))
            return 1;
        if (rw(SYSCON + CORE_RST[i], 0))
            return 1;
        if (rw(SYSCON + CORE_PMS[i], word))
            return 1;
        if (rw(SYSCON + CORE_RST[i], 1))
            return 1;
        usleep(10);
        if (rr(SYSCON + CORE_BYP[i], &val))
            return 1;
        if (rw(SYSCON + CORE_BYP[i], val & ~(1u << 2)))
            return 1;
        if (rw(SYSCON + 0x100, 0xfu))
            return 1;
    }
    return 0;
}

static void usage(const char *argv0) {
    std::fprintf(stderr,
                 "usage:\n"
                 "  %s dump [dev]\n"
                 "  %s set <hex_word> [dev]\n"
                 "examples:\n"
                 "  %s dump\n"
                 "  %s set 0x6781\n",
                 argv0, argv0, argv0, argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 1;
    }

    int dev = 0;
    const char *cmd = argv[1];
    uint32_t word = 0;

    if (std::strcmp(cmd, "dump") == 0) {
        if (argc >= 3)
            dev = std::atoi(argv[2]);
    } else if (std::strcmp(cmd, "set") == 0) {
        if (argc < 3) {
            usage(argv[0]);
            return 1;
        }
        word = (uint32_t)std::strtoul(argv[2], nullptr, 0);
        if (argc >= 4)
            dev = std::atoi(argv[3]);
        if (word == 0 || core_mhz(word) == 0) {
            std::fprintf(stderr, "refusing word 0x%x (decodes to 0 MHz)\n", word);
            return 1;
        }
    } else {
        usage(argv[0]);
        return 1;
    }

    int dc = 0;
    if (xpu_device_count(&dc) != 0 || dc < 1) {
        std::fprintf(stderr, "xpu_device_count failed\n");
        return 1;
    }
    if (dev < 0 || dev >= dc) {
        std::fprintf(stderr, "dev %d out of range (count=%d)\n", dev, dc);
        return 1;
    }
    int r = xpu_set_device(dev);
    if (r != 0) {
        std::fprintf(stderr, "xpu_set_device(%d) failed %d\n", dev, r);
        return 1;
    }
    std::printf("dev=%d (count=%d)\n", dev, dc);

    if (std::strcmp(cmd, "dump") == 0)
        return dump();

    if (dump())
        return 1;
    if (static_pll_set_word(word))
        return 1;
    usleep(1000);
    std::printf("after set:\n");
    return dump();
}
