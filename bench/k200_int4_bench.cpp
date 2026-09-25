// Native CDNN INT4 D4W4 tiled GEMM.
// Public xdnn::fc INT4 is NOT_IMPLEMENT. This launches cdnn_mac_int4_gemm.bin:
// independent N-strips of 32 (XDNN mac_int4_helper), K/M tiled from HBM,
// grid 4x1 (one MAC per cluster) on both PDs.
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <string>
#include <vector>
#include <xpu/runtime.h>

static int fail(const char *msg, int ret = 0) {
    if (ret)
        std::fprintf(stderr, "FAIL: %s (ret=%d)\n", msg, ret);
    else
        std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static std::vector<unsigned char> read_bin(const char *path) {
    std::ifstream in(path, std::ios::binary);
    return std::vector<unsigned char>((std::istreambuf_iterator<char>(in)),
                                      std::istreambuf_iterator<char>());
}

static int8_t pack_nibbles(int lo, int hi) {
    return (int8_t)(((hi & 0xf) << 4) | (lo & 0xf));
}

static int nibble_s(int v) {
    v &= 0xf;
    return v >= 8 ? v - 16 : v;
}

// Along-K varying signed nibbles. B's second 2048 packed columns (the half
// past a 64 KB L1D slice at mm=32) are xor-3 so a wrapped or clamped tile
// cannot match the full-K dot.
static int8_t vary_a(int row, int p) {
    return pack_nibbles((p * 3 + row) & 0xf, (p * 3 + row + 5) & 0xf);
}

static int8_t vary_b(int p, int col) {
    int lo = (p * 5 + col + 1) & 0xf;
    int hi = (p * 5 + col + 9) & 0xf;
    if (p >= 2048) {
        lo ^= 0x3;
        hi ^= 0x3;
    }
    return pack_nibbles(lo, hi);
}

// Matches kernel b_panel: contiguous [k_tile/2, nn_tile] blocks.
static size_t b_off(int p, int col, int n, int k_tile, int nn_tile) {
    int kt = k_tile / 2;
    int tile_k = p / kt;
    int p_in = p - tile_k * kt;
    int tile_n = col / nn_tile;
    int c_in = col - tile_n * nn_tile;
    int npan = n / nn_tile;
    return ((size_t)(tile_k * npan + tile_n) * kt + p_in) * (size_t)nn_tile + c_in;
}

// mode 0: unique K. mode 1: only the first `span` packed K (clamped cfg).
// mode 2: address wrap, A uses p%span (per-core L1D). mode 3: both A and B wrap.
static double vary_dot(const std::vector<int8_t> &A, const std::vector<int8_t> &B,
                       int packed_k, int n, int row, int col, int reps, int mode,
                       int span, int sgn, int k_tile, int nn_tile) {
    double acc = 0;
    int lim = (mode == 1 && span > 0 && span < packed_k) ? span : packed_k;
    for (int p = 0; p < lim; p++) {
        int pa = p;
        int pb = p;
        if (span > 0 && (mode == 2 || mode == 3))
            pa = p % span;
        if (span > 0 && mode == 3)
            pb = p % span;
        int a = (unsigned char)A[(size_t)row * packed_k + pa];
        int b = (unsigned char)B[b_off(pb, col, n, k_tile, nn_tile)];
        int alo = sgn ? nibble_s(a) : (a & 0xf);
        int ahi = sgn ? nibble_s(a >> 4) : ((a >> 4) & 0xf);
        int blo = sgn ? nibble_s(b) : (b & 0xf);
        int bhi = sgn ? nibble_s(b >> 4) : ((b >> 4) & 0xf);
        acc += (double)alo * blo + (double)ahi * bhi;
    }
    return acc * (double)reps;
}

static int parse_int(const char *s, int *out) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end || v < 0)
        return 1;
    *out = (int)v;
    return 0;
}

struct Pd {
    XPUFunc fn = nullptr;
    void *code = nullptr;
    int8_t *a = nullptr, *b = nullptr;
    float *c = nullptr;
    int m = 0;
};

static int load_kernel(Pd *pd, const char *bin) {
    auto bytes = read_bin(bin);
    if (bytes.empty())
        return fail("empty kernel bin");
    if (xpu_malloc(&pd->code, bytes.size()))
        return fail("malloc code");
    if (xpu_memcpy(pd->code, bytes.data(), bytes.size(), XPU_HOST_TO_DEVICE))
        return fail("H2D code");
    xpu_wait();
    int ret = xpu_create_sd_func(&pd->fn, (uint64_t)(uintptr_t)pd->code,
                                 (uint32_t)bytes.size(), 0, 0, 0,
                                 "cdnn_mac_int4_gemm", true);
    if (ret)
        return fail("xpu_create_sd_func", ret);
    return 0;
}

static int launch_once(Pd *pd, int n, int k, int mm, int nn_tile, int k_tile,
                       int reps, int ncl, int ncore) {
    size_t off = 0;
    xpu_launch_argument_set(&pd->c, 8, off);
    off += 8;
    xpu_launch_argument_set(&pd->a, 8, off);
    off += 8;
    xpu_launch_argument_set(&pd->b, 8, off);
    off += 8;
    xpu_launch_argument_set(&pd->m, 4, off);
    off += 4;
    xpu_launch_argument_set(&n, 4, off);
    off += 4;
    xpu_launch_argument_set(&k, 4, off);
    off += 4;
    xpu_launch_argument_set(&mm, 4, off);
    off += 4;
    xpu_launch_argument_set(&nn_tile, 4, off);
    off += 4;
    xpu_launch_argument_set(&k_tile, 4, off);
    off += 4;
    xpu_launch_argument_set(&reps, 4, off);
    int ret = xpu_launch_config(ncl, ncore);
    if (ret)
        return fail("xpu_launch_config", ret);
    ret = xpu_launch_async(pd->fn);
    if (ret)
        return fail("xpu_launch_async", ret);
    return 0;
}

static void usage(const char *a0) {
    std::fprintf(stderr,
                 "usage: %s <cdnn_mac_int4_gemm.bin> [M N K] [--loops N] "
                 "[--warmup N] [--grid C,N] [--reps N] [--mm N] [--nn-tile N] "
                 "[--k-tile N] [--claimed TOPS] [--nocheck] "
                 "[--pattern const|vary]\n"
                 "defaults: 32768 512 4096  mm=32 nn-tile=256 k-tile=128 reps=1 "
                 "grid=4,1\n",
                 a0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    const char *bin = argv[1];
    int m = 32768, n = 512, k = 4096;
    int loops = 200, warmup = 8, ncl = 4, ncore = 1;
    int mm = 32, nn_tile = 256, k_tile = 128, reps = 1, do_check = 1, vary = 0;
    double claimed = 0;
    int i = 2;
    if (i < argc && argv[i][0] != '-') {
        if (argc < 5 || parse_int(argv[2], &m) || parse_int(argv[3], &n) ||
            parse_int(argv[4], &k)) {
            usage(argv[0]);
            return 2;
        }
        i = 5;
    }
    for (; i < argc; i++) {
        std::string a = argv[i];
        auto need = [&](int *dst) -> int {
            if (i + 1 >= argc)
                return fail("missing value");
            return parse_int(argv[++i], dst);
        };
        if (a == "--loops") {
            if (need(&loops))
                return 1;
        } else if (a == "--warmup") {
            if (need(&warmup))
                return 1;
        } else if (a == "--claimed") {
            if (i + 1 >= argc)
                return fail("missing --claimed");
            claimed = std::atof(argv[++i]);
        } else if (a == "--reps") {
            if (need(&reps))
                return 1;
        } else if (a == "--mm") {
            if (need(&mm))
                return 1;
        } else if (a == "--nn-tile") {
            if (need(&nn_tile))
                return 1;
        } else if (a == "--k-tile") {
            if (need(&k_tile))
                return 1;
        } else if (a == "--nocheck") {
            do_check = 0;
        } else if (a == "--pattern" && i + 1 < argc) {
            std::string p = argv[++i];
            if (p == "vary")
                vary = 1;
            else if (p == "const")
                vary = 0;
            else
                return fail("pattern is const or vary");
        } else if (a == "--grid" && i + 1 < argc) {
            int c = 0, nc = 0;
            if (std::sscanf(argv[++i], "%d,%d", &c, &nc) != 2 || c < 1 ||
                nc < 1)
                return fail("grid");
            ncl = c;
            ncore = nc;
        } else if (a == "--mode") {
            if (i + 1 < argc)
                ++i;
        } else {
            usage(argv[0]);
            return 2;
        }
    }
    if (warmup < 1)
        warmup = 1;
    if (loops < 1)
        loops = 1;
    if (reps < 1)
        reps = 1;
    if (mm < 32 || (mm % 32) || nn_tile < 32 || (nn_tile % 32) || k_tile < 32 ||
        (k_tile % 32))
        return fail("mm/nn-tile/k-tile must be multiples of 32");
    if ((n % nn_tile) || (k % k_tile) || (k % 32) || (n % 32))
        return fail("nn-tile must divide N, k-tile must divide K (and 32)");

    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    int mslot = m0 > m1 ? m0 : m1;
    int split = ncl * mm;
    if ((m0 % split) || (m1 % split && m1 != 0))
        return fail("M/PD must be divisible by ncluster*mm (4*32)");

    // reps>1 replays one tile in L1. reps==1 streams B from HBM and keeps
    // every N partial in L1E until the K tiles are done. Grid 4x8 double-
    // buffers that B panel (L1W only); A stays in one L1D buffer.
    int l1_bufs = (ncore >= 2 && reps == 1) ? 2 : 1;
    size_t l1e = (size_t)mm * nn_tile * 4;
    size_t l1d = (size_t)mm * (k_tile / 2);
    // B is the small side. When the whole B matrix fits in L1W the kernel
    // keeps it there and streams A, and grid 4x8 double-buffers that A panel.
    int b_resident = (reps == 1 && k_tile >= k &&
                      (size_t)(k / 2) * (size_t)n <= 2048 * 1024);
    size_t l1w = b_resident ? (size_t)(k / 2) * (size_t)n
                            : (size_t)(k_tile / 2) * nn_tile * (size_t)l1_bufs;
    // Overlap keeps two groups of four 64 KB A panels in L1D.
    if (b_resident && ncore >= 2)
        l1d = (size_t)8 * (size_t)mm * (size_t)(k_tile / 2);
    size_t l2 = (size_t)mm * (k_tile / 2);
    size_t l2b = (size_t)(k_tile / 2) * nn_tile;
    if (l2b > l2)
        l2 = l2b;
    if (l1e > 256 * 1024 || l1d > 512 * 1024 || l1w > 2048 * 1024 ||
        l2 > 128 * 1024)
        return fail("tile does not fit L1/L2 (drop nn-tile or k-tile)");
    if (reps == 1 && (size_t)mm * (size_t)n * 4 > 256 * 1024)
        return fail("N partials do not fit L1E");

    const int packed_k = k / 2;
    const int8_t packed = pack_nibbles(1, 2);
    std::vector<int8_t> hA((size_t)mslot * packed_k, packed);
    std::vector<int8_t> hB((size_t)packed_k * n, packed);
    if (vary) {
        for (int row = 0; row < mslot; row++)
            for (int p = 0; p < packed_k; p++)
                hA[(size_t)row * packed_k + p] = vary_a(row, p);
        for (int p = 0; p < packed_k; p++)
            for (int col = 0; col < n; col++)
                hB[b_off(p, col, n, k_tile, nn_tile)] = vary_b(p, col);
    }

    Pd pd[2];
    int ms[2] = {m0, m1};
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        if (load_kernel(&pd[d], bin))
            return 1;
        pd[d].m = ms[d];
        if (xpu_malloc((void **)&pd[d].a, (size_t)ms[d] * packed_k) ||
            xpu_malloc((void **)&pd[d].b, (size_t)packed_k * n) ||
            xpu_malloc((void **)&pd[d].c, (size_t)ms[d] * n * 4))
            return fail("malloc gemm");
        if (xpu_memcpy(pd[d].a, hA.data(), (size_t)ms[d] * packed_k,
                       XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), (size_t)packed_k * n,
                       XPU_HOST_TO_DEVICE))
            return fail("H2D");
        std::vector<float> z((size_t)ms[d] * n, 0.f);
        xpu_memcpy(pd[d].c, z.data(), z.size() * 4, XPU_HOST_TO_DEVICE);
        xpu_wait();
    }

    auto once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].fn)
                continue;
            if (xpu_set_device(d))
                return 1;
            if (launch_once(&pd[d], n, k, mm, nn_tile, k_tile, reps, ncl,
                            ncore))
                return 1;
        }
        return 0;
    };

    int nstrips = nn_tile / 32;
    std::printf("K200 INT4 GEMM bin=%s m=%d n=%d k=%d mm=%d nn_tile=%d k_tile=%d "
                "nstrips=%d reps=%d grid=%d,%d npd=%d warmup=%d loops=%d "
                "l1d=%zu l1e=%zu l1w=%zu l2=%zu pattern=%s\n",
                bin, m, n, k, mm, nn_tile, k_tile, nstrips, reps, ncl, ncore,
                npd, warmup, loops, l1d, l1e, l1w, l2, vary ? "vary" : "const");

    for (int w = 0; w < warmup; w++) {
        if (once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        if (!pd[d].fn)
            continue;
        xpu_set_device(d);
        if (xpu_wait())
            return fail("warmup wait");
    }

    // Packed 0x21 = nibbles (1,2). One D4W4 K=32 with dequant 1.0 is 80 per
    // output, times K/32 K-tiles times inner_reps.
    if (do_check) {
        const float expect = 80.f * (float)(k / 32) * (float)reps;
        if (xpu_set_device(0))
            return fail("check device");
        int probe_n = n < 64 ? n : 64;
        std::vector<float> row((size_t)probe_n);
        if (xpu_memcpy(row.data(), pd[0].c, row.size() * 4,
                       XPU_DEVICE_TO_HOST) ||
            xpu_wait())
            return fail("check D2H");
        if (!vary)
            std::printf("INT4 check expect=%.1f C[0]=%.1f C[31]=%.1f", expect,
                        row[0], probe_n > 31 ? row[31] : -1.f);
        else
            std::printf("INT4 check vary C[0]=%.1f C[31]=%.1f", row[0],
                        probe_n > 31 ? row[31] : -1.f);
        if (n > 32 && probe_n > 32)
            std::printf(" C[32]=%.1f", row[32]);
        std::printf("\n");
        std::fflush(stdout);
        if (row[0] == 0.f)
            return fail("mm_int4 produced C=0 (kernel did not write L1E)");
        if (vary) {
            // 64 KB is L1D per core. A panel larger than that only matches
            // full-K gold if core 0 can see the whole cluster L1D.
            const int span = mm > 0 ? (64 * 1024) / mm : 0;
            auto show = [&](const char *name, int mode, int sgn) -> int {
                double e0 = vary_dot(hA, hB, packed_k, n, 0, 0, reps, mode, span,
                                     sgn, k_tile, nn_tile);
                double e31 = probe_n > 31
                                 ? vary_dot(hA, hB, packed_k, n, 0, 31, reps, mode,
                                            span, sgn, k_tile, nn_tile)
                                 : 0;
                double e32 = probe_n > 32
                                 ? vary_dot(hA, hB, packed_k, n, 0, 32, reps, mode,
                                            span, sgn, k_tile, nn_tile)
                                 : 0;
                int hit0 = std::fabs(row[0] - (float)e0) < 0.6;
                int hit31 = probe_n <= 31 ||
                            std::fabs(row[31] - (float)e31) < 0.6;
                int hit32 = probe_n <= 32 ||
                            std::fabs(row[32] - (float)e32) < 0.6;
                std::printf("INT4 vary %s e0=%.1f e31=%.1f e32=%.1f %s\n", name, e0,
                            e31, e32, (hit0 && hit31 && hit32) ? "MATCH" : "miss");
                return hit0 && hit31 && hit32;
            };
            show("full_u", 0, 0);
            int vary_ok = show("full_s", 0, 1);
            if (span > 0 && span < packed_k) {
                show("clamp_u", 1, 0);
                show("clamp_s", 1, 1);
                show("awrap_u", 2, 0);
                show("awrap_s", 2, 1);
                show("abwrap_u", 3, 0);
                show("abwrap_s", 3, 1);
            }
            if (!vary_ok)
                return fail("vary gold mismatch");
        } else if (row[0] != expect) {
            std::printf("WARN: C[0] != expect (layout/acc); continuing for TOPS\n");
        }
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int it = 0; it < loops; it++) {
        if (once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        if (!pd[d].fn)
            continue;
        xpu_set_device(d);
        if (xpu_wait())
            return fail("timed wait");
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                (double)loops;

    double ops = 2.0 * (double)m * (double)n * (double)k * (double)reps;
    double sec = us * 1e-6;
    double tops = (ops / sec) / 1e12;
    double frac = claimed > 0 ? tops / claimed : 0;
    std::printf("RESULT profile=int4 api=native_gemm m=%d n=%d k=%d mm=%d "
                "nn_tile=%d k_tile=%d nstrips=%d reps=%d grid=%d,%d npd=%d "
                "warmup=%d loops=%d latency_us=%.3f tops_si=%.4f "
                "claimed_tops=%.3f frac_claimed=%.3f "
                "note=mm_int4_d4w4_tiled_nstrips\n",
                m, n, k, mm, nn_tile, k_tile, nstrips, reps, ncl, ncore, npd,
                warmup, loops, us, tops, claimed, frac);
    std::printf("JSON {\"profile\":\"int4\",\"api\":\"native_gemm\",\"m\":%d,"
                "\"n\":%d,\"k\":%d,\"mm\":%d,\"nn_tile\":%d,\"k_tile\":%d,"
                "\"nstrips\":%d,\"reps\":%d,\"ncluster\":%d,\"ncore\":%d,"
                "\"npd\":%d,\"warmup\":%d,\"loops\":%d,\"latency_us\":%.3f,"
                "\"tops_si\":%.4f,\"claimed_tops\":%.4f,\"frac_claimed\":%.4f,"
                "\"note\":\"mm_int4_d4w4_tiled_nstrips\"}\n",
                m, n, k, mm, nn_tile, k_tile, nstrips, reps, ncl, ncore, npd,
                warmup, loops, us, tops, claimed, frac);
    std::fflush(stdout);

    for (int d = 0; d < npd; d++) {
        if (!pd[d].fn)
            continue;
        xpu_set_device(d);
        if (pd[d].a)
            xpu_free(pd[d].a);
        if (pd[d].b)
            xpu_free(pd[d].b);
        if (pd[d].c)
            xpu_free(pd[d].c);
        if (pd[d].code)
            xpu_free(pd[d].code);
    }
    return 0;
}
