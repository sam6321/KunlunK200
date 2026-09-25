// Like-for-like CDNN pipe GEMM. No xdnn::fc.
// Kernels: cdnn_mac_{int8,int16,fp16,fp32}_pipe. C is fp32. Const ones.
// fp16 and fp32 are DMA-quantized to int16 (max=1), then one mm_int16.
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

static int parse_int(const char *s, int *out) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end || v < 0)
        return 1;
    *out = (int)v;
    return 0;
}

enum Dtype { DT_INT8, DT_INT16, DT_FP16, DT_FP32 };

static int dtype_of(const char *s, Dtype *d) {
    if (!std::strcmp(s, "int8"))
        *d = DT_INT8;
    else if (!std::strcmp(s, "int16"))
        *d = DT_INT16;
    else if (!std::strcmp(s, "fp16"))
        *d = DT_FP16;
    else if (!std::strcmp(s, "fp32"))
        *d = DT_FP32;
    else
        return 1;
    return 0;
}

static int elem_size(Dtype d) {
    if (d == DT_INT8)
        return 1;
    if (d == DT_INT16 || d == DT_FP16)
        return 2;
    return 4;
}

struct Pd {
    XPUFunc fn = nullptr;
    void *code = nullptr;
    void *a = nullptr;
    void *b = nullptr;
    float *c = nullptr;
    int m = 0;
};

static void release(Pd *pd) {
    if (pd->a)
        xpu_free(pd->a);
    if (pd->b)
        xpu_free(pd->b);
    if (pd->c)
        xpu_free(pd->c);
    if (pd->code)
        xpu_free(pd->code);
    pd->a = nullptr;
    pd->b = nullptr;
    pd->c = nullptr;
    pd->code = nullptr;
    pd->fn = nullptr;
}

static int load_kernel(Pd *pd, const char *bin, const char *symbol) {
    auto bytes = read_bin(bin);
    if (bytes.empty())
        return fail("empty kernel bin");
    if (xpu_malloc(&pd->code, bytes.size()))
        return fail("malloc code");
    if (xpu_memcpy(pd->code, bytes.data(), bytes.size(), XPU_HOST_TO_DEVICE))
        return fail("H2D code");
    xpu_wait();
    int ret = xpu_create_sd_func(&pd->fn, (uint64_t)(uintptr_t)pd->code,
                                 (uint32_t)bytes.size(), 0, 0, 0, symbol, true);
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

static int zero_device(void *dst, size_t bytes) {
    std::vector<char> z(32u << 20, 0);
    for (size_t off = 0; off < bytes;) {
        size_t n = bytes - off;
        if (n > z.size())
            n = z.size();
        if (xpu_memcpy((char *)dst + off, z.data(), n, XPU_HOST_TO_DEVICE))
            return 1;
        off += n;
    }
    return 0;
}

int main(int argc, char **argv) {
    if (argc < 7) {
        std::fprintf(stderr,
                     "usage: %s <bin> <symbol> <int8|int16|fp16|fp32> M N K "
                     "[--loops N] [--warmup N] [--grid C,N] [--split-k]\n",
                     argv[0]);
        return 2;
    }
    const char *bin = argv[1];
    const char *symbol = argv[2];
    Dtype dt;
    if (dtype_of(argv[3], &dt))
        return fail("dtype");
    int m = 0, n = 0, k = 0;
    if (parse_int(argv[4], &m) || parse_int(argv[5], &n) || parse_int(argv[6], &k))
        return fail("M N K");
    int loops = 20, warmup = 2, ncl = 4, ncore = 8, split_k = 0;
    int mm = 32, nn_tile = 64, k_tile = k, reps = 1;
    for (int i = 7; i < argc; i++) {
        std::string a = argv[i];
        if (a == "--loops" && i + 1 < argc)
            parse_int(argv[++i], &loops);
        else if (a == "--warmup" && i + 1 < argc)
            parse_int(argv[++i], &warmup);
        else if (a == "--grid" && i + 1 < argc) {
            if (std::sscanf(argv[++i], "%d,%d", &ncl, &ncore) != 2)
                return fail("grid");
        } else if (a == "--split-k")
            split_k = 1;
        else
            return fail("unknown arg");
    }
    if (loops < 1)
        loops = 1;
    if (warmup < 1)
        warmup = 1;

    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    int mslot = m0 > m1 ? m0 : m1;
    int split = ncl * mm;
    if ((m0 % split) || (m1 % split && m1 != 0))
        return fail("M/PD must be divisible by ncluster*mm");
    if ((n % 32) || (k % 32))
        return fail("N and K must be multiples of 32");

    int es = elem_size(dt);
    size_t a_elems = (size_t)mslot * (size_t)k;
    size_t b_elems = (size_t)k * (size_t)n;
    std::vector<char> hA(a_elems * (size_t)es);
    std::vector<char> hB(b_elems * (size_t)es);
    if (dt == DT_INT8) {
        std::memset(hA.data(), 1, hA.size());
        std::memset(hB.data(), 1, hB.size());
        if (split_k) {
            auto *a = (int8_t *)hA.data();
            int half = k / 2;
            for (size_t row = 0; row < a_elems / (size_t)k; row++) {
                for (int col = half; col < k; col++)
                    a[row * (size_t)k + col] = 2;
            }
        }
    } else if (dt == DT_INT16) {
        auto *a = (int16_t *)hA.data();
        auto *b = (int16_t *)hB.data();
        for (size_t i = 0; i < a_elems; i++)
            a[i] = 1;
        for (size_t i = 0; i < b_elems; i++)
            b[i] = 1;
    } else if (dt == DT_FP16) {
        auto *a = (uint16_t *)hA.data();
        auto *b = (uint16_t *)hB.data();
        for (size_t i = 0; i < a_elems; i++)
            a[i] = 0x3c00;
        for (size_t i = 0; i < b_elems; i++)
            b[i] = 0x3c00;
    } else {
        auto *a = (float *)hA.data();
        auto *b = (float *)hB.data();
        for (size_t i = 0; i < a_elems; i++)
            a[i] = 1.f;
        for (size_t i = 0; i < b_elems; i++)
            b[i] = 1.f;
    }

    const char *name = argv[3];
    Pd pd[2];
    int ms[2] = {m0, m1};
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        if (load_kernel(&pd[d], bin, symbol)) {
            release(&pd[d]);
            return 1;
        }
        pd[d].m = ms[d];
        size_t ab = (size_t)ms[d] * (size_t)k * (size_t)es;
        size_t bb = (size_t)k * (size_t)n * (size_t)es;
        size_t cb = (size_t)ms[d] * (size_t)n * 4u;
        if (xpu_malloc(&pd[d].a, ab) || xpu_malloc(&pd[d].b, bb) ||
            xpu_malloc((void **)&pd[d].c, cb)) {
            release(&pd[d]);
            return fail("malloc gemm");
        }
        if (xpu_memcpy(pd[d].a, hA.data(), ab, XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), bb, XPU_HOST_TO_DEVICE) ||
            zero_device(pd[d].c, cb)) {
            release(&pd[d]);
            return fail("H2D");
        }
        xpu_wait();
    }

    auto once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].fn)
                continue;
            if (xpu_set_device(d))
                return 1;
            if (launch_once(&pd[d], n, k, mm, nn_tile, k_tile, reps, ncl, ncore))
                return 1;
        }
        return 0;
    };

    std::printf("K200 pipe dtype=%s bin=%s m=%d n=%d k=%d grid=%d,%d npd=%d "
                "warmup=%d loops=%d\n",
                name, bin, m, n, k, ncl, ncore, npd, warmup, loops);
    std::fflush(stdout);

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

    if (xpu_set_device(0))
        return fail("check device");
    int probe_n = n < 64 ? n : 64;
    std::vector<float> row((size_t)probe_n);
    if (xpu_memcpy(row.data(), pd[0].c, row.size() * 4, XPU_DEVICE_TO_HOST) ||
        xpu_wait())
        return fail("check D2H");
    float expect = split_k ? (float)(k / 2) + 2.f * (float)(k / 2) : (float)k;
    float tol = (dt == DT_FP16 || dt == DT_FP32) ? expect * 1e-3f + 0.5f : 0.5f;
    std::printf("%s check expect=%.3f C[0]=%.3f C[31]=%.3f C[32]=%.3f\n", name,
                expect, row[0], probe_n > 31 ? row[31] : -1.f,
                probe_n > 32 ? row[32] : -1.f);
    std::fflush(stdout);
    auto close_enough = [&](float v) { return std::fabs(v - expect) <= tol; };
    if (!close_enough(row[0]) || (probe_n > 31 && !close_enough(row[31])) ||
        (probe_n > 32 && !close_enough(row[32]))) {
        for (int d = 0; d < npd; d++) {
            if (pd[d].fn) {
                xpu_set_device(d);
                release(&pd[d]);
            }
        }
        return fail("gold mismatch");
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
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() / (double)loops;
    double ops = 2.0 * (double)m * (double)n * (double)k;
    double tops = (ops / (us * 1e-6)) / 1e12;
    std::printf("RESULT profile=%s api=cdnn_pipe m=%d n=%d k=%d grid=%d,%d npd=%d "
                "warmup=%d loops=%d latency_us=%.3f tops_si=%.4f\n",
                name, m, n, k, ncl, ncore, npd, warmup, loops, us, tops);
    std::fflush(stdout);

    for (int d = 0; d < npd; d++) {
        if (!pd[d].fn)
            continue;
        xpu_set_device(d);
        release(&pd[d]);
    }
    return 0;
}
