// Dual-PD GEMM throughput for KL1 / K200.
// Vintage: fc_int8 / fc_int16 (KL1-era host APIs still in libxpuapi).
// Modern:  public xdnn::fc<T,...> template instantiations.
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>
#include <xpu/refactor/context/xpu_act_type.h>

namespace xdnn = baidu::xpu::api;

namespace baidu {
namespace xpu {
namespace api {
int fc_int8(Context *ctx, bool TransA, bool TransB, int M, int N, int K,
            const int8_t *A, float max_a, const int8_t *B, float max_b,
            int8_t *C, float max_c);
// Packed INT16. libxpuapi mangles a trailing Activation_t; the 12-arg form
// is not in this .so.
int fc_int16(Context *ctx, bool TransA, bool TransB, int M, int N, int K,
             const int16_t *A, float max_a, const int16_t *B, float max_b,
             int16_t *C, float max_c, Activation_t act);
// Float I/O, INT16 MAC (slide-era "INT/FP32" vintage path).
int fc_int16(Context *ctx, bool TransA, bool TransB, int M, int N, int K,
             float alpha, const float *A, const float *B, float beta, float *C,
             const float *bias, Activation_t act);
// Float16 activations, INT16 weights, float16 output (vintage FP16 I/O).
int fc_int16(Context *ctx, bool TransA, bool TransB, int M, int N, int K,
             float alpha, const float16 *A, const int16_t *B, float max_b,
             float beta, float16 *C, const float *bias, Activation_t act);
}
}
}

static int fail(const char *msg, int ret = 0) {
    if (ret)
        std::fprintf(stderr, "FAIL: %s (ret=%d)\n", msg, ret);
    else
        std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static const char *xdnn_err(int e) {
    switch (e) {
    case 0: return "OK";
    case 1: return "INVALID_PARAM";
    case 2: return "RUNTIME_ERROR";
    case 3: return "NO_ENOUGH_WORKSPACE";
    case 4: return "NOT_IMPLEMENT";
    default: return "OTHER";
    }
}

static bool ends_with(const std::string &s, const char *suf) {
    size_t n = std::strlen(suf);
    return s.size() >= n && s.compare(s.size() - n, n, suf) == 0;
}

static void *alloc_buf(size_t n, int *l3_left) {
    void *p = nullptr;
    if (*l3_left > 0 && (int)n < *l3_left) {
        if (xpu_malloc(&p, n, XPU_MEM_L3) == 0) {
            *l3_left -= (int)n;
            return p;
        }
    }
    if (xpu_malloc(&p, n, XPU_MEM_HBM) != 0)
        return nullptr;
    return p;
}

struct Buf {
    void *a = nullptr, *b = nullptr, *c = nullptr;
    void *xmax = nullptr, *wmax = nullptr, *ymax = nullptr;
    xdnn::Context *ctx = nullptr;
    int m = 0;
};

static void free_pd(Buf &b) {
    if (b.ctx)
        xdnn::destroy_context(b.ctx);
    if (b.a)
        xpu_free(b.a);
    if (b.b)
        xpu_free(b.b);
    if (b.c)
        xpu_free(b.c);
    if (b.xmax)
        xpu_free(b.xmax);
    if (b.wmax)
        xpu_free(b.wmax);
    if (b.ymax)
        xpu_free(b.ymax);
    b = Buf{};
}

static int alloc_maxptrs(Buf &b, int *l3, float absv) {
    b.xmax = alloc_buf(sizeof(float), l3);
    b.wmax = alloc_buf(sizeof(float), l3);
    b.ymax = alloc_buf(sizeof(float), l3);
    if (!b.xmax || !b.wmax || !b.ymax)
        return fail("xpu_malloc maxptr");
    if (xpu_memcpy(b.xmax, &absv, sizeof(float), XPU_HOST_TO_DEVICE) ||
        xpu_memcpy(b.wmax, &absv, sizeof(float), XPU_HOST_TO_DEVICE) ||
        xpu_memcpy(b.ymax, &absv, sizeof(float), XPU_HOST_TO_DEVICE))
        return fail("H2D maxptr");
    return 0;
}

static double baidu_gops(int m, int n, int k) {
    return (m / 1024.0) * (n / 1024.0) * (k / 1024.0) * 2.0;
}

static void print_json(const char *profile, const char *api, int m, int n,
                       int k, int loops, int warmup, int npd, double us,
                       double claimed, const char *note) {
    double ops = 2.0 * (double)m * (double)n * (double)k;
    double sec = us * 1e-6;
    double tops_si = (ops / sec) / 1e12;
    double gops_baidu = baidu_gops(m, n, k) / sec;
    double frac = claimed > 0 ? tops_si / claimed : 0;
    if (!note)
        note = "";
    if (!api)
        api = "";
    std::printf("RESULT profile=%s api=%s m=%d n=%d k=%d npd=%d warmup=%d loops=%d "
                "latency_us=%.3f tops_si=%.3f gops_baidu=%.3f claimed_tops=%.3f "
                "frac_claimed=%.3f ops_per_gemm=%.0f note=%s\n",
                profile, api, m, n, k, npd, warmup, loops, us, tops_si,
                gops_baidu, claimed, frac, ops, note[0] ? note : "-");
    std::printf("JSON {\"profile\":\"%s\",\"api\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
                "\"npd\":%d,\"warmup\":%d,\"loops\":%d,\"latency_us\":%.3f,"
                "\"tops_si\":%.4f,\"gops_baidu\":%.4f,\"claimed_tops\":%.4f,"
                "\"frac_claimed\":%.4f,\"ops_per_gemm\":%.0f,\"note\":\"%s\"}\n",
                profile, api, m, n, k, npd, warmup, loops, us, tops_si,
                gops_baidu, claimed, frac, ops, note);
    std::fflush(stdout);
}

static int parse_int(const char *s, int *out) {
    char *end = nullptr;
    long v = std::strtol(s, &end, 10);
    if (!end || *end || v < 0)
        return 1;
    *out = (int)v;
    return 0;
}

enum FillMode { FILL_FLOAT = 0, FILL_I8 = 1, FILL_I16 = 2 };

static int run_int8(const char *profile, const char *api, bool modern, int m,
                    int n, int k, int warmup, int loops, double claimed) {
    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    std::vector<int8_t> hA((size_t)m * k), hB((size_t)n * k);
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = (int8_t)((i * 131u + 17u) % 127);
    for (size_t i = 0; i < hB.size(); i++)
        hB[i] = (int8_t)((i * 89u + 3u) % 127);

    Buf pd[2];
    int ms[2] = {m0, m1};
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        int l3 = 16 * 1024 * 1024;
        pd[d].m = ms[d];
        pd[d].c = alloc_buf((size_t)ms[d] * n, &l3);
        pd[d].a = alloc_buf((size_t)ms[d] * k, &l3);
        pd[d].b = alloc_buf((size_t)n * k, &l3);
        if (!pd[d].a || !pd[d].b || !pd[d].c)
            return fail("xpu_malloc");
        const int8_t *srcA = hA.data() + (d == 0 ? 0 : (size_t)m0 * k);
        if (xpu_memcpy(pd[d].a, srcA, (size_t)ms[d] * k, XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), (size_t)n * k, XPU_HOST_TO_DEVICE))
            return fail("H2D");
        if (modern && alloc_maxptrs(pd[d], &l3, 127.f))
            return 1;
        xpu_wait();
        pd[d].ctx = xdnn::create_context();
        if (!pd[d].ctx)
            return fail("create_context");
        pd[d].ctx->set_ncluster(4);
        pd[d].ctx->set_nsdnn(4);
    }

    auto gemm_once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].ctx)
                continue;
            if (xpu_set_device(d))
                return fail("xpu_set_device loop");
            int ret = 0;
            if (!modern) {
                ret = xdnn::fc_int8(pd[d].ctx, false, true, pd[d].m, n, k,
                                    (int8_t *)pd[d].a, 127.f, (int8_t *)pd[d].b,
                                    127.f, (int8_t *)pd[d].c, 127.f);
            } else {
                ret = xdnn::fc<int8_t, int8_t, int8_t, int8_t>(
                    pd[d].ctx, (int8_t *)pd[d].a, (int8_t *)pd[d].b,
                    (int8_t *)pd[d].c, pd[d].m, n, k, false, true,
                    (const float *)pd[d].xmax, (const float *)pd[d].wmax,
                    (float *)pd[d].ymax);
            }
            if (ret)
                return fail(xdnn_err(ret), ret);
        }
        return 0;
    };

    for (int i = 0; i < warmup; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                (double)loops;

    print_json(profile, api, m, n, k, loops, warmup, npd, us, claimed,
               modern ? "modern_xdnn_fc_int8_transb_devmax"
                      : "vintage_fc_int8_transb");
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        free_pd(pd[d]);
    }
    return 0;
}

static int run_int16(const char *profile, const char *api, bool modern, int m,
                     int n, int k, int warmup, int loops, double claimed) {
    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    std::vector<int16_t> hA((size_t)m * k), hB((size_t)n * k);
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = (int16_t)((i * 131u + 17u) % 32767);
    for (size_t i = 0; i < hB.size(); i++)
        hB[i] = (int16_t)((i * 89u + 3u) % 32767);

    Buf pd[2];
    int ms[2] = {m0, m1};
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        int l3 = 16 * 1024 * 1024;
        pd[d].m = ms[d];
        pd[d].c = alloc_buf((size_t)ms[d] * n * sizeof(int16_t), &l3);
        pd[d].a = alloc_buf((size_t)ms[d] * k * sizeof(int16_t), &l3);
        pd[d].b = alloc_buf((size_t)n * k * sizeof(int16_t), &l3);
        if (!pd[d].a || !pd[d].b || !pd[d].c)
            return fail("xpu_malloc");
        const int16_t *srcA = hA.data() + (d == 0 ? 0 : (size_t)m0 * k);
        if (xpu_memcpy(pd[d].a, srcA, (size_t)ms[d] * k * sizeof(int16_t),
                       XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), (size_t)n * k * sizeof(int16_t),
                       XPU_HOST_TO_DEVICE))
            return fail("H2D");
        if (modern && alloc_maxptrs(pd[d], &l3, 32767.f))
            return 1;
        xpu_wait();
        pd[d].ctx = xdnn::create_context();
        if (!pd[d].ctx)
            return fail("create_context");
        pd[d].ctx->set_ncluster(4);
        pd[d].ctx->set_nsdnn(4);
    }

    auto gemm_once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].ctx)
                continue;
            if (xpu_set_device(d))
                return fail("xpu_set_device loop");
            int ret = 0;
            if (!modern) {
                ret = xdnn::fc_int16(
                    pd[d].ctx, false, true, pd[d].m, n, k, (int16_t *)pd[d].a,
                    32767.f, (int16_t *)pd[d].b, 32767.f, (int16_t *)pd[d].c,
                    32767.f, xdnn::Activation_t(xdnn::Activation_t::LINEAR));
            } else {
                ret = xdnn::fc<int16_t, int16_t, int16_t, int16_t>(
                    pd[d].ctx, (int16_t *)pd[d].a, (int16_t *)pd[d].b,
                    (int16_t *)pd[d].c, pd[d].m, n, k, false, true,
                    (const float *)pd[d].xmax, (const float *)pd[d].wmax,
                    (float *)pd[d].ymax);
            }
            if (ret)
                return fail(xdnn_err(ret), ret);
        }
        return 0;
    };

    for (int i = 0; i < warmup; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                (double)loops;

    print_json(profile, api, m, n, k, loops, warmup, npd, us, claimed,
               modern ? "modern_xdnn_fc_int16_transb_devmax"
                      : "vintage_fc_int16_transb");
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        free_pd(pd[d]);
    }
    return 0;
}

template <typename T, typename TG>
static int run_fc(const char *profile, const char *api, int m, int n, int k,
                  int warmup, int loops, double claimed, bool trans_b,
                  float max_abs, int fill_mode, const char *note) {
    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    std::vector<T> hA((size_t)m * k), hB(trans_b ? (size_t)n * k : (size_t)k * n);
    if (fill_mode == FILL_I8) {
        for (size_t i = 0; i < hA.size(); i++)
            hA[i] = T((int8_t)((i * 131u + 17u) % 127));
        for (size_t i = 0; i < hB.size(); i++)
            hB[i] = T((int8_t)((i * 89u + 3u) % 127));
    } else if (fill_mode == FILL_I16) {
        for (size_t i = 0; i < hA.size(); i++)
            hA[i] = T((int16_t)((i * 131u + 17u) % 32767));
        for (size_t i = 0; i < hB.size(); i++)
            hB[i] = T((int16_t)((i * 89u + 3u) % 32767));
    } else {
        for (size_t i = 0; i < hA.size(); i++)
            hA[i] = T(0.01f * (float)((i % 50) + 1));
        for (size_t i = 0; i < hB.size(); i++)
            hB[i] = T(0.02f * (float)((i % 40) + 1));
    }

    Buf pd[2];
    int ms[2] = {m0, m1};
    size_t bbytes = hB.size() * sizeof(T);
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        int l3 = 16 * 1024 * 1024;
        pd[d].m = ms[d];
        pd[d].c = alloc_buf((size_t)ms[d] * n * sizeof(T), &l3);
        pd[d].a = alloc_buf((size_t)ms[d] * k * sizeof(T), &l3);
        pd[d].b = alloc_buf(bbytes, &l3);
        if (!pd[d].a || !pd[d].b || !pd[d].c)
            return fail("xpu_malloc");
        const T *srcA = hA.data() + (d == 0 ? 0 : (size_t)m0 * k);
        if (xpu_memcpy(pd[d].a, srcA, (size_t)ms[d] * k * sizeof(T),
                       XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), bbytes, XPU_HOST_TO_DEVICE))
            return fail("H2D");
        if (max_abs > 0.f && alloc_maxptrs(pd[d], &l3, max_abs))
            return 1;
        xpu_wait();
        pd[d].ctx = xdnn::create_context();
        if (!pd[d].ctx)
            return fail("create_context");
        pd[d].ctx->set_ncluster(4);
        pd[d].ctx->set_nsdnn(4);
    }

    auto gemm_once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].ctx)
                continue;
            if (xpu_set_device(d))
                return fail("xpu_set_device loop");
            const float *px = max_abs > 0.f ? (const float *)pd[d].xmax : nullptr;
            const float *pw = max_abs > 0.f ? (const float *)pd[d].wmax : nullptr;
            float *py = max_abs > 0.f ? (float *)pd[d].ymax : nullptr;
            int ret = xdnn::fc<T, T, T, TG>(pd[d].ctx, (T *)pd[d].a, (T *)pd[d].b,
                                            (T *)pd[d].c, pd[d].m, n, k, false,
                                            trans_b, px, pw, py);
            if (ret)
                return fail(xdnn_err(ret), ret);
        }
        return 0;
    };

    for (int i = 0; i < warmup; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                (double)loops;
    print_json(profile, api, m, n, k, loops, warmup, npd, us, claimed, note);
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        free_pd(pd[d]);
    }
    return 0;
}

static int run_fp32_vintage(const char *profile, const char *api, int m, int n,
                            int k, int warmup, int loops, double claimed) {
    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    std::vector<float> hA((size_t)m * k), hB((size_t)n * k);
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = 0.01f * (float)((i % 50) + 1);
    for (size_t i = 0; i < hB.size(); i++)
        hB[i] = 0.02f * (float)((i % 40) + 1);

    Buf pd[2];
    int ms[2] = {m0, m1};
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        int l3 = 16 * 1024 * 1024;
        pd[d].m = ms[d];
        pd[d].c = alloc_buf((size_t)ms[d] * n * sizeof(float), &l3);
        pd[d].a = alloc_buf((size_t)ms[d] * k * sizeof(float), &l3);
        pd[d].b = alloc_buf((size_t)n * k * sizeof(float), &l3);
        if (!pd[d].a || !pd[d].b || !pd[d].c)
            return fail("xpu_malloc");
        const float *srcA = hA.data() + (d == 0 ? 0 : (size_t)m0 * k);
        if (xpu_memcpy(pd[d].a, srcA, (size_t)ms[d] * k * sizeof(float),
                       XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), (size_t)n * k * sizeof(float),
                       XPU_HOST_TO_DEVICE))
            return fail("H2D");
        xpu_wait();
        pd[d].ctx = xdnn::create_context();
        if (!pd[d].ctx)
            return fail("create_context");
        pd[d].ctx->set_ncluster(4);
        pd[d].ctx->set_nsdnn(4);
    }

    auto gemm_once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].ctx)
                continue;
            if (xpu_set_device(d))
                return fail("xpu_set_device loop");
            int ret = xdnn::fc_int16(
                pd[d].ctx, false, true, pd[d].m, n, k, 1.f, (const float *)pd[d].a,
                (const float *)pd[d].b, 0.f, (float *)pd[d].c, nullptr,
                xdnn::Activation_t(xdnn::Activation_t::LINEAR));
            if (ret)
                return fail(xdnn_err(ret), ret);
        }
        return 0;
    };

    for (int i = 0; i < warmup; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                (double)loops;
    print_json(profile, api, m, n, k, loops, warmup, npd, us, claimed,
               "vintage_fc_int16_float_io_transb");
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        free_pd(pd[d]);
    }
    return 0;
}

static int run_fp16_vintage(const char *profile, const char *api, int m, int n,
                            int k, int warmup, int loops, double claimed) {
    int dc = 0;
    if (xpu_device_count(&dc) || dc < 1)
        return fail("xpu_device_count");
    int npd = dc >= 2 ? 2 : 1;
    int m0 = (npd == 2) ? m / 2 : m;
    int m1 = m - m0;
    std::vector<float16> hA((size_t)m * k);
    std::vector<int16_t> hB((size_t)n * k);
    for (size_t i = 0; i < hA.size(); i++)
        hA[i] = float16(0.01f * (float)((i % 50) + 1));
    for (size_t i = 0; i < hB.size(); i++)
        hB[i] = (int16_t)((i * 89u + 3u) % 32767);

    Buf pd[2];
    int ms[2] = {m0, m1};
    for (int d = 0; d < npd; d++) {
        if (ms[d] <= 0)
            continue;
        if (xpu_set_device(d))
            return fail("xpu_set_device");
        int l3 = 16 * 1024 * 1024;
        pd[d].m = ms[d];
        pd[d].c = alloc_buf((size_t)ms[d] * n * sizeof(float16), &l3);
        pd[d].a = alloc_buf((size_t)ms[d] * k * sizeof(float16), &l3);
        pd[d].b = alloc_buf((size_t)n * k * sizeof(int16_t), &l3);
        if (!pd[d].a || !pd[d].b || !pd[d].c)
            return fail("xpu_malloc");
        const float16 *srcA = hA.data() + (d == 0 ? 0 : (size_t)m0 * k);
        if (xpu_memcpy(pd[d].a, srcA, (size_t)ms[d] * k * sizeof(float16),
                       XPU_HOST_TO_DEVICE) ||
            xpu_memcpy(pd[d].b, hB.data(), (size_t)n * k * sizeof(int16_t),
                       XPU_HOST_TO_DEVICE))
            return fail("H2D");
        xpu_wait();
        pd[d].ctx = xdnn::create_context();
        if (!pd[d].ctx)
            return fail("create_context");
        pd[d].ctx->set_ncluster(4);
        pd[d].ctx->set_nsdnn(4);
    }

    auto gemm_once = [&]() -> int {
        for (int d = 0; d < npd; d++) {
            if (!pd[d].ctx)
                continue;
            if (xpu_set_device(d))
                return fail("xpu_set_device loop");
            int ret = xdnn::fc_int16(
                pd[d].ctx, false, true, pd[d].m, n, k, 1.f,
                (const float16 *)pd[d].a, (const int16_t *)pd[d].b, 32767.f, 0.f,
                (float16 *)pd[d].c, nullptr,
                xdnn::Activation_t(xdnn::Activation_t::LINEAR));
            if (ret)
                return fail(xdnn_err(ret), ret);
        }
        return 0;
    };

    for (int i = 0; i < warmup; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }

    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < loops; i++) {
        if (gemm_once())
            return 1;
    }
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        xpu_wait();
    }
    auto t1 = std::chrono::steady_clock::now();
    double us = std::chrono::duration<double, std::micro>(t1 - t0).count() /
                (double)loops;
    print_json(profile, api, m, n, k, loops, warmup, npd, us, claimed,
               "vintage_fc_int16_f16A_i16B_f16C_transb");
    for (int d = 0; d < npd; d++) {
        xpu_set_device(d);
        free_pd(pd[d]);
    }
    return 0;
}

static void usage(const char *argv0) {
    std::fprintf(
        stderr,
        "usage: %s PROFILE [M N K] [--loops N] [--warmup N] [--claimed TOPS] "
        "[--transb 0|1] [--api vintage|modern]\n"
        "PROFILE: int8|int16|fp16|fp32|int32, optional -vintage / -modern suffix\n"
        "pairs (same MNK, dual-PD, TransB):\n"
        "  int8  vintage fc_int8              vs modern fc<int8> + device maxptrs\n"
        "  int16 vintage fc_int16             vs modern fc<int16> + device maxptrs\n"
        "  fp16  vintage fc_int16(f16,i16,f16) vs modern fc<float16>\n"
        "  fp32  vintage fc_int16(float I/O)  vs modern fc<float>\n"
        "defaults: int8  65408 512 4096   int16/fp16 32704 512 4096\n"
        "          fp32/int32 16384 512 4096\n",
        argv0);
}

int main(int argc, char **argv) {
    if (argc < 2) {
        usage(argv[0]);
        return 2;
    }
    std::string profile = argv[1];
    std::string dtype = profile;
    std::string api;
    if (ends_with(dtype, "-vintage")) {
        api = "vintage";
        dtype.resize(dtype.size() - 8);
    } else if (ends_with(dtype, "-modern")) {
        api = "modern";
        dtype.resize(dtype.size() - 7);
    }

    int m = 0, n = 512, k = 4096;
    int loops = 200, warmup = 10;
    double claimed = 0;
    int transb = 1;
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
        } else if (a == "--transb") {
            if (need(&transb))
                return 1;
        } else if (a == "--api") {
            if (i + 1 >= argc)
                return fail("missing --api");
            api = argv[++i];
            if (api != "vintage" && api != "modern")
                return fail("--api must be vintage or modern");
        } else if (a == "--claimed") {
            if (i + 1 >= argc)
                return fail("missing --claimed");
            claimed = std::atof(argv[++i]);
        } else {
            return fail(a.c_str());
        }
    }
    if (api.empty()) {
        if (dtype == "int8" || dtype == "int16")
            api = "vintage";
        else
            api = "modern";
    }
    if (m == 0) {
        if (dtype == "int8")
            m = 65408;
        else if (dtype == "int16" || dtype == "fp16")
            m = 32704;
        else
            m = 16384;
    }
    if (warmup < 1)
        warmup = 1;
    if (loops < 1)
        loops = 1;

    const char *prof = profile.c_str();
    const char *apics = api.c_str();
    std::printf("K200 GEMM bench profile=%s dtype=%s api=%s m=%d n=%d k=%d "
                "warmup=%d loops=%d transb=%d claimed=%.3f\n",
                prof, dtype.c_str(), apics, m, n, k, warmup, loops, transb,
                claimed);

    if (dtype == "int8")
        return run_int8(prof, apics, api == "modern", m, n, k, warmup, loops,
                        claimed);
    if (dtype == "int16")
        return run_int16(prof, apics, api == "modern", m, n, k, warmup, loops,
                         claimed);
    if (dtype == "fp16") {
        if (api == "vintage")
            return run_fp16_vintage(prof, apics, m, n, k, warmup, loops, claimed);
        return run_fc<float16, short>(prof, apics, m, n, k, warmup, loops,
                                      claimed, transb != 0, 0.f, FILL_FLOAT,
                                      "modern_xdnn_fc_float16");
    }
    if (dtype == "fp32") {
        if (api == "vintage")
            return run_fp32_vintage(prof, apics, m, n, k, warmup, loops, claimed);
        return run_fc<float, int16_t>(prof, apics, m, n, k, warmup, loops,
                                      claimed, transb != 0, 0.f, FILL_FLOAT,
                                      "modern_xdnn_fc_float");
    }
    if (dtype == "int32") {
        if (api == "vintage")
            return fail("no vintage packed INT32 GEMM (no fc_int32 / mm_int32)");
        std::printf("note=no packed INT32 GEMM in XDNN/CDNN (no fc_int32, no "
                    "mm_int32). Slide INT/FP32 is the 32-bit MAC; this run is "
                    "xdnn::fc<float,float,float,int> at the FP32 peak size.\n");
        return run_fc<float, int>(prof, apics, m, n, k, warmup, loops, claimed,
                                  transb != 0, 0.f, FILL_FLOAT,
                                  "tgemm_int_public_fc");
    }
    usage(argv[0]);
    return 2;
}
