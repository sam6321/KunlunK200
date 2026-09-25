// First-bring-up smoke: H2D/D2H pattern + tiny xdnn::fc on each PD.
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <vector>
#include <xpu/runtime.h>
#include <xpu/xdnn.h>

namespace xdnn = baidu::xpu::api;

static int fail(const char *msg) {
    std::fprintf(stderr, "FAIL: %s\n", msg);
    return 1;
}

static int memcpy_pattern(int dev, size_t sz) {
    if (xpu_set_device(dev) != 0)
        return fail("xpu_set_device");
    void *d = nullptr;
    if (xpu_malloc(&d, sz) != 0)
        return fail("xpu_malloc");
    std::vector<unsigned char> h(sz), h2(sz, 0);
    for (size_t i = 0; i < sz; i++)
        h[i] = (unsigned char)(i * 131u + 7u + (unsigned)dev);
    if (xpu_memcpy(d, h.data(), sz, XPU_HOST_TO_DEVICE) != 0)
        return fail("H2D");
    xpu_wait();
    if (xpu_memcpy(h2.data(), d, sz, XPU_DEVICE_TO_HOST) != 0)
        return fail("D2H");
    xpu_wait();
    xpu_free(d);
    if (std::memcmp(h.data(), h2.data(), sz) != 0)
        return fail("pattern mismatch");
    std::printf("  memcpy %zu KB: PASS\n", sz / 1024);
    return 0;
}

static int fc_gold(int dev, int m, int n, int k) {
    if (xpu_set_device(dev) != 0)
        return fail("xpu_set_device");
    std::vector<float> ha(m * k), hb(k * n), hc(m * n, 0.f), gold(m * n, 0.f);
    for (int i = 0; i < m * k; i++)
        ha[i] = 0.01f * (float)((i + 1) * (dev + 1));
    for (int i = 0; i < k * n; i++)
        hb[i] = 0.02f * (float)((i + 3) * (dev + 1));
    for (int i = 0; i < m; i++)
        for (int j = 0; j < n; j++) {
            float s = 0.f;
            for (int t = 0; t < k; t++)
                s += ha[i * k + t] * hb[t * n + j];
            gold[i * n + j] = s;
        }

    float *A = nullptr, *B = nullptr, *C = nullptr;
    if (xpu_malloc((void **)&A, ha.size() * 4) ||
        xpu_malloc((void **)&B, hb.size() * 4) ||
        xpu_malloc((void **)&C, hc.size() * 4))
        return fail("xpu_malloc fc");
    xpu_memcpy(A, ha.data(), ha.size() * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(B, hb.data(), hb.size() * 4, XPU_HOST_TO_DEVICE);
    xpu_wait();

    auto ctx = xdnn::create_context();
    int ret = xdnn::fc<float, float, float, int16_t>(
        ctx, A, B, C, m, n, k, false, false, nullptr, nullptr, nullptr);
    xpu_wait();
    xdnn::destroy_context(ctx);
    if (ret != 0) {
        std::fprintf(stderr, "FAIL: xdnn::fc ret=%d\n", ret);
        xpu_free(A); xpu_free(B); xpu_free(C);
        return 1;
    }
    xpu_memcpy(hc.data(), C, hc.size() * 4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    xpu_free(A); xpu_free(B); xpu_free(C);

    float max_abs = 0.f, max_rel = 0.f;
    for (int i = 0; i < m * n; i++) {
        float a = std::fabs(hc[i] - gold[i]);
        float r = a / (std::fabs(gold[i]) + 1e-6f);
        if (a > max_abs) max_abs = a;
        if (r > max_rel) max_rel = r;
    }
    std::printf("  fc %dx%dx%d: max_abs=%.5g max_rel=%.5g C[0]=%.6f gold[0]=%.6f\n",
                m, n, k, max_abs, max_rel, hc[0], gold[0]);
    if (max_abs > 1e-3f && max_rel > 1e-3f)
        return fail("fc mismatch vs CPU gold");
    std::printf("  fc: PASS\n");
    return 0;
}

int main() {
    int dc = 0;
    if (xpu_device_count(&dc) != 0 || dc < 2)
        return fail("need two PDs");
    std::printf("devices=%d\n", dc);
    for (int dev = 0; dev < 2; dev++) {
        std::printf("=== device %d ===\n", dev);
        if (memcpy_pattern(dev, 4u << 20))
            return 1;
        if (fc_gold(dev, 32, 32, 32))
            return 1;
    }
    std::printf("ALL PASS\n");
    return 0;
}
