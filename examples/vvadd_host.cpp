// Launch examples/simd_vvadd.bin on device 0 and check y = a + b.
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <string>
#include <vector>
#include <xpu/runtime.h>

static int fail(const char *msg, int ret = 0) {
    std::fprintf(stderr, "FAIL: %s", msg);
    if (ret)
        std::fprintf(stderr, " (%d)", ret);
    std::fprintf(stderr, "\n");
    return 1;
}

int main(int argc, char **argv) {
    const char *bin = argc > 1 ? argv[1] : "simd_vvadd.bin";
    std::string symbol = "simd_vvadd";
    if (argc > 2)
        symbol = argv[2];
    std::ifstream in(bin, std::ios::binary);
    std::vector<unsigned char> bytes((std::istreambuf_iterator<char>(in)),
                                     std::istreambuf_iterator<char>());
    if (bytes.empty())
        return fail("empty bin");

    if (xpu_set_device(0) != 0)
        return fail("xpu_set_device");
    const int n = 256;
    std::vector<float> ha(n), hb(n), hy(n), gold(n);
    for (int i = 0; i < n; i++) {
        ha[i] = (float)i;
        hb[i] = 1.0f;
        gold[i] = ha[i] + hb[i];
    }
    void *da = nullptr, *db = nullptr, *dy = nullptr, *code = nullptr;
    XPUFunc fn = nullptr;
    if (xpu_malloc(&da, n * 4) || xpu_malloc(&db, n * 4) || xpu_malloc(&dy, n * 4) ||
        xpu_malloc(&code, bytes.size()))
        return fail("malloc");
    xpu_memcpy(da, ha.data(), n * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(db, hb.data(), n * 4, XPU_HOST_TO_DEVICE);
    xpu_memcpy(code, bytes.data(), bytes.size(), XPU_HOST_TO_DEVICE);
    xpu_wait();
    int ret = xpu_create_cl_func(&fn, (uint64_t)(uintptr_t)code, (uint32_t)bytes.size(),
                                 0, 0, 0, symbol.c_str(), true);
    if (ret)
        return fail("xpu_create_cl_func", ret);
    size_t off = 0;
    xpu_launch_argument_set(&dy, 8, off);
    off += 8;
    xpu_launch_argument_set(&da, 8, off);
    off += 8;
    xpu_launch_argument_set(&db, 8, off);
    off += 8;
    xpu_launch_argument_set(&n, 4, off);
    if (xpu_launch_config(1, 1) || xpu_launch_async(fn) || xpu_wait())
        return fail("launch");
    xpu_memcpy(hy.data(), dy, n * 4, XPU_DEVICE_TO_HOST);
    xpu_wait();
    for (int i = 0; i < n; i++) {
        if (hy[i] != gold[i]) {
            std::fprintf(stderr, "FAIL: y[%d]=%g gold=%g\n", i, hy[i], gold[i]);
            return 1;
        }
    }
    std::printf("simd_vvadd n=%d PASS y[255]=%g\n", n, hy[255]);
    return 0;
}
