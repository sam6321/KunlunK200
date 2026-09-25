#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_BROADCAST_CALC_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_BROADCAST_CALC_H

#include "xpu/xdnn.h"
#include "xpu/refactor/gtest/xdnn_gtest.h"

#define GTEST_BROADCAST_HELPER(type, ctx)                                         \
    gtest_broadcast<type>(ctx, GM, GM, {1}, {7890});                              \
    gtest_broadcast<type>(ctx, GM, GM, {3, 1, 3}, {3, 4, 3});                     \
    gtest_broadcast<type>(ctx, GM, GM, {7, 1, 9, 1, 11}, {7, 8, 9, 10, 11});      \
    gtest_broadcast<type>(ctx, GM, GM, {7, 1, 1, 1, 11}, {7, 8, 9, 10, 11});      \
    gtest_broadcast<type>(ctx, GM, GM, {7, 1, 1, 1, 11}, {7, 8, 1, 10, 11});      \
    gtest_broadcast<type>(ctx, GM, GM, {7, 1, 9, 1, 1}, {7, 8, 9, 10, 11});       \
    gtest_broadcast<type>(ctx, GM, GM, {2, 3, 4}, {2, 3, 4});                     \
    gtest_broadcast<type>(ctx, GM, GM, {81, 33, 1}, {81, 33, 33});                \
    gtest_broadcast<type>(ctx, GM, GM, {7, 6}, {21, 6});                          \
    gtest_broadcast<type>(ctx, GM, GM, {2, 3, 1}, {2, 600, 2});                   \
    gtest_broadcast<type>(ctx, GM, GM, {5, 32, 20}, {25, 128, 60});               \

#define GTEST_BROADCAST_CALC_HELPER(op, type, ctx)                         \
    gtest_##op<type>(ctx, GM, GM, GM, {6}, {1});                           \
    gtest_##op<type>(ctx, GM, GM, GM, {1}, {6});                           \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 5}, {5});                        \
    gtest_##op<type>(ctx, GM, GM, GM, {5}, {6, 5});                        \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 1}, {6, 5});                     \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 5}, {6, 1});                     \
    gtest_##op<type>(ctx, GM, GM, GM, {16, 1578}, {1578});                 \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 5, 4, 3}, {5, 1, 1});            \
    gtest_##op<type>(ctx, GM, GM, GM, {5, 1, 1}, {6, 5, 4, 3});            \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 1}, {1, 6});                     \
    gtest_##op<type>(ctx, GM, GM, GM, {2, 64, 3, 1}, {2, 64, 1, 1});       \
    gtest_##op<type>(ctx, GM, GM, GM, {2, 64, 1, 1}, {2, 64, 3, 1});       \
    gtest_##op<type>(ctx, GM, GM, GM, {1, 1, 16, 17}, {1, 1, 1});          \
    gtest_##op<type>(ctx, GM, GM, GM, {5, 16, 17}, {5, 1, 17});            \
    gtest_##op<type>(ctx, GM, GM, GM, {1, 1}, {1, 1});

#define GTEST_BROADCAST_CALC_GRAD_HELPER(op, type, ctx)                          \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {10, 3, 12}, {10, 1, 12});     \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {6}, {1});                     \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {1}, {6});                     \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {6, 5}, {5});                  \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {5}, {6, 5});                  \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {6, 1}, {6, 5});               \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {6, 5}, {6, 1});               \
    gtest_##op<type>(ctx, GM, GM, GM, GM, GM, GM, {6, 5, 4, 3}, {5, 1, 1});      \

#define GTEST_BROADCAST_COMPARE_HELPER(op, type, ctx)                      \
    gtest_##op<type>(ctx, GM, GM, GM, {6}, {1});                           \
    gtest_##op<type>(ctx, GM, GM, GM, {1}, {6});                           \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 5}, {5});                        \
    gtest_##op<type>(ctx, GM, GM, GM, {5}, {6, 5});                        \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 1}, {6, 5});                     \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 5}, {6, 1});                     \
    gtest_##op<type>(ctx, GM, GM, GM, {16, 1578}, {1578});                 \
    gtest_##op<type>(ctx, GM, GM, GM, {6, 5, 4, 3}, {5, 1, 1});            \

static std::vector<int> get_zshape(const std::vector<int>& xshape, const std::vector<int>& yshape) {
    std::vector<int> ret;
    int shape_len_max = std::max<int>(xshape.size(), yshape.size());
    for (int i = 0; i < shape_len_max; i++) {
        int xi = i + xshape.size() - shape_len_max;
        int yi = i + yshape.size() - shape_len_max;
        xi = ((xi < 0) ? 1 : xshape[xi]);
        yi = ((yi < 0) ? 1 : yshape[yi]);
        ret.push_back(std::max<int>(xi, yi));
    }
    return ret;
}

static int get_sign(float v) {
    if (v >= 0) {
        return 1;
    } else {
        return -1;
    }
}

template<typename T> static void gtest_broadcast(api::DeviceType dev, std::string xpos, std::string ypos,
        const std::vector<int>& xshape, const std::vector<int>& yshape) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    int xlen = vector_prod(xshape);
    int ylen = vector_prod(yshape);
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(-10.0f, 10.0f, xlen).astype(dt);
    auto y0 = api::randfloat(-10.0f, 10.0f, ylen).astype(dt);
    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);

    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, api::broadcast<T>(&ctx_xpu, x1ptr, y1ptr, xshape, yshape));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, (xlen + ylen) * sizeof(T), "broadcast, rlen(%d) wlen(%d)", xlen, ylen);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, api::broadcast<T>(&ctx_cpu, x0ptr, y0ptr, xshape, yshape));
    TENSOR_PRINT_DIFF(y0, y1);
    TENSOR_ALLCLOSE(y0, y1, 0, 0);
    GTEST_CPU_END(&ctx_cpu);
}

template<typename T> static void gtest_broadcast_calc(std::string funcname,
        std::function<int(api::Context*, const T*, const T*, T*,
                const std::vector<int>&, const std::vector<int>&)> func, float diff,
        api::DeviceType dev, std::string xpos, std::string ypos, std::string zpos,
        const std::vector<int>& xshape, const std::vector<int>& yshape,
        float x_minv, float x_maxv, float y_minv, float y_maxv, float eps = 0.0f) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    std::vector<int> zshape = get_zshape(xshape, yshape);
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(x_minv, x_maxv, xshape).astype(dt);
    auto y0 = api::randfloat(y_minv, y_maxv, yshape).astype(dt);
    auto z0 = api::randfloat(-10.0f, 10.0f, zshape).astype(dt);
    for (int j = 0; j < x0.numel(); ++j) {
        x0.data<T>()[j] += get_sign(x0.data<T>()[j]) * eps;
    }
    for (int j = 0; j < y0.numel(); ++j) {
        y0.data<T>()[j] += get_sign(y0.data<T>()[j]) * eps;
    }
    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);
    GTEST_DEFINE_PTR(T, zpos, z0, z1, z0ptr, z1ptr);

    int xlen = vector_prod(xshape);
    int ylen = vector_prod(yshape);
    int zlen = vector_prod(zshape);
    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func(&ctx_xpu, x1ptr, y1ptr, z1ptr, xshape, yshape));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, (xlen + ylen + zlen) * sizeof(T), "%s, rlen(%d) wlen(%d)", funcname.c_str(),
            xlen + ylen, zlen);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func(&ctx_cpu, x0ptr, y0ptr, z0ptr, xshape, yshape));
    TENSOR_PRINT_DIFF(z0, z1);
    TENSOR_ALLCLOSE(z0, z1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

template<typename T> static void gtest_broadcast_calc_grad(std::string funcname,
        std::function<int(api::Context*, const T*, const T*, const T*, const T*, T*, T*,
                const std::vector<int>&, const std::vector<int>&)> func, float diff,
        api::DeviceType dev, std::string xpos, std::string ypos, std::string zpos,
        std::string dxpos, std::string dypos, std::string dzpos,
        const std::vector<int>& xshape, const std::vector<int>& yshape,
        float minv, float maxv, float eps = 0.0f) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    std::vector<int> zshape = get_zshape(xshape, yshape);
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(minv, maxv, xshape).astype(dt);
    auto y0 = api::randfloat(minv, maxv, yshape).astype(dt);
    auto z0 = api::randfloat(minv, maxv, zshape).astype(dt);
    auto dx0 = api::randfloat(minv, maxv, xshape).astype(dt);
    auto dy0 = api::randfloat(minv, maxv, yshape).astype(dt);
    auto dz0 = api::randfloat(minv, maxv, zshape).astype(dt);
    for (int j = 0; j < x0.numel(); ++j) {
        x0.data<T>()[j] += get_sign(x0.data<T>()[j]) * eps;
    }
    for (int j = 0; j < y0.numel(); ++j) {
        y0.data<T>()[j] += get_sign(y0.data<T>()[j]) * eps;
    }
    for (int j = 0; j < z0.numel(); ++j) {
        z0.data<T>()[j] += get_sign(z0.data<T>()[j]) * eps;
    }
    for (int j = 0; j < dz0.numel(); ++j) {
        dz0.data<T>()[j] += get_sign(dz0.data<T>()[j]) * eps;
    }
    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);
    GTEST_DEFINE_PTR(T, zpos, z0, z1, z0ptr, z1ptr);
    GTEST_DEFINE_PTR(T, dxpos, dx0, dx1, dx0ptr, dx1ptr);
    GTEST_DEFINE_PTR(T, dypos, dy0, dy1, dy0ptr, dy1ptr);
    GTEST_DEFINE_PTR(T, dzpos, dz0, dz1, dz0ptr, dz1ptr);

    int xlen = vector_prod(xshape);
    int ylen = vector_prod(yshape);
    int zlen = vector_prod(zshape);
    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func(&ctx_xpu, x1ptr, y1ptr, z1ptr, dz1ptr, dy1ptr, dx1ptr, xshape, yshape));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, (2 * xlen + 2 * ylen + zlen) * sizeof(T),
            "%s, rlen(%d) wlen(%d)", funcname.c_str(), xlen + ylen + zlen, xlen + ylen);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func(&ctx_cpu, x0ptr, y0ptr, z0ptr, dz0ptr, dy0ptr, dx0ptr, xshape, yshape));
    TENSOR_PRINT_DIFF(dx0, dx1);
    TENSOR_PRINT_DIFF(dy0, dy1);
    TENSOR_ALLCLOSE(dx0, dx1, diff, diff);
    TENSOR_ALLCLOSE(dy0, dy1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

template<typename T> static void gtest_broadcast_compare(std::string funcname,
        std::function<int(api::Context*, const T*, const T*, bool*,
                const std::vector<int>&, const std::vector<int>&)> func,
        api::DeviceType dev, std::string xpos, std::string ypos, std::string zpos,
        const std::vector<int>& xshape, const std::vector<int>& yshape,
        float minv, float maxv) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    std::vector<int> zshape = get_zshape(xshape, yshape);
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(minv, maxv, xshape).astype(dt);
    auto y0 = api::randfloat(minv, maxv, yshape).astype(dt);
    auto z0 = api::randint(-10, 10, zshape).astype(api::kINT8);

    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);
    GTEST_DEFINE_PTR(bool, zpos, z0, z1, z0ptr, z1ptr);

    int xlen = vector_prod(xshape);
    int ylen = vector_prod(yshape);
    int zlen = vector_prod(zshape);
    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func(&ctx_xpu, x1ptr, y1ptr, z1ptr, xshape, yshape));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, (xlen + ylen) * sizeof(T) + zlen, "%s, rlen(%d) wlen(%d)", funcname.c_str(),
            xlen + ylen, zlen);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func(&ctx_cpu, x0ptr, y0ptr, z0ptr, xshape, yshape));
    TENSOR_PRINT_DIFF(z0, z1);
    TENSOR_ALLCLOSE(z0, z1, 0, 0);
    GTEST_CPU_END(&ctx_cpu);
}

#endif