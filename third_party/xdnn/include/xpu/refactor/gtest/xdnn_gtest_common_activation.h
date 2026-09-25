#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_COMMON_ACTIVATION_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_COMMON_ACTIVATION_H

#include "xpu/xdnn.h"
#include "xpu/refactor/gtest/xdnn_gtest.h"

template<typename T> static void gtest_common_param0_activation(std::string funcname,
        std::function<int(api::Context*, const T*, T*, int, const float*, float*)> func, float diff,
        api::DeviceType dev, std::string xpos, std::string ypos, int len) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto y0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);

    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);

    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func(&ctx_xpu, x1ptr, y1ptr, len, nullptr, nullptr));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, 2 * len * sizeof(T), "%s,len(%d)", funcname.c_str(), len);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func(&ctx_cpu, x0ptr, y0ptr, len, nullptr, nullptr));
    TENSOR_PRINT_DIFF(y0, y1);
    TENSOR_ALLCLOSE(y0, y1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

template<typename T> static void gtest_common_param1_activation(std::string funcname,
        std::function<int(api::Context*, const T*, T*, int, float, const float*, float*)> func, float diff,
        api::DeviceType dev, std::string xpos, std::string ypos, int len, float param) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto y0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);

    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);

    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func(&ctx_xpu, x1ptr, y1ptr, len, param, nullptr, nullptr));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, 2 * len * sizeof(T), "%s,len(%d)", funcname.c_str(), len, param);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func(&ctx_cpu, x0ptr, y0ptr, len, param, nullptr, nullptr));
    TENSOR_PRINT_DIFF(y0, y1);
    TENSOR_ALLCLOSE(y0, y1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

template<typename T> static void gtest_common_param0_activation_grad(std::string funcname,
        std::function<int(api::Context*, const T*, T*, int, const float*, float*)> func,
        std::function<int(api::Context*, const T*, const T*, const T*, T*, int)> func_grad,
        float diff, api::DeviceType dev, std::string xpos, std::string ypos, std::string dypos, std::string dxpos,
        int len, int l3size = 0) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    void* l3ptr = nullptr;
    if (l3size > 0) {
        xpu_malloc((void**)&l3ptr, l3size, XPU_MEM_L3);
        ASSERT_NE(l3ptr, nullptr);
        ctx_xpu._l3_mgr.set(l3ptr, l3size);
    }
    api::Dtype dt = api::CPPTypeToDtype<T>();
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto y0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto dy0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto dx0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    // init y0 using func
    ASSERT_EQ(0, func(&ctx_cpu, x0.data<T>(), y0.data<T>(), len, nullptr, nullptr));

    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);
    GTEST_DEFINE_PTR(T, dypos, dy0, dy1, dy0ptr, dy1ptr);
    GTEST_DEFINE_PTR(T, dxpos, dx0, dx1, dx0ptr, dx1ptr);

    int dma_cnt = 4;
    if (xpos == "NULL") {
        dma_cnt--;
    }
    if (ypos == "NULL") {
        dma_cnt--;
    }
    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func_grad(&ctx_xpu, x1ptr, y1ptr, dy1ptr, dx1ptr, len));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, dma_cnt * len * sizeof(T), "%s,len(%d)", funcname.c_str(), len);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func_grad(&ctx_cpu, x0ptr, y0ptr, dy0ptr, dx0ptr, len));
    TENSOR_PRINT_DIFF(dx0, dx1);
    TENSOR_ALLCLOSE(dx0, dx1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

template<typename T> static void gtest_common_param1_activation_grad(std::string funcname,
        std::function<int(api::Context*, const T*, T*, int, float, const float*, float*)> func,
        std::function<int(api::Context*, const T*, const T*, const T*, T*, int, float)> func_grad,
        float diff, api::DeviceType dev, std::string xpos, std::string ypos, std::string dypos, std::string dxpos,
        int len, float param, int l3size = 0) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    void* l3ptr = nullptr;
    if (l3size > 0) {
        xpu_malloc((void**)&l3ptr, l3size, XPU_MEM_L3);
        ASSERT_NE(l3ptr, nullptr);
        ctx_xpu._l3_mgr.set(l3ptr, l3size);
    }
    api::Dtype dt = api::CPPTypeToDtype<T>();
    GTEST_INIT(&ctx_xpu);
    auto x0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto y0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto dy0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    auto dx0 = api::randfloat(-16.0f, 16.0f, len).astype(dt);
    // init y0 using func
    ASSERT_EQ(0, func(&ctx_cpu, x0.data<T>(), y0.data<T>(), len, param, nullptr, nullptr));

    GTEST_DEFINE_PTR(T, xpos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, ypos, y0, y1, y0ptr, y1ptr);
    GTEST_DEFINE_PTR(T, dypos, dy0, dy1, dy0ptr, dy1ptr);
    GTEST_DEFINE_PTR(T, dxpos, dx0, dx1, dx0ptr, dx1ptr);

    int dma_cnt = 4;
    if (xpos == "NULL") {
        dma_cnt--;
    }
    if (ypos == "NULL") {
        dma_cnt--;
    }
    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func_grad(&ctx_xpu, x1ptr, y1ptr, dy1ptr, dx1ptr, len, param));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, dma_cnt * len * sizeof(T), "%s,len(%d)", funcname.c_str(), len);

    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func_grad(&ctx_cpu, x0ptr, y0ptr, dy0ptr, dx0ptr, len, param));
    TENSOR_PRINT_DIFF(dx0, dx1);
    TENSOR_ALLCLOSE(dx0, dx1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

#endif
