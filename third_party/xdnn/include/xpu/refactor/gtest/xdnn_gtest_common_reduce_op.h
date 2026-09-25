#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_COMMON_REDUCE_OP_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_COMMON_REDUCE_OP_H

#include "xpu/xdnn.h"
#include "xpu/refactor/gtest/xdnn_gtest.h"

template<typename T> static void gtest_common_reduce_op(std::string funcname,
        std::function<int(api::Context*, const T*, T*, const std::vector<int>&, const std::vector<int>&)> func,
        float diff, api::DeviceType dev, const std::string& x_pos, const std::string& y_pos,
        const std::vector<int>& xshape, const std::vector<int>& rdims, bool is_all_any_op = false) {
    api::Context ctx_cpu(api::kCPU);
    api::Context ctx_xpu(dev);
    api::Dtype dt = api::CPPTypeToDtype<T>();
    auto x0 = api::tensor(0);
    auto y0 = api::tensor(0);
    int lenx = vector_prod(xshape);
    int leny = lenx;
    GTEST_INIT(&ctx_xpu);
    for (int i = 0; i < rdims.size(); ++i) {
        leny /= xshape[rdims[i]];
    }
    if (is_all_any_op) {
        x0 = api::randint(0, 1, lenx).astype(dt);
        y0 = api::randint(0, 1, leny).astype(dt);
    } else if (funcname == "reduce_prod") {
        x0 = api::randfloat(0.2f, 1.5f, lenx).astype(dt);
        y0 = api::randfloat(0.2f, 1.5f, leny).astype(dt);
    } else {
        x0 = api::randfloat(-10.0f, 10.0f, lenx).astype(dt);
        y0 = api::randfloat(-10.0f, 10.0f, leny).astype(dt);
    }
    GTEST_DEFINE_PTR(T, x_pos, x0, x1, x0ptr, x1ptr);
    GTEST_DEFINE_PTR(T, y_pos, y0, y1, y0ptr, y1ptr);
    GTEST_XPU_START(&ctx_xpu);
    ASSERT_EQ(0, func(&ctx_xpu, x1ptr, y1ptr, xshape, rdims));
    GTEST_XPU_END_DMA_FMT(&ctx_xpu, (lenx + leny) * sizeof(T), "%s profiling", funcname.c_str());
    GTEST_CPU_START(&ctx_cpu);
    ASSERT_EQ(0, func(&ctx_cpu, x0ptr, y0ptr, xshape, rdims));
    TENSOR_ALLCLOSE(y0, y1, diff, diff);
    GTEST_CPU_END(&ctx_cpu);
}

const std::vector<std::vector<std::vector<int> > > plist_xpu1 = {
    {{1}, {0}},
    {{5, 1, 5}, {1}},
    {{1, 1, 1}, {0, 2}},
    {{50, 50}, {1}},
    {{2, 3, 4, 3}, {0, 2}}, //连续多次调用kernel，影响精度，稍后修改
    {{2, 3, 4, 3}, {2, 3}},
    {{2, 5, 3, 3, 4, 3}, {0, 1, 4}},
    {{2, 2, 3, 5, 4, 3}, {2, 3, 5}},
    {{2, 1, 3, 1, 4, 3}, {0, 1, 4}},
    {{2, 1, 3, 1, 4, 3}, {2, 3, 5}},
    {{1, 4096, 6}, {1}},   // 调用big_t + mtn, 且在big_t里没有更新全部元素
    {{1, 1024, 2}, {1}},   // 调用big_t
    // {{1, 2050, 3000},{1}}, // 调用big_t + mtn, 且在big_t里没有更新全部元素, t<n；单测耗时太长暂时注释掉
    {{1, 10, 20}, {1}},    // 调用big_t, t<n
    {{2, 20, 5}, {1}},     // 调用mtn, 连续下载多个n*t
    {{60, 3, 200}, {1}},   // 调用mtn
    {{5, 200, 1}, {1}},    // 调用mt
};

const std::vector<std::vector<std::vector<int> > > plist_xpu2 = plist_xpu1;

const std::vector<std::vector<std::vector<int> > > plist_xpu3 = {
    {{1}, {0}},
    {{5, 1, 5}, {1}},
    {{50, 50}, {1}},
    {{2, 3, 4, 3}, {2, 3}},
    {{1, 4096, 6}, {1}},   // 调用big_t + mtn, 且在big_t里没有更新全部元素
    {{1, 1024, 2}, {1}},   // 调用big_t
    {{1, 10, 20}, {1}},    // 调用big_t, t<n
    {{2, 20, 5}, {1}},     // 调用mtn, 连续下载多个n*t
    {{60, 3, 200}, {1}},   // 调用mtn
    {{5, 200, 1}, {1}},    // 调用m
};

#endif