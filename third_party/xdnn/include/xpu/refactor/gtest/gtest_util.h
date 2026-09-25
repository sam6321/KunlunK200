#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_UTIL_GTEST_UTIL_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_UTIL_GTEST_UTIL_H
#include <functional>
#include "xpu/refactor/util/vector_util.h"

#define TENSOR_ALLCLOSE(tensor1, tensor2, rtol, atol)   \
ASSERT_EQ(tensor1.dtype(), tensor2.dtype());    \
ASSERT_EQ(tensor1.numel(), tensor2.numel());    \
ASSERT_EQ(tensor1.numel(), baidu::xpu::api::count_allclose(tensor1, tensor2, rtol, atol));

#define TENSOR_MAXCLOSE(tensor1, tensor2, rtol, atol)   \
ASSERT_EQ(baidu::xpu::api::abs_max_close(tensor1, tensor2, rtol, atol), true);

#define TENSOR_PRINT_DIFF(tensor1, tensor2)                                                                 \
ASSERT_EQ(tensor1.dtype(), tensor2.dtype());                                                                \
ASSERT_EQ(tensor1.numel(), tensor2.numel());                                                                \
ASSERT_EQ(0, baidu::xpu::api::print_tensor_diff(tensor1, tensor2));

#define QUANT_TENSOR_MAXCLOSE(qtensor1, qtensor2, rtol, atol)   \
ASSERT_EQ(1, baidu::xpu::api::max_allclose(qtensor1, qtensor2, rtol, atol));


#endif
