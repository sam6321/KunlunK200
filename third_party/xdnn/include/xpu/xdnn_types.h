/// \file   types.h
/// \brief  XPU API types
/// \author shijiaxin01@baidu.com
/// \copyright (C) 2019 Baidu, Inc
#ifndef BAIDU_XPU_API_INCLUDE_XPU_TYPES_H
#define BAIDU_XPU_API_INCLUDE_XPU_TYPES_H

#include <cstdint>
#include <stdlib.h>
#include "xpu/refactor/util/float16.h"
#include "xpu/refactor/context/newcontext.h"
#include "xpu/dll_export.h"
#include <vector>

namespace baidu {
namespace xpu {
namespace api {
/// \brief  XPUAPI使用的错误类型
typedef enum {
    SUCCESS = 0,
    INVALID_PARAM = 1,
    RUNTIME_ERROR = 2,
    NO_ENOUGH_WORKSPACE = 3,
    NOT_IMPLEMENT = 4,
} Error_t;

extern int do_host2device(Context* ctx, const void* src, void* dst, int bytes);
template <typename T>
class DLL_EXPORT VectorParam {
public:
    const T* cpu;
    int len;
    T* xpu;
    VectorParam to_xpu(ctx_guard& RAII) const {
        VectorParam ret{cpu, len, xpu};
        if (ret.xpu == nullptr && RAII._ctx->dev().type() != api::kCPU) {
            ret.xpu = RAII.alloc<T>(len);
            if (ret.xpu != nullptr) { // alloc success
                int copy_success = do_host2device(RAII._ctx, cpu, ret.xpu, len * sizeof(T));
                if (copy_success != 0) {
                    ret.xpu = nullptr;
                }
            }
        }
        return ret;
    }
};
}  // namespace api
}  // namespace xpu
}  // namespace baidu

#endif //BAIDU_XPU_API_INCLUDE_XPU_TYPES_H
