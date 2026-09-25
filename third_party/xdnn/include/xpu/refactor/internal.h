#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_INTERNAL_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_INTERNAL_H
#include "xpu/dll_export.h"
#include "xpu/refactor/context/newcontext.h"
#include <vector>
namespace baidu {
namespace xpu {
namespace api {

DLL_EXPORT int do_host2device(Context* ctx, const void* src, void* dst, int bytes);
DLL_EXPORT int do_device2host(Context* ctx, const void* src, void* dst, int bytes);

template <typename T> int from_fp32(Context* ctx, const float* x, T* y, int len, float* maxptr);
template <typename T> int to_fp32(Context* ctx, const T* x, float* y, int len, const float* maxptr);
template <typename T> int from_fp16(Context* ctx, const float16* x, T* y, int len, float* maxptr);
template <typename T> int to_fp16(Context* ctx, const T* x, float16* y, int len, const float* maxptr);
template <typename T> int from_nchw(Context* ctx, const T* x, T* y, int n, int c, int h, int w, bool is_nchw);
template <typename T> int to_nchw(Context* ctx, const T* x, T* y, int n, int c, int h, int w, bool is_nchw);
template <typename T> int from_nhwc(Context* ctx, const T* x, T* y, int n, int c, int h, int w, bool is_nchw);
template <typename T> int to_nhwc(Context* ctx, const T* x, T* y, int n, int c, int h, int w, bool is_nchw);

template <typename T> const float* to_fp32_if_necessary(api::ctx_guard& RAII_GUARD, Context* ctx,
        const T* x, int len, const float* maxptr);
template<typename T> const float* do_findmax_if_necessary(api::ctx_guard& RAII_GUARD, Context* ctx,
        const T* x, const float* maxptr, int len);
template <typename T> const T* to_nchw_if_necessary(api::ctx_guard& RAII_GUARD, Context* ctx,
        const T* x, int n, int c, int h, int w, bool is_nchw);

template<typename TGEMM> std::vector<void*> resnet_unit_fusion_get_reserve_space_ptrs(
        Context* ctx, void* reserve_space, const std::vector<const float*>& x_maxlist,
        const std::vector<const float*>& w_maxlist, const Activation_t& act, bool has_shortcut);

int im2col_param_check(Context* ctx, int n, int c, int h, int w,
        const std::vector<int>& ksize_extend, const std::vector<int>& stride_extend,
        const std::vector<int>& pad_extend, const std::vector<int>& dilation_extend,
        bool allow_pad_ksize_equal = false);

template<typename T> int image_pad2d_if_necessary(api::ctx_guard& RAII_GUARD, Context* ctx, const T** x,
        int n, int c, int& xh, int& xw, const std::vector<int>& ksize, const std::vector<int>& dilation,
        bool is_nchw, std::vector<int>& pad);

template<typename TW> TW* filter_change_group(api::ctx_guard& RAII_GUARD, Context* ctx,
        const TW* old_filter, int f, int c, const std::vector<int>& ksize, int oldg, int newg);

}
}
}
#endif
