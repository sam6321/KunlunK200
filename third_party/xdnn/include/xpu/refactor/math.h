#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_MATH_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_MATH_H

#include "xpu/refactor/context/newcontext.h"
#include "xpu/xdnn_types.h"
#include "xpu/refactor/deprecated.h"
#ifdef _MSC_VER
#ifdef max
#undef max
#endif
#ifdef min
#undef min
#endif
#endif
namespace baidu {
namespace xpu {
namespace api {

// quant
template<typename T> DLL_EXPORT int findmax(Context* ctx, const T* x, float* maxptr, int len);
template<typename TX, typename TY> DLL_EXPORT int quantization(Context* ctx,
        const TX* x, TY* y, int len, const float* maxptr);
template<typename TX, typename TY> DLL_EXPORT int dequantization(Context* ctx,
        const TX* x, TY* y, int len, const float* maxptr);

// basic op
template<typename T> DLL_EXPORT int constant(Context* ctx, T* x, int len, T val);
template<typename T> DLL_EXPORT int copy(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int copy2d(Context* ctx, const T* x, T* y, int m, int ldx, int ldy, int n);
template<typename TX, typename TY> DLL_EXPORT int cast(Context* ctx, const TX* x, TY* y, int len);
template<typename T> DLL_EXPORT int range(Context* ctx, T* y, T begin, T step, int len);
template<typename T> DLL_EXPORT int any(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int stack(Context* ctx, const std::vector<const T*>& x_ptrs, T* y, int height,
        int width);
// unary op
template<typename T> DLL_EXPORT int scale(Context* ctx, const T* x, T* y, int len,
        bool bias_after_scale, float _scale, float _bias);
template<typename T> DLL_EXPORT int abs(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int log(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int log1p(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int sqrt(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int exp(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int neg(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int square(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int rsqrt(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int sign(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int erf(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int reciprocal(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int clip(Context* ctx, const T* x, T* y, int len, T min_val, T max_val);
template<typename T> DLL_EXPORT int sqrt_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int rsqrt_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int square_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int reciprocal_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int abs_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int cumsum(Context* ctx, const T* x, T* y, const std::vector<int>& xshape,
        bool reverse, bool exclusive, int axis);
// is infinity or is nans
template<typename T> DLL_EXPORT int isfinite(Context* ctx, const T* x, int8_t* y, int len);
template<typename T> DLL_EXPORT int isnan(Context* ctx, const T* x, int8_t* y, int len);
template<typename T> DLL_EXPORT int isfinite(Context* ctx, const T* x, bool* y, int len);
template<typename T> DLL_EXPORT int isnan(Context* ctx, const T* x, bool* y, int len);
// count infinity or nans
template<typename T> DLL_EXPORT int count_nan_or_inf(Context* ctx, const T* x, int* y, int len);
template<typename T> DLL_EXPORT int check_nan_or_inf(Context* ctx, const T* x, bool* y, int len);
// binary op
template<typename T> DLL_EXPORT int add(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int sub(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int mul(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int div(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int max(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int min(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int pow(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int floordiv(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int add_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dx, T* dy, int len);
template<typename T> DLL_EXPORT int sub_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dx, T* dy, int len);
template<typename T> DLL_EXPORT int mul_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dx, T* dy, int len);
template<typename T> DLL_EXPORT int div_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dx, T* dy, int len);
template<typename T> DLL_EXPORT int max_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dx, T* dy, int len);
template<typename T> DLL_EXPORT int min_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dx, T* dy, int len);
template<typename T> DLL_EXPORT int ceil(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int floor(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int fmod(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int remainder(Context* ctx, const T* x, const T* y, T* z, int len);
// trigonometric op
template<typename T> DLL_EXPORT int sin(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int cos(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int tan(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int arcsin(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int arccos(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int arctan(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int arctan2(Context* ctx, const T* x, const T* x2, T* y, int len);
// hyperbolic op
template<typename T> DLL_EXPORT int sinh(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int cosh(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int coth(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int sech(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int csch(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int asinh(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int acosh(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int atanh(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int acoth(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int asech(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int acsch(Context* ctx, const T* x, T* y, int len);
//compare op
template<typename T> DLL_EXPORT int equal(Context* ctx, const T* x, const T* y, bool* z, int len);
template<typename T> DLL_EXPORT int not_equal(Context* ctx, const T* x, const T* y, bool* z, int len);
template<typename T> DLL_EXPORT int less_than(Context* ctx, const T* x, const T* y, bool* z, int len);
template<typename T> DLL_EXPORT int less_equal(Context* ctx, const T* x, const T* y, bool* z, int len);
template<typename T> DLL_EXPORT int greater_than(Context* ctx, const T* x, const T* y, bool* z, int len);
template<typename T> DLL_EXPORT int greater_equal(Context* ctx, const T* x, const T* y, bool* z, int len);
//logical op
template<typename T> DLL_EXPORT int logical_not(Context* ctx, const T* x, T* y, int len);
template<typename T> DLL_EXPORT int logical_and(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int logical_or(Context* ctx, const T* x, const T* y, T* z, int len);
template<typename T> DLL_EXPORT int logical_xor(Context* ctx, const T* x, const T* y, T* z, int len);
// reduce op
template<typename T> DLL_EXPORT int reduce_sum(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_mean(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_max(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_min(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_prod(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_L2(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_all(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int reduce_any(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& rdims);

// data movement
template<typename T> DLL_EXPORT int broadcast(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int meshgrid(Context* ctx, const std::vector<const T*>& x_list,
        const std::vector<T*>& y_list,
        const std::vector<std::vector<int> >& xshape_list);
template<typename T> DLL_EXPORT int transpose(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& permute);
template<typename T> DLL_EXPORT int concat(Context* ctx, const std::vector<const T*>& x_list, T* y,
        const std::vector<std::vector<int> >& xshape_list, int axis);
template<typename T> DLL_EXPORT int split(Context* ctx, const T* x, const std::vector<T*>& y_list,
        const std::vector<int>& xshape, const std::vector<int>& split_list, int axis);
template<typename T, typename TID> DLL_EXPORT int gather(Context* ctx, const T* x, const TID* index,
        T* y, const std::vector<int>& xshape, int index_len, int axis);
template<typename T, typename TID> DLL_EXPORT int gather_nd(Context* ctx, const T* x, const TID* index,
        T* y, const VectorParam<int>& xshape, const std::vector<int>& index_shape);
template<typename T, typename TID> DLL_EXPORT int gather_grad(Context* ctx, const T* dy, const TID* index,
        T* dx, const std::vector<int>& xshape, int index_len, int axis, bool overwrite = false);
template<typename T> DLL_EXPORT int pad(Context* ctx, const T* x, T* y, const std::vector<int>& xshape,
        const std::vector<int>& pad_left, const std::vector<int>& pad_right, T pad_value = 0);
template<typename T> DLL_EXPORT int slice(Context* ctx,
        const T* x, T* y, const std::vector<int>& xshape, const std::vector<int>& starts, const std::vector<int>& ends);
template<typename T> DLL_EXPORT int strided_slice(Context* ctx, const T* x, T* y, const std::vector<int>& xshape,
        const std::vector<int>& starts, const std::vector<int>& ends, const std::vector<int>& strides);
template<typename T> DLL_EXPORT int strided_slice_grad(Context* ctx, const T* dy, T* dx, const std::vector<int>& xshape,
        const std::vector<int>& starts, const std::vector<int>& ends, const std::vector<int>& strides);
template<typename T> DLL_EXPORT int nonzero_count(Context* ctx, const T* x, int* y, int len);
template<typename T> DLL_EXPORT int where(Context* ctx, const T* x, int64_t* y, const std::vector<int>& xshape,
        int nonzero_size);
template<typename T> DLL_EXPORT int select(Context* ctx, const bool* condition, const T* x, const T* y, T* z,
        const std::vector<int>& condition_shape, const std::vector<int>& xshape);
template<typename T> DLL_EXPORT int masked_select(Context* ctx, const T* x, const bool* mask, T* y,
        const std::vector<int>& x_shape, const std::vector<int>& mask_shape, int ture_count);
template<typename T> DLL_EXPORT int masked_select_grad(Context* ctx, const T* y, const bool* mask, T* x,
        const std::vector<int>& x_shape, const std::vector<int>& mask_shape, int true_count);
template<typename T, typename TID> DLL_EXPORT int in_topk(Context* ctx, const T* x, const TID* y, bool* z,
        int m, int n, int k);
template<typename T> DLL_EXPORT int left_shift(Context* ctx, T* x, T* y, int* val, int len);
template<typename T> DLL_EXPORT int right_shift(Context* ctx, T* x, T* y, int* val, int len);
template<typename T, typename TID> DLL_EXPORT int scatter(Context* ctx, const T* x,
        T* y, const VectorParam<TID>& indices, int dim0, int dim1, bool is_overwrite);
template<typename T, typename TID> DLL_EXPORT int scatter(Context* ctx, const T* x, const T* updates, T* y,
        const VectorParam<TID>& index, const std::vector<int>& xshape, int axis, bool is_overwrite);
template<typename T, typename TID> DLL_EXPORT int scatter_nd(Context* ctx, const T* x, const T* updates, T* y,
        const VectorParam<TID>& index, const VectorParam<int>& xshape, const std::vector<int>& index_shape,
        bool is_overwrite);

// sorting
template<typename T, typename TID> DLL_EXPORT int sort(Context* ctx, const T* x, T* y, TID* index, int m, int n,
        bool compare = false);
template<typename T> DLL_EXPORT int sorted_topk(Context* ctx, const T* x, T* y, int* index, int m, int n, int k,
        bool largest = true);
template<typename TX, typename TY = int64_t> DLL_EXPORT int
argmax(Context* ctx, const TX* x, TY* y, const std::vector<int>& xshape, int axis);
template<typename T, typename TID> DLL_EXPORT int search_sorted(Context* ctx, const T* x, const T* values, TID* y,
        int m, int xn, int yn, bool is_rightside, bool is_ascend);
// rand generator
template<typename T> DLL_EXPORT int random(Context* ctx, T* x, int len, T min, T max, int seed);
// broadcast ops
template<typename T> DLL_EXPORT int broadcast_add(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_sub(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_mul(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_div(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_max(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_min(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_pow(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_mod(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_floordiv(Context* ctx, const T* x, const T* y, T* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_equal(Context* ctx, const T* x, const T* y, bool* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_greater_than(Context* ctx, const T* x, const T* y, bool* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_greater_equal(Context* ctx, const T* x, const T* y, bool* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_less_than(Context* ctx, const T* x, const T* y, bool* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_less_equal(Context* ctx, const T* x, const T* y, bool* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_not_equal(Context* ctx, const T* x, const T* y, bool* z,
        const std::vector<int>& xshape, const std::vector<int>& yshape);
// broadcast_grad ops
template<typename T> DLL_EXPORT int broadcast_add_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dy, T* dx, const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_sub_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dy, T* dx, const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_mul_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dy, T* dx, const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_div_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dy, T* dx, const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_max_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dy, T* dx, const std::vector<int>& xshape, const std::vector<int>& yshape);
template<typename T> DLL_EXPORT int broadcast_min_grad(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz, T* dy, T* dx, const std::vector<int>& xshape, const std::vector<int>& yshape);
// fast Fourier transform
template<typename T> DLL_EXPORT int fft(Context* ctx, int batch_size, int N,
        const T* x_r, const T* x_i, T* y_r, T* y_i);
template<typename T> DLL_EXPORT int ifft(Context* ctx, int batch_size, int N,
        const T* x_r, const T* x_i, T* y_r, T* y_i);
template<typename T> DLL_EXPORT int fft2d(Context* ctx, const std::vector<int>& shape,
        const T* x_r, const T* x_i, T* y_r, T* y_i);
template<typename T> DLL_EXPORT int fft3d(Context* ctx, const std::vector<int>& shape,
        const T* x_r, const T* x_i, T* y_r, T* y_i);
template<typename T> DLL_EXPORT int sum(Context* ctx, const std::vector<const T*>& x_list, T* y, int len);
template<typename T> DLL_EXPORT int logsumexp(Context* ctx, const T* x, T* y,
        const std::vector<int>& xshape, const std::vector<int>& axis);
template<typename T> DLL_EXPORT int logsumexp_grad(Context* ctx, const T* x, const T* y, const T* dy,
        T* dx, const std::vector<int>& xshape, const std::vector<int>& axis_shape);
//matrix
template<typename T> DLL_EXPORT int tril(Context* ctx, const T* x,  T* y,
        const std::vector<int>& xshape, int diagonal);
template<typename T> DLL_EXPORT int triu(Context* ctx, const T* x,  T* y,
        const std::vector<int>& xshape, int diagonal);
template<typename T> DLL_EXPORT int rbf_kernel(Context* ctx, const T* x, T* y, int len, float mu, float sigma);
template<typename T> DLL_EXPORT int roll(Context* ctx, const T* x, T* y,
        std::vector<int>& xshape, const std::vector<int>& shifts, const std::vector<int>& axis);
template<typename T> DLL_EXPORT int histogram(Context* ctx,
        const int* x, const T* weight, int* y, T* weight_y, int xlen, int ylen);

}
}
}
#endif
