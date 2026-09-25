#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_NN_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_NN_H
#include "xpu/refactor/context/newcontext.h"
#include "xpu/xdnn_types.h"
#include "xpu/refactor/deprecated.h"
namespace baidu {
namespace xpu {
namespace api {
// Conv
template<typename T> DLL_EXPORT int im2col(Context* ctx,
        const T* x, T* y, int n, int c, int h, int w, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation, bool is_nchw);
template<typename T> DLL_EXPORT int im2im(Context* ctx,
        const T* x, T* y, int n, int c, int h, int w, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation, bool is_nchw);
template<typename T> DLL_EXPORT int col2im(Context* ctx,
        const T* y, T* x, int n, int c, int xh, int xw, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation, bool is_nchw);
template<typename T> DLL_EXPORT int deformable_im2col(Context* ctx, const T* x,
        const float* offset, const float* mask, T* y, int n, int c, int xh, int xw, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation,
        const int deformable_group, bool is_nchw);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int conv2d(Context* ctx,
        const TX* x, const TW* weight, TY* y, int n, int c, int h, int w, int f, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation, int group,
        const float* x_maxptr, const float* weight_maxptr, float* y_maxptr, bool is_nchw);
template<typename TX, typename TW> DLL_EXPORT int
search_varconv(Context* ctx, int batch, int c, int f, int winh, int winw, int strideh, int stridew,
        const TX* input_im, const TW* weight, const VectorParam<int>& offset_x, const VectorParam<int>& offset_y,
        float* output_im, float max_weight, const Activation_t act, bool is_nchw = true);
DLL_EXPORT int batched_search_varconv(Context* ctx, int batch, int c, int f, int winh, int winw, int strideh,
        int stridew,
        const float* input_im, const float* weight, const VectorParam<int>& offset_x, const VectorParam<int>& offset_y,
        float* output_im, float max_weight, const Activation_t act, bool is_nchw = true);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int conv2d_grad(Context* ctx,
        const TX* x, const TW* weight, const TY* dy, TX* dx, TW* dweight, int n, int c, int h, int w,
        int f, const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation, int group, const float* x_maxptr, const float* w_maxptr,
        const float* dy_maxptr, float* dx_maxptr, float* dw_maxptr, bool is_nchw);
template<typename TY, typename TW, typename TX, typename TGEMM> DLL_EXPORT int conv2d_transpose(
        Context* ctx, const TY* y, const TW* weight, TX* x, int n, int yc, int yh, int yw, int xc,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation, int group, const float* y_maxptr,
        const float* weight_maxptr, float* x_maxptr, bool is_nchw);
template<typename TY, typename TW, typename TX, typename TGEMM> DLL_EXPORT int conv3d_transpose(
        Context* ctx, const TY* y, const TW* weight, TX* x, int n, int yc, int yd, int yh, int yw, int xc,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation, int group, const float* y_maxptr,
        const float* weight_maxptr, float* x_maxptr, bool is_ndhwc);
template<typename T> DLL_EXPORT int col2vol(Context* ctx,
        const T* y, T* x, int n, int c, int xd, int xh, int xw, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation, bool is_ndhwc);
template<typename TY, typename TW, typename TX, typename TGEMM> DLL_EXPORT int conv2d_transpose_grad(
        Context* ctx, const TY* y, const TW* weight, const TX* dx, TY* dy, TW* dweight,
        int n, int yc, int yh, int yw, int xc, int xh, int xw,
        const std::vector<int>& _ksize, const std::vector<int>& _stride, const std::vector<int>& _pad,
        const std::vector<int>& _dilation, int group, const float* y_maxptr, const float* w_maxptr,
        const float* dx_maxptr, float* dy_maxptr, float* dw_maxptr, bool is_nchw);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int conv3d(Context* ctx,
        const TX* x, const TW* weight, TY* y, int n, int c, int d, int h, int w, int f, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation, int group,
        const float* x_maxptr, const float* weight_maxptr, float* y_maxptr, bool is_ncdhw);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int deformable_conv(
        Context* ctx, const TX* x, const TW* weight, const float* offset, const float* mask,
        TY* y, int n, int c, int h, int w, int f,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation, int group, int deformable_group,
        const float* x_maxptr, const float* w_maxptr, float* y_maxptr, bool is_nchw);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int deformable_conv_grad(
        Context* ctx, const TX* x, const TW* weight, const float* offset, const float* mask,
        const TY* dy, TX* dx, TW* dw, float* doffset, float* dmask, int n, int c, int h, int w, int f,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation, int group, int deformable_group,
        const float* x_maxptr, const float* w_maxptr, float* dy_maxptr, float* dx_maxptr, float* dw_maxptr, bool is_nchw);
// pooling

// 1d
template<typename T> DLL_EXPORT int avg_pool1d(Context* ctx, const T* x, T* y, int n, int c, int w,
        int ksize, int stride, const std::vector<int>& pad, bool count_include_pad, bool is_ncw,
        const float* x_maxptr = nullptr, float* y_maxptr = nullptr);

template<typename T> DLL_EXPORT int max_pool1d(Context* ctx, const T* x, T* y, int* indices, int n, int c, int w,
        int ksize, int stride, const std::vector<int>& pad, bool is_ncw,
        const float* x_maxptr = nullptr, float* y_maxptr = nullptr);

//2d
template<typename T> DLL_EXPORT int avg_pool2d(Context* ctx, const T* x, T* y, int n, int c, int h, int w,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        bool count_include_pad, bool is_nchw, const float* x_maxptr = nullptr, float* y_maxptr = nullptr);
template<typename T> DLL_EXPORT int max_pool2d(Context* ctx, const T* x, T* y, int* indices, int n, int c, int h, int w,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad, bool is_nchw,
        const float* x_maxptr = nullptr, float* y_maxptr = nullptr);
template<typename T> DLL_EXPORT int avg_pool2d_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx,
        int n, int c, int h, int w, const std::vector<int>& ksize, const std::vector<int>& stride,
        const std::vector<int>& pad, bool count_include_pad, bool is_nchw);
template<typename T> DLL_EXPORT int max_pool2d_grad(Context* ctx, const T* x, const T* y, const int* indices,
        const T* dy,
        T* dx, int n, int c, int h, int w, const std::vector<int>& ksize, const std::vector<int>& stride,
        const std::vector<int>& pad, bool is_nchw);
template<typename T> DLL_EXPORT int adaptive_avg_pool2d(Context* ctx, const T* x, T* y,
        int n, int c, int xh, int xw, int yh, int yw, bool is_nchw);
template<typename T> DLL_EXPORT int adaptive_max_pool2d(Context* ctx, const T* x, T* y, int* indices,
        int n, int c, int xh, int xw, int yh, int yw, bool is_nchw);

//3d
template<typename T> DLL_EXPORT int adaptive_avg_pool3d(Context* ctx, const T* x, T* y,
        int n, int c, int xd, int xh, int xw, int yd, int yh, int yw, bool is_ncdhw);
template<typename T> DLL_EXPORT int adaptive_max_pool3d(Context* ctx, const T* x, T* y, int* indice,
        int n, int c, int xd, int xh, int xw, int yd, int yh, int yw, bool is_ncdhw);

// Norm
template<typename T> DLL_EXPORT int batch_norm(Context* ctx, const T* x, T* y, int n, int c, int h, int w,
        float eps, float momentum, const float* scale, const float* bias,
        float* batch_mean, float* batch_inv_std, float* global_mean, float* global_var, bool is_nchw);
template<typename T> DLL_EXPORT int batch_norm_grad(Context* ctx, const T* x, const T* dy, T* dx,
        int n, int c, int h, int w, const float* scale, const float* batch_mean,
        const float* batch_inv_std, float* dscale, float* dbias, bool is_nchw, const float* global_mean = nullptr,
        const float* global_var = nullptr, const float epsilon = 1e-5f);
template<typename T> DLL_EXPORT int batch_norm_infer(Context* ctx, const T* x, T* y, int n, int c, int h,
        int w, float eps, const float* scale, const float* bias, const float* batch_mean,
        const float* batch_var, bool is_nchw);
template<typename T> DLL_EXPORT int layer_norm(Context* ctx, const T* x, T* y, int m, int n, float eps,
        const float* scale, const float* bias, float* mean, float* var);
template<typename T> DLL_EXPORT int layer_norm_grad(Context* ctx, const T* x, const T* dy, T* dx, int m, int n,
        float eps, const float* scale, const float* mean, const float* var, float* dscale, float* dbias);
template<typename T> DLL_EXPORT int l2_norm(Context* ctx,
        const T* x, T* y, T* norm, const std::vector<int>& xshape, int axis, float eps);
template<typename T> DLL_EXPORT int lrn(Context* ctx, const T* x, T* y,
        int n, int c, int h, int w, int m, float k, float alpha, float beta);
template<typename T> DLL_EXPORT int instance_norm(Context* ctx, const T* x, T* y, int n, int c, int h, int w,
        float eps, const float* scale, const float* bias, float* saved_mean, float* saved_var, bool is_nchw);
template<typename T> DLL_EXPORT int instance_norm_grad(Context* ctx, const T* x, const T* dy,  T* dx,
        const float* scale, const float* mean, const float* var, float* dscale, float* dbias,
        int n, int c, int h, int w, float eps, bool is_nchw);
template<typename T> DLL_EXPORT int clip_by_norm(Context* ctx, const T* x, T* y, float max_norm,
        const std::vector<int>& xshape, const std::vector<int>& rdims);
template<typename T> DLL_EXPORT int group_norm(Context* ctx, const T* x, T* y, int n, int c, int h, int w,
        int groups, float eps, const float* scale, const float* bias, float* mean, float* var, bool is_nchw);
// Activation
// Simple Activation
template<typename T> DLL_EXPORT int elu(Context* ctx, const T* x, T* y, int len,
        float alpha = 1.0f, const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int selu(Context* ctx, const T* x, T* y, int len,
        float scale = 1.050701, float alpha = 1.6732632, const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int celu(Context* ctx, const T* x, T* y, int len,
        float alpha = 1.0f, const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int relu(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int relu6(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int leaky_relu(Context* ctx, const T* x, T* y, int len, float alpha,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int hard_sigmoid(Context* ctx, const T* x, T* y, int len, float slope,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int logit(Context* ctx, const T* x, T* y, int len, float eps,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int mish(Context* ctx, const T* x, T* y, int len, float threshold = 20.0f,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int sigmoid(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int tanh(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int gelu(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int approximate_gelu(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int swish(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int hard_swish(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int softsign(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int tanhsoft(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int fast_sigmoid(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int fast_tanh(Context* ctx, const T* x, T* y, int len,
        const float* max_x = nullptr, float* max_y = nullptr);
// Deprecated@2.6
template<typename T> DLL_EXPORT int prelu(Context* ctx, const T* x, const T* alpha, T* y, int m, int t, int n);

template<typename T, typename TS> DLL_EXPORT int prelu(Context* ctx, const T* x, const TS* slope, T* y,
        const std::vector<int>& xshape, const std::vector<int>& slope_shape,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int softplus(Context* ctx, const T* x, T* y, int len, float beta = 1.0f,
        float threshold = 20.0f);
template<typename T> DLL_EXPORT int relu_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int relu6_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int sigmoid_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int tanh_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int hard_swish_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int gelu_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int softsign_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int leaky_relu_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx,
        int len, float alpha);
template<typename T> DLL_EXPORT int softplus_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx,
        int len, float beta = 1.0f, float threshold = 20.0f);
template<typename T> DLL_EXPORT int swish_grad(Context* ctx, const T* x, const T* dy, T* dx, int len);
template<typename T> DLL_EXPORT int pow_grad(Context* ctx, const T* x, const T* dy, T* dx, int len, float factor);
template<typename T> DLL_EXPORT int mish_grad(Context* ctx, const T* x, const T* y, const T* dy, T* dx, int len,
        float threshold = 20.0f);
// Complex Activation
template<typename T> DLL_EXPORT int softmax(Context* ctx, const T* x, T* y, const std::vector<int>& xshape, int axis);
template<typename T> DLL_EXPORT int softmax_grad(Context* ctx, const T* y, const T* dy, T* dx,
        const std::vector<int>& xshape, int axis);
template<typename T> DLL_EXPORT int label_smooth(Context* ctx, const T* x, T* y, int n, float epsilon, int label_dim);
// FC
template<typename T> DLL_EXPORT int per_row_norm(Context* ctx, const T* x, T* y, T* scales, int row, int col,
        T norm_max);
template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT int fc(Context* ctx, const TX* x, const TW* w, TY* y, int m, int n, int k, bool x_trans, bool w_trans,
        const float* x_maxptr, const float* w_maxptr, float* y_maxptr);
template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT int fc_batched(Context* ctx, int batch_size, bool x_trans, bool w_trans, int m, int n, int k,
        float alpha, const TX* x, int stride_a, const TW* w, int stride_b, float beta, TY* y, int stride_c,
        const float* x_maxptr, const float* w_maxptr);
template<typename TX, typename TW, typename TY, typename TID, typename TGEMM>
DLL_EXPORT int fc_batched_vsl(Context* ctx, const TX* x, const TW* w, TY* y, const VectorParam<TID>& m_list,
        const VectorParam<TID>& n_list, const VectorParam<TID>& k_list, bool x_trans, bool w_trans, float alpha,
        bool need_softmax);

// Loss
template<typename T> DLL_EXPORT int log_loss(Context* ctx, const T* predict, const T* labels,
        T* loss, int batch, float epsilon);
template<typename T> DLL_EXPORT int log_loss_grad(Context* ctx, const T* predict, const T* labels,
        const T* dloss, T* dpredict,  int batch, float epsilon);
template<typename T, typename TID> DLL_EXPORT int hard_cross_entropy(Context* ctx, const T* x, const TID* label,
        T* y, T* match_x, int m, int n, int ignore_index);
template<typename T, typename TID> DLL_EXPORT int hard_cross_entropy_grad(Context* ctx, const TID* label,
        const T* match_x, const T* dy, T* dx, int m, int n, int ignore_index);
template<typename T> DLL_EXPORT int soft_cross_entropy(Context* ctx, const T* x, const T* label, T* y, int m, int n);
template<typename T> DLL_EXPORT int soft_cross_entropy_grad(Context* ctx, const T* x, const T* label, const T* dy,
        T* dx, int m, int n);
template<typename T> DLL_EXPORT int sigmoid_cross_entropy_with_logits(Context* ctx,
        const T* x, const T* label, T* y, int m, int n, int* hit = nullptr, int ignore_index = -100);
template<typename T> DLL_EXPORT int sigmoid_cross_entropy_with_logits_grad(Context* ctx,
        const T* x, const T* label, const T* dy, T* dx, int m, int n, int* hit = nullptr, int ignore_index = -100);
template<typename T> DLL_EXPORT int soft_softmax_with_cross_entropy(Context* ctx, const T* x, const T* label,
        T* softmax, T* y, int m, int n);
template<typename T> DLL_EXPORT int soft_softmax_with_cross_entropy_grad(Context* ctx, const T* dy, const T* label,
        const T* softmax, T* dx, bool softmax_switch, int m, int n);
template<typename T, typename TID> DLL_EXPORT int hard_softmax_with_cross_entropy(Context* ctx, const T* x,
        const TID* label, T* softmax, T* y, int ignore_index, int m, int n);
template<typename T, typename TID> DLL_EXPORT int hard_softmax_with_cross_entropy_grad(Context* ctx, const T* dy,
        const TID* label, const T* softmax, T* dx, int ignore_index, bool softmax_switch, int m, int n);
template<typename T> DLL_EXPORT int smooth_l1_loss(Context* ctx, T* x, T* y, T* diff, T* out,
        bool has_weights, T* inside_weights, T* outside_weights, int batch_size, int pos_num, float sigma);
template<typename T> DLL_EXPORT int smooth_l1_loss_grad(Context* ctx, T* dx, T* dy, T* diff, T* dout,
        bool has_weights, T* inside_weights, T* outside_weights, int batch_size, int pos_num, float sigma);
template<typename T> DLL_EXPORT int huber_loss(Context* ctx, const T* x, const T* y, T* diff, T* out,
        int batch_size, int pos_num, float delta);
template<typename T> DLL_EXPORT int huber_loss_grad(Context* ctx, const T* diff, const T* dout, T* dx, T* dy,
        int batch_size, int pos_num, float delta);
template<typename T> DLL_EXPORT int nll_loss(Context* ctx, const T* x, T* y,
        const std::vector<int>& x_shape, const int32_t* target, int ignore_index = -100);
template<typename T> DLL_EXPORT int nll_loss_grad(Context* ctx, const float* dy,  T* dx,
        const std::vector<int>& shape, const int32_t* target, int ignore_index = -100);
template<typename T> DLL_EXPORT int cos_sim(Context* ctx, const T* x, const T* y, T* z, int xm, int ym, int n,
        T* x_norm, T* y_norm);
template<typename T> DLL_EXPORT int cos_sim_grad(Context* ctx, const T* x, const T* y, const T* z, const T* dz,
        T* dx, T* dy, int xm, int ym, int n, const T* x_norm, const T* y_norm);
template<typename T> DLL_EXPORT int bce_loss(Context* ctx, const T* x, const T* label, T* y, int len);
template<typename T> DLL_EXPORT int bce_loss_grad(Context* ctx, const T* x, const T* label, const T* dy, T* dx,
        int len);
template <typename T> DLL_EXPORT int as_strided(Context* ctx, const T* x, T* y,
        const std::vector<int>& yshape,
        const std::vector<int>& strides);
template <typename T> DLL_EXPORT int as_strided_view_update(Context* ctx, const T* x, T* y,
        const std::vector<int>& xshape,
        const std::vector<int>& strides);
template<typename T> DLL_EXPORT int strided_slice_view_update(Context* ctx, const T* x, T* y,
        const std::vector<int>& xshape,
        const std::vector<int>& yshape,
        const std::vector<int>& starts,
        const std::vector<int>& ends,
        const std::vector<int>& strides);


// Optimizer
template<typename T> DLL_EXPORT int adam(Context* ctx,
        const T* g, const float* mom1, const float* mom2, const T* param,
        const float* beta1_pow, const float* beta2_pow, const float* lr,
        float* moment1_out, float* moment2_out, T* param_out,
        float beta1, float beta2, float epsilon, int n);
template<typename T> DLL_EXPORT int sparse_adam(Context* ctx,
        const T* g, const float* mom1, const float* mom2, const T* param,
        const float* beta1_pow, const float* beta2_pow, const float* lr,
        float* moment1_out, float* moment2_out, T* param_out,
        float beta1, float beta2, float epsilon, int ori_rows,
        const int* rows, int row_numel, int row_count, int lazy_mode);
template<typename T> DLL_EXPORT int adamw(Context* ctx,
        const T* g, const float* mom1, const float* mom2, const T* param,
        const float* beta1_pow, const float* beta2_pow, const float* lr,
        float* moment1_out, float* moment2_out, T* param_out,
        float beta1, float beta2, float epsilon, float coeff, int n);
template<typename T> DLL_EXPORT int lamb(Context* ctx, const T* g, const float* mom1,
        const float* mom2, const T* param, const float* beta1_pow, const float* beta2_pow,
        float* mom1_out, float* mom2_out, T* param_out, float* beta1_pow_out, float* beta2_pow_out,
        float beta1, float beta2, float epsilon, float weight_decay, const float* lr, int n, float param_norm_clamp_value = 0);
template<typename T> DLL_EXPORT int momentum(Context* ctx,
        const T* param, const T* velocity, const T* grad, T* param_out, T* velocity_out,
        int len, const float* lr, int use_nesterov, float mu, float l2_weight_decay = 0.0f);
template<typename T> DLL_EXPORT int rmsprop(Context* ctx, const T* g, const T* p, const float* ms, const float* mom,
        T* p_out, float* ms_out, float* mom_out, float epsilon, float rho, float momentum, float lr, int n);
template<typename T> DLL_EXPORT int sgd(Context* ctx, const T* grad, const T* param,
        const float* lr, T* param_out, int n);
template <typename T> DLL_EXPORT int merged_momentum(Context* ctx, const std::vector<T*>& param_list,
        const std::vector<T*>& velocity_list,
        const std::vector<T*>& grad_list,
        std::vector<T*>& param_out_list,
        std::vector<T*>& velocity_out_list,
        const std::vector<float>& l2_weight_decay,
        const std::vector<int>& sizes, const float* lr,
        float mu, int use_nesterov);
template <typename T> DLL_EXPORT int lars_momentum(Context* ctx, const std::vector<T*>& param_list,
        const std::vector<T*>& grad_list,
        const std::vector<float*>& velocity_list, const std::vector<float*>& lrs,
        const std::vector<float*>& master_param_list,
        const std::vector<T*>& param_out_list, const std::vector<float*>& velocity_out_list,
        const std::vector<float*>& master_param_out_list, const std::vector<float>& lars_weight_decay,
        const std::vector<int>& param_sizes, float mu, float lars_coeff, float epsilon,
        float rescale_grad);
// Vision
template<typename T> DLL_EXPORT int nearest_resize1d(Context* ctx, const T* x, T* y, int n, int c,
        int xw, int yw, int coordinate_transformation_mode, int nearest_mode, bool is_ncw);
template<typename T> DLL_EXPORT int nearest_resize2d(Context* ctx, const T* x, T* y, int n, int c,
        int xh, int xw, int yh, int yw, int coordinate_transformation_mode, int nearest_mode, bool is_nchw);

template<typename T> DLL_EXPORT int space_to_depth(Context* ctx, const T* x, T* y, int n, int xc, int xh, int xw,
        int block_size, bool is_nchw);
template<typename T> DLL_EXPORT int depth_to_space(Context* ctx, const T* x, T* y, int n, int xc, int xh, int xw,
        int block_size, bool is_nchw);
template<typename T> DLL_EXPORT int pixel_shuffle(Context* ctx, const T* x, T* y, int n, int xc, int xh, int xw,
        int block_size, bool is_nchw);
template<typename T> DLL_EXPORT int interpolate2d(Context* ctx, const T* x, T* y, int n, int c, int xh, int xw,
        int yh, int yw, bool is_nearest, int trans_mode, bool is_nchw,
        const float* max_x = nullptr, float* max_y = nullptr);
template<typename T> DLL_EXPORT int interpolate2d_grad(Context* ctx, const T* dy, T* dx, int n, int c, int xh, int xw,
        int yh, int yw, bool is_nearest, int trans_mode, bool is_nchw);
template<typename T> DLL_EXPORT int interpolate1d(Context* ctx, const T* x, T* y, int n, int c, int xlen,
        int ylen, bool is_nearest, int trans_mode, bool is_ncw);
template<typename T, typename TID> DLL_EXPORT int roi_align(Context* ctx, const T* x, T* y,
        const T* rois, const TID* lod, int xn, int c, int xh, int xw, int yn, int yh, int yw,
        float spatial_scale, int sampling_ratio, bool is_nchw, bool continuous_coordinate = false, bool has_theta = false);
template<typename T, typename TID> DLL_EXPORT int roi_align_grad(Context* ctx, const T* dy, T* dx,
        const T* rois, const TID* lod, int xn, int c, int xh, int xw, int yn, int yh, int yw,
        float spatial_scale, int sampling_ratio, bool is_nchw, bool continuous_coordinate = false);
template<typename T> DLL_EXPORT int density_prior_box(Context* ctx, T* boxes,
        int img_h, int img_w, int feature_h, int feature_w,
        const std::vector<float>& fixed_sizes, const std::vector<float>& fixed_ratios, const std::vector<int>& densities,
        float step_w, float step_h, float offset, bool is_clip);
template<typename T> DLL_EXPORT int pad2d(Context* ctx, const T* x, T* y, int n, int c, int h, int w,
        const std::vector<int>& pad, const char* mode, T value = 0, bool is_nchw = true);
template<typename T> DLL_EXPORT int anchor_generator(Context* ctx, T* anchors, int h, int w,
        const std::vector<float>& aspect_ratios, const std::vector<float>& anchor_sizes,
        const std::vector<float>& strides, float offset);
template<typename T> DLL_EXPORT int remove_small_boxes(Context* ctx, const T* boxes, const T* im_info, int* index,
        int* n_keep, int n_boxes, float min_size, bool is_scale = true, bool pixel_offset = true);
template<typename T> DLL_EXPORT int box_decoder(Context* ctx, const T* prior_box, const T* prior_box_var,
        const T* target_box,
        T* proposals, int n_boxes, bool normalized, bool clip = false, const T* im_info = nullptr);
template<typename T> DLL_EXPORT int box_coder_encoder(Context* ctx, const T* prior_box, const T* target_box,
        const T* prior_box_var, const T* variance, T* output, int row, int col, bool normalized = true);
template<typename T> DLL_EXPORT int box_coder_decoder(Context* ctx, const T* prior_box, const T* target_box,
        const T* prior_box_var, const T* variance, T* output, int row, int col, int axis = 0, bool normalized = true);
template<typename T> DLL_EXPORT int yolo_box(Context* ctx, const T* input, const int* img_size, T* boxes_data,
        T* scores_data,
        int n, int h, int w, const std::vector<int>& anchors, int anchor_num, int class_num,
        float conf_thresh, int downsample_ratio, float scale = 1.0f, float bias = 0.0f, bool score_transpose = false);
template<typename T> DLL_EXPORT int sorted_nms(Context* ctx, const T* boxes, int* index, int& n_keep, int n_boxes,
        float iou_thres,
        bool pixel_offset = true);
template<typename T> DLL_EXPORT int accuracy(Context* ctx, const T* x, const T* y, int m, int n,
        int* correct, int* total, float* accuracy);
template<typename T> DLL_EXPORT int clip_box_to_image(Context* ctx, const T* boxes, T* clipped_boxes, int n_boxes,
        int h, int w);
template<typename T> DLL_EXPORT int correlation(Context* ctx, const T* x, const T* y, T* z, int n, int xc, int xh,
        int xw, int pad_size, int kernel_size, int stride1, int stride2, int max_displacement, int corr_type_multiply);
template<typename T> DLL_EXPORT int grid_sample(Context* ctx, const T* x, const T* grid, T* y, int n, int c, int xh,
        int xw,
        int yh, int yw, bool is_nearest, bool align_corners, int padding_mode, bool is_nchw);
template<typename T> DLL_EXPORT int iou_similarity(Context* ctx, const T* x, const T* y, T* z, int m, int n,
        float eps, bool normalized);
template<typename T> DLL_EXPORT int gen_prior_box(Context* ctx, T* boxes,
        const VectorParam<float>& aspect_ratios, const VectorParam<float>& min_sizes,
        const VectorParam<float>& max_sizes, int height, int width, int im_height,
        int im_width, float offset, float step_height, float step_width, bool is_clip,
        bool min_max_aspect_ratios_order);
template<typename T> DLL_EXPORT int region(Context* ctx, const T* x, T* y, int n, int w, int h,
        int num_box, int classes, int coords);
template<typename T> DLL_EXPORT int temporal_shift(Context* ctx, const T* x, T* y, int n,
        int c, int h, int w, int t, float shift_ratio, bool is_nhwc);
template<typename T> DLL_EXPORT int temporal_shift_grad(Context* ctx, const T* dy, T* dx, int n,
        int c, int h, int w, int t, float shift_ratio, bool is_nhwc);
//REC
template<typename T, typename TID> DLL_EXPORT int merge_dup_rows(Context* ctx, const T* x_data, T* y_data,
        const TID* x_rows,
        const TID* y_rows, int xm, int n, int ym);
// NLP
template<typename T, typename TID> DLL_EXPORT int embedding(Context* ctx, const T* x, const TID* indices, T* y,
        int xm, int n, int ym, int padding_idx, TID start_index = 0);
template<typename T, typename TID> DLL_EXPORT int embedding_grad(Context* ctx, const T* dy, const TID* indices, T* dx,
        int xm, int n, int ym, int padding_idx);
template<typename T, typename TID> DLL_EXPORT int sequence_max_pool(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lod,
        int batch, int dim, float pad_value, TID* max_index);
template<typename T, typename TID> DLL_EXPORT int sequence_first_pool(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lod,
        int batch, int dim, float pad_value);
template<typename T, typename TID> DLL_EXPORT int sequence_last_pool(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lod,
        int batch, int dim, float pad_value);
template<typename T, typename TID> DLL_EXPORT int sequence_sum_pool(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lod,
        int batch, int dim, float pad_value);
template<typename T, typename TID> DLL_EXPORT int sequence_max_pool_grad(Context* ctx, const T* dy, const TID* lod,
        T* dx, int batch, int dim, const TID* max_index);
template<typename T, typename TID> DLL_EXPORT int sequence_first_pool_grad(Context* ctx, const T* dy, const TID* lod,
        T* dx, int batch, int dim);
template<typename T, typename TID> DLL_EXPORT int sequence_last_pool_grad(Context* ctx, const T* dy, const TID* lod,
        T* dx, int batch, int dim);
template<typename T, typename TID> DLL_EXPORT int sequence_sum_pool_grad(Context* ctx, const T* dy, const TID* lod,
        T* dx, int batch, int dim);
template <typename T, typename TID> DLL_EXPORT int sequence_pad(Context* ctx, const T* x, const TID* lod, T* y,
        int batch, int seqlen, int dim, float pad_value);
template <typename T, typename TID> DLL_EXPORT int sequence_pad(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lod,
        int batch, int seqlen, int dim, float pad_value);
template <typename T, typename TID> DLL_EXPORT int sequence_pad(Context* ctx, const T* x, const TID* lod,
        T* y, int batch, int max_seq_len, int dim, T* pad_value_ptr, int pad_value_size, TID* length_ptr);
template <typename T, typename TID> DLL_EXPORT int sequence_slice(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lodx,
        const VectorParam<int>& offset, const VectorParam<TID>& lody, int dims);
template <typename T, typename TID> DLL_EXPORT int sequence_unpad(Context* ctx,
        const T* x, T* y, const VectorParam<TID>& lody, int max_seqlen, int dim);
template<typename T, typename TID> DLL_EXPORT int sequence_topk_avg_pooling(Context* ctx, const T* x, T* y, TID* y_pos,
        int channel_num,
        const VectorParam<TID>& row_lod,
        const VectorParam<TID>& col_lod,
        const VectorParam<int>& topks);
template<typename T, typename TID> DLL_EXPORT int sequence_expand(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& lodx,
        const VectorParam<TID>& lody,
        const VectorParam<TID>& lod_ref,
        int dims);
template <typename TX, typename TY> DLL_EXPORT int sequence_mask(Context* ctx, const TX* x, TY* y,
        int batch, int max_seq_len);
template <typename T, typename TID> DLL_EXPORT int sequence_reverse(Context* ctx, const T* x, T* y,
        const VectorParam<TID>& x_lod, int dim);
template <typename T> DLL_EXPORT int sequence_concat(Context* ctx, const std::vector<const T*>& x_list,
        const std::vector<std::vector<int>>& seqlens_list, T* y, int dim);
template <typename T, typename TID> DLL_EXPORT int sequence_context_projection(Context* ctx,
        const T* x, T* y, const T* padding_data, const VectorParam<TID>& lodx, int dim,
        int context_start, int context_len, int context_stride, const std::vector<int>& pad);
template <typename T, typename TID> DLL_EXPORT int sequence_context_projection_grad(Context* ctx,
        T* x, const T* y, const T* padding_data, const VectorParam<TID>& lodx, int dim,
        int context_start, int context_len, int context_stride, const std::vector<int>& pad);
template <typename T, typename TID> DLL_EXPORT int sequence_to_batch(Context* ctx, const T* x, T* y, int dim,
        const VectorParam<TID>& idx_sorted, const VectorParam<TID>& x_lod, const VectorParam<TID>& y_lod,
        bool is_reverse = false);
template <typename T, typename TID> DLL_EXPORT int batch_to_sequence(Context* ctx, const T* y, T* x, int dim,
        const VectorParam<TID>& idx_sorted, const VectorParam<TID>& y_lod, const VectorParam<TID>& x_lod,
        bool is_reverse = false);
template <typename T, typename TID> DLL_EXPORT int attention_padding_mask(Context* ctx, const T* x,
        const TID* pad_begin, T* y,
        int att_batch, int att_len, int src_len, int src_batch, float mask);

template<typename T> DLL_EXPORT int one_hot(Context* ctx, const T* x, float* y, int len, int depth,
        float on_value = 1.0f, float off_value = 0.0f);
template<typename T> DLL_EXPORT int nms(Context* ctx, const T* boxes, const T* scores, int* keep,
        int n, int c, int n_boxes, int nms_topk, float iou_thres, float score_thres, int box_format, bool pixel_offset = true);
template<typename T, typename TID>
DLL_EXPORT int multiclass_nms2(Context* ctx, const T* bboxes, const T* scores,
        std::vector<T>& out, std::vector<TID>& out_index, std::vector<size_t>& accumlated_det_num,
        int n, int b, int class_num, int out_dim, int nms_topk,
        float score_threshold, int keep_top_k, float nms_threshold,
        int background_label, bool normalized, float nms_eta, bool return_index);
template<typename T, typename TID>
DLL_EXPORT int multiclass_nms3(Context* ctx, const T* bboxes, const T* scores,
        const std::vector<int>& rois_num, std::vector<T>& out, std::vector<TID>& out_index,
        std::vector<size_t>& accumlated_det_num, int n, int b, int class_num, int out_dim,
        int nms_topk, float score_threshold, int keep_top_k, float nms_threshold,
        int background_label, bool normalized, float nms_eta, bool return_index, bool is_lod);
template<typename T> DLL_EXPORT int polygon_box_transform(Context* ctx, const T* input, T* output,
        int batch, int geo_channel, int height, int width);
template<typename T> DLL_EXPORT int dropout(Context* ctx, const T* input, T* res, T* mask,
        unsigned int seed, int n, bool is_upscale, float dropout_prob);

template<typename T> DLL_EXPORT int dropout_grad(Context* ctx, const T* mask, const T* dy, T* dx, float dropout_prob,
        int n);

DLL_EXPORT int pow2_decay_with_linear_warmup(Context* ctx, float* lr, int64_t* step,
        int64_t warmup_steps, int64_t total_steps, float base_lr, float end_lr);

template<typename T> DLL_EXPORT int cvm(Context* ctx, const T* x, T* y, int batch_size, int len, bool use_cvm);
// GRU
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT
int gru_unit(Context* ctx,
        const TX* x/*input*/, const TX* hidden_prev/*can be nullptr*/, const TW* weight, TY* y/*hidden*/,
        int batch, int hdim,
        const float* x_maxptr/*can be nullptr*/, const float* hidden_prev_maxptr/*can be nullptr*/,
        const float* w_maxptr/*float[8]*/, float* y_maxptr/*can be nullptr*/,
        const float* bias/*can be nullptr*/,
        const Activation_t& act = Activation_t::TANH,
        const Activation_t& gate_act = Activation_t::SIGMOID,
        bool origin_mode = false);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT
int gru_core(Context* ctx,
        const TX* x/*input*/, const TX* hidden_prev/*can be nullptr*/, const TW* weight, TY* y/*hidden*/,
        int batch, int seq_len, int hdim,
        const float* x_maxptr/*can be nullptr*/, const float* hidden_prev_maxptr/*can be nullptr*/,
        const float* w_maxptr/*float[8]*/, float* y_maxptr/*can be nullptr*/,
        const float* bias/*can be nullptr*/,
        const Activation_t& act = Activation_t::TANH,
        const Activation_t& gate_act = Activation_t::SIGMOID,
        bool origin_mode = false,
        bool is_reverse = false,
        bool reset_after = false);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT
int gru_core(Context* ctx,
        const TX* x/*input*/, const TX* hidden_prev/*can be nullptr*/, const TW* weight, TY* y/*hidden*/,
        const std::vector<uint64_t>& seq_len_lod, int hdim,
        const float* x_maxptr/*can be nullptr*/, const float* hidden_prev_maxptr/*can be nullptr*/,
        const float* w_maxptr/*float[8]*/, float* y_maxptr/*can be nullptr*/,
        const float* bias/*can be nullptr*/,
        const Activation_t& act = Activation_t::TANH,
        const Activation_t& gate_act = Activation_t::SIGMOID,
        bool origin_mode = false,
        bool is_reverse = false);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT
int bigru_core(Context* ctx,
        const TX* fw_x/*fw_input*/, const TX* bw_x/*bw_input*/,
        const TX* fw_hidden_prev, const TX* bw_hidden_prev, // can be nullptr
        const TW* fw_weight, const TW* bw_weight,
        TY* fw_y/*fw_hidden*/, TY* bw_y/*bw_hidden*/,
        int batch, int seq_len, int hdim,
        const float* fw_x_maxptr, const float* bw_x_maxptr, // can be nullptr
        const float* fw_hidden_prev_maxptr, const float* bw_hidden_prev_maxptr, // can be nullptr
        const float* fw_w_maxptr, const float* bw_w_maxptr, // float[8]
        float* fw_y_maxptr, float* bw_y_maxptr, // can be nullptr
        const float* fw_bias, const float* bw_bias, // can be nullptr
        const Activation_t& act = Activation_t::TANH,
        const Activation_t& gate_act = Activation_t::SIGMOID,
        bool origin_mode = false);
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT
int bigru_core(Context* ctx,
        const TX* fw_x/*fw_input*/, const TX* bw_x/*bw_input*/,
        const TX* fw_hidden_prev, const TX* bw_hidden_prev, // can be nullptr
        const TW* fw_weight, const TW* bw_weight,
        TY* fw_y/*fw_hidden*/, TY* bw_y/*bw_hidden*/,
        const std::vector<uint64_t>& seq_len_lod, int hdim,
        const float* fw_x_maxptr, const float* bw_x_maxptr, // can be nullptr
        const float* fw_hidden_prev_maxptr, const float* bw_hidden_prev_maxptr, // can be nullptr
        const float* fw_w_maxptr, const float* bw_w_maxptr, // float[8]
        float* fw_y_maxptr, float* bw_y_maxptr, // can be nullptr
        const float* fw_bias, const float* bw_bias, // can be nullptr
        const Activation_t& act = Activation_t::TANH,
        const Activation_t& gate_act = Activation_t::SIGMOID,
        bool origin_mode = false);
template<typename T> DLL_EXPORT
int sequence_softmax(Context* ctx, const T* x, T* y, const std::vector<int>& xshape, int axis,
        const VectorParam<int>& lod);

template <typename T, typename T_IDX>
DLL_EXPORT int bidirection_embedding_add(Context* ctx, const T* x, T* y0, T* y1, const VectorParam<T_IDX>& lodVP,
        const VectorParam<T_IDX>& idx0VP, const VectorParam<T_IDX>& idx1VP,
        int table_len, int dim, T_IDX padding_idx);

template<typename T> DLL_EXPORT int dfmb_psroi_align(Context* ctx,
        const T* x, T* y, const float* rois, int xn, int xc, int xh, int xw, int yn, int yc, int yh, int yw);
template<typename T> DLL_EXPORT int apollo_roi_pool(Context* ctx,
        const T* x, T* y, int* y_argmax, const float* rois, int xn, int xc, int xh, int xw, int yn, int yc, int yh, int yw);
template<typename T> DLL_EXPORT int rcnn_proposal(Context* ctx,
        const T* cls_score_softmax, const T* bbox_pred, const T* rois, const T* im_info, T* result_boxes, int batch_size,
        const std::vector<float>& bbox_mean_, const std::vector<float>& bbox_std_, const std::vector<float>& thresholds_,
        float min_size_h_, float min_size_w_, float threshold_objectness_, float overlap_ratio_,
        int num_class_, int num_rois_, int max_candidate_n_, int min_size_mode_, int top_n_,
        int out_channel_, int acc_box_num_, bool refine_out_of_map_bbox_ = true,
        bool regress_agnostic_ = false, bool rpn_proposal_output_score_ = true);
template<typename T> DLL_EXPORT int rpn_proposal_ssd(Context* ctx,
        const T* rpn_cls_prob_reshape, const T* rpn_bbox_pred, const T* im_info, const T* anchor_heights_,
        const T* anchor_widths_, T* out_rois,
        int batchSize, int num_anchor_per_point_, int height_, int width_, const std::vector<T>& bbox_mean_,
        const std::vector<T>& bbox_std_, int top_n_,
        int heat_map_a_, bool refine_out_of_map_bbox_, float threshold_objectness_, int min_size_mode_, float min_size_h_,
        float min_size_w_,
        int max_candidate_n_, float overlap_ratio_, int* out_rois_num_);

template<typename T> DLL_EXPORT int apollo_yolo_get_object(Context* ctx,
        int n, const T* loc_data, const T* obj_data, const T* cls_data, const T* ori_data, const T* dim_data,
        const T* lof_data, const T* lor_data, const T* area_id_data,
        const T* visible_ratio_data, const T* cut_off_ratio_data,
        const T* brvis_data, const T* brswt_data, const T* ltvis_data,
        const T* ltswt_data, const T* rtvis_data, const T* rtswt_data,
        const T* anchor_data, const T* expand_data, int width, int height,
        int num_anchors, int num_classes, float confidence_threshold,
        float light_vis_conf_threshold, float light_swt_conf_threshold,
        bool with_box3d, bool with_frbox, bool with_lights, bool with_ratios,
        bool multi_scale, int num_areas, T* res_box_data, T* res_cls_data,
        int res_cls_offset, int all_scales_num_candidates);
template<typename T> DLL_EXPORT int apply_nms(Context* ctx, const T* bbox_data, const T* conf_data,
        const std::vector<int>& origin_indices, const int bbox_step,
        const float confidence_threshold, const int top_k,
        const float nms_threshold, std::vector<int>& indices,
        bool* overlapped, int* idx_sm);
template<typename T> DLL_EXPORT int build_nodes(Context* ctx, const T* offset_map, const T* prob_map,
        uint32_t* center_node, uint16_t* status, int start_row_index, int end_row_index, int rows, int cols,
        float scale, float objectness_threshold);
template<typename T> DLL_EXPORT int build_nodes(Context* ctx, const T* offset_map, const T* prob_map,
        void* nodes, int start_row_index, int end_row_index, int rows, int cols,
        float scale, float objectness_threshold);
template<typename T> DLL_EXPORT int feature_generator(Context* ctx, const T* pc, int* point2grid, T* log_table,
        T* top_intensity_data, T* max_height_data, T* mean_height_data,
        T* mean_intensity_data, T* count_data, T* nonempty_data,
        const int cloud_size, const int map_size, const int max_log_num, bool use_intensity_feature);

}
}
}
#endif
