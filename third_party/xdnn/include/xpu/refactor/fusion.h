#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_FUSION_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_FUSION_H
#include "xpu/refactor/context/newcontext.h"
#include "xpu/xdnn_types.h"
#include "xpu/refactor/deprecated.h"
#include "xpu/refactor/attention.h"

#ifdef _MSC_VER
#include <algorithm>
#endif

namespace baidu {
namespace xpu {
namespace api {

template<typename T> DLL_EXPORT int findmax_copy_fusion(Context* ctx, const T* x, float* maxptr, T* y, int len);

struct DLL_EXPORT ResnetExtraParam {
    // Pad algo of max pooling is *SAME*(defined the same as tensorflow), otherwise, use *DEFAULT* padding({1})
    // With *SAME* padding, yh and yw is computed as:
    // yh = ceil(xh / stride_h), yw = ceil(xw / stride_w)
    // Padding is computed as follows, take pad_up, pad_down as an example:
    // pad_h = max(filter_h - (xh % stride_h == 0 ? stride_h : xh % stride_h), 0)
    // pad_up = pad_h / 2
    // pad_down = pad_h - pad_up
    bool max_pool_pad_algo_use_tf_same;
};

template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT int fc_fusion(Context* ctx, const TX* x, const TW* w, TY* y, int m, int n, int k, bool x_trans,
        bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr, int ldx, int ldw, int ldy,
        float alpha, float beta, const float* bias, const Activation_t& act, const float*  scale = nullptr);

template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT int fc_ny_fusion(Context* ctx, const TX* x, const TW* w, std::vector<TY*> &y_list, int m, int n, int k,
        bool x_trans, bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr, int ldx,
        int ldw, int ldy, float alpha, float beta, const float* bias, const Activation_t& act,
        const float* scale = nullptr);

template <typename TIN, typename TOUT>
DLL_EXPORT int fc_maxvalue_norm(Context* ctx, const TIN* data_in, int row,
        int col, int batch_size, float normal_value,
        const int* lod, TOUT* data_out, float* scale_out, float* maxptr);

template<typename TX, typename TW, typename TY> DLL_EXPORT int fc_fusion_norm(Context* ctx,
        const TX* x, const TW* w, TY* y, int m, int n, int k, bool x_trans, bool w_trans,
        const float* x_maxptr, const float* w_maxptr, float* y_maxptr, int ldx, int ldw,
        int ldy, float alpha, float beta, const float* bias, const Activation_t& act,
        float normal_value, int batch_size, std::vector<int>& lod);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int conv2d_fusion(Context* ctx,
        const TX* x, const TW* weight, TY* y, int n, int c, int h, int w, int f,
        const std::vector<int>& ksize, const std::vector<int>& stride, const std::vector<int>& pad,
        const std::vector<int>& dilation, int group, const float* x_maxptr, const float* weight_maxptr,
        float* y_maxptr, bool is_nchw, const float* bias, const TY* branch, const Activation_t& act,
        const float* branch_maxptr = nullptr);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int fc_fusion_pc(Context* ctx,
        const TX* x, const TW* w, TY* y, int m, int n, int k, bool x_trans, bool w_trans,
        const float* x_maxptr, const float* w_maxptr, float* y_maxptr,
        int ldx, int ldw, int ldy, float alpha, float beta, const float* bias,
        const float* scale, const Activation_t& act);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int fc_fusion_multi_y(Context* ctx,
        const TX* x, const TW* w, std::vector<TY*>& y_list, int m, int n, int k,
        bool x_trans, bool w_trans, const float* x_maxptr, const float* w_maxptr, float* y_maxptr, int ldx,
        int ldw, int ldy, float alpha, float beta, const float* bias, const Activation_t& act,
        const float* scale = nullptr);

template<typename TY, typename TW, typename TX, typename TGEMM> DLL_EXPORT
int conv2d_transpose_fusion(Context* ctx, const TY* y, const TW* weight, TX* x, int n, int yc, int yh, int yw, int xc,
        const std::vector<int>& _ksize, const std::vector<int>& _stride, const std::vector<int>& _pad,
        const std::vector<int>& _dilation, int group,
        const float* y_maxptr, const float* w_maxptr, float* x_maxptr,
        const float* bias, const Activation_t& act, bool is_nchw);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int conv2d_with_pooling(
        Context* ctx, const TX* x, const TW* weight, TY* y, int n, int c, int h, int w, int f,
        const std::vector<int>& _conv_ksize, const std::vector<int>& _conv_stride,
        const std::vector<int>& _conv_pad, const std::vector<int>& _conv_dilation,
        const std::vector<int>& _pool_ksize, const std::vector<int>& _pool_stride,
        const std::vector<int>& _pool_pad, bool count_include_pad, bool is_avg, bool is_nchw,
        const float* x_maxptr, const float* w_maxptr, float* y_maxptr,
        const float* bias, const Activation_t& act);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int conv1d_fusion(Context* ctx,
        const TX* x, const TW* weight, TY* y, int n, int c, int w, int f, int ksize_w, int stride_w,
        const std::vector<int>& pad, int dilation_w, int group, const float* x_maxptr, const float* weight_maxptr,
        float* y_maxptr, bool is_nchw, const float* bias, const TY* branch, const Activation_t& act,
        const float* branch_maxptr = nullptr);

template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT int var_conv2d_fusion(Context* ctx, const TX* x, const TW* weight, TY* y, int n, int c,
        const VectorParam<int>& xh_lod, const VectorParam<int>& xw_lod, int f, const std::vector<int>& ksize,
        const std::vector<int>& stride, const std::vector<int>& pad, const std::vector<int>& dilation,
        int group, const float* x_maxptr, const float* weight_maxptr, float* y_maxptr, bool is_nchw,
        const float* bias, const Activation_t& act);

template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT int var_conv1d_fusion(Context* ctx, const TX* x, const TW* weight, TY* y, int n, int c,
        const VectorParam<int>& xw_lod, int f, int ksize_w, int stride_w, const std::vector<int>& _pad,
        int dilation_w, int group, const float* x_maxptr, const float* w_maxptr, float* y_maxptr,
        bool is_nchw, const float* bias, const Activation_t& act);

template<typename TX, typename TW, typename TY, typename TGEMM>
DLL_EXPORT size_t resnet_unit_fusion_get_reserve_space_size(Context* ctx,
        const std::vector<std::vector<int>>& x_shape_list, int f, const std::vector<std::vector<int>>& ksize_list,
        const std::vector<std::vector<int>>& stride_list, const std::vector<int>& pad, const std::vector<int>& dilation,
        int group, const std::vector<const float*>& x_maxlist, const std::vector<const float*>& w_maxlist,
        const Activation_t& act, bool is_nchw, bool has_shortcut, bool fused_add);

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int resnet_unit_fusion(
        Context* ctx, const std::vector<const TX*>& x_list, const std::vector<const TW*>& w_list,
        const std::vector<TY*>& conv_y_list, TY* y, const std::vector<std::vector<int>>& x_shape_list,
        int f, const std::vector<std::vector<int>>& ksize_list, const std::vector<std::vector<int>>& stride_list,
        const std::vector<int>& pad, const std::vector<int>& dilation, int group, float eps, float momentum,
        const std::vector<const float*>& x_maxlist, const std::vector<const float*>& w_maxlist,
        const std::vector<const float*>& scale_list, const std::vector<const float*>& bias_list,
        const std::vector<float*>& batch_mean_list, const std::vector<float*>& batch_inv_std_list,
        const std::vector<float*>& global_mean_list, const std::vector<float*>& global_var_list,
        const Activation_t& act, bool is_nchw, bool has_shortcut, bool fused_add, bool is_train,
        void* reserve_space = nullptr);

template<typename T, typename TW, typename TGEMM, typename TM> DLL_EXPORT int resnet50(Context* ctx,
        const T* x, const std::vector<const TW*>& weight_list, T* y, int n, int c, int h, int w,
        const std::vector<const float*>& x_maxlist, const std::vector<const float*>& w_maxlist,
        const std::vector<float*>& y_maxlist, const std::vector<const float*>& blist,
        const std::vector<T*>& feature_list, bool is_nchw,
        const ResnetExtraParam& extra_params = {false});

template<typename T, typename TW, typename TGEMM, typename TM> DLL_EXPORT int resnet50_v10(Context* ctx,
        const T* x, const std::vector<const TW*>& weight_list, T* y, int n, int c, int h, int w,
        const std::vector<const float*>& x_maxlist, const std::vector<const float*>& w_maxlist,
        const std::vector<float*>& y_maxlist, const std::vector<const float*>& blist,
        const std::vector<T*>& feature_list, bool is_nchw,
        const ResnetExtraParam& extra_params = {false});

template<typename T, typename TW, typename TGEMM, typename TM> DLL_EXPORT int resnet101(Context* ctx,
        const T* x, const std::vector<const TW*>& weight_list, T* y, int n, int c, int h, int w,
        const float* x_maxptr, const std::vector<const float*>& w_maxptr_list, float* y_maxptr,
        const std::vector<const float*>& bias_list, const std::vector<T*>& feature_list, bool is_nchw,
        const ResnetExtraParam& extra_params = {false});

template<typename T, typename TW, typename TGEMM, typename TM> DLL_EXPORT int resnet34(Context* ctx,
        const T* x, const std::vector<const TW*>& weight_list, T* y, int n, int c, int h, int w,
        const std::vector<const float*>& x_maxlist, const std::vector<const float*>& w_maxlist,
        const std::vector<float*>& y_maxlist, const std::vector<const float*>& blist,
        const std::vector<T*>& feature_list, bool is_nchw,
        const ResnetExtraParam& extra_params = {false});

template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int resnet_unit_grad_fusion(Context* ctx,
        const std::vector<const TX*>& x_list, const std::vector<const TW*>& w_list, const TY* dy,
        const TY* y, const std::vector<const TY*>& conv_y_list, std::vector<TX*>& dx_list,
        std::vector<TW*>& dw_list, const std::vector<std::vector<int>>& x_shape_list, int f,
        const std::vector<std::vector<int>>& ksize_list, const std::vector<std::vector<int>>& stride_list,
        const std::vector<int>& pad, const std::vector<int>& dilation, int group,
        const std::vector<const float*>& x_maxlist, const std::vector<const float*>& w_maxlist,
        const std::vector<const float*>& scale_list, const std::vector<const float*>& batch_mean_list,
        const std::vector<const float*>& batch_inv_std_list, std::vector<float*>& dscale_list,
        std::vector<float*>& dbias_list, const Activation_t& act, float eps,
        bool is_nchw, bool has_shortcut, bool fused_add, void* reserve_space = nullptr);

// lstm&bilstm fusion
template<typename T, typename TW, typename TGEMM> DLL_EXPORT int lstm_train(Context* ctx, const T* x, const T* init_h,
        const T* init_c, const TW* w_x, const TW* w_h, const TW* b_x, const TW* b_h, T* y, T* last_h, T* last_c,
        int batch_size, int xdim, int hdim, int seq_len, const std::vector<int>& seq_len_tensor, bool is_reverse,
        const float* x_maxptr, const float* h_maxptr, const float* wx_maxptr, const float* wh_maxptr,
        T* i_f_g_o, T* c, const Activation_t& act, const Activation_t& recurrent_act);

template<typename T, typename TW, typename TGEMM> DLL_EXPORT int lstm_grad(Context* ctx, const T* x, const T* init_h,
        const T* init_c, const TW* w_x, const TW* w_h, const T* y, const T* y_grad, const T* last_h_grad,
        const T* last_c_grad, T* x_grad, T* init_h_grad, T* init_c_grad, TW* w_x_grad, TW* w_h_grad, TW* b_x_grad,
        TW* b_h_grad, int batch_size, int xdim, int hdim, int seq_len, const std::vector<int>& seq_len_tensor,
        const float* x_maxptr, const float* h_maxptr, const float* wx_maxptr, const float* wh_maxptr, const T* i_f_g_o,
        const T* c);

DLL_EXPORT int lstm_inference(Context* ctx,
        const float* x,             // embedding_1.tmp_0 [batch_size, seq_len, xdim]
        bool x_need_transpose,      // true
        const float* init_h,        // nullptr
        const float* init_c,        // nullptr
        const int64_t* x_seq_len,   // seq_len [batch_size]
        const float* wx,            // lstm_cell_0.w_0 [4 * hdim, xdim]
        const float* wx_maxptr,     // nullptr
        const float* wh,            // lstm_cell_0.w_1 [4 * hdim, hdim]
        const float* wh_maxptr,     // nullptr
        const float* bx,            // lstm_cell_0.b_0 [4 * hdim]
        const float* bh,            // lstm_cell_0.b_1 [4 * hdim]
        float* last_h,              // lstm_0.tmp_1 [1, batch_size, hdim]
        int batch_size,
        int seq_len,
        int xdim,
        int hdim);

template<typename T, typename TW, typename TGEMM = int16_t> DLL_EXPORT int lstm_inference(Context* ctx,
        int seq_len, int batch_size, int xdim, int hdim, bool is_reverse,
        const T* x, const T* init_h, const T* init_c,
        const int64_t* x_seq_len, // can be nullptr
        const TW* wx, const float* wx_maxptr,
        const TW* wh, const float* wh_maxptr,
        const TW* bx, const TW* bh,
        T* y, T* last_h, T* last_c);

template<typename T, typename TW, typename TGEMM> DLL_EXPORT int bilstm_inference(Context* ctx, const T* x,
        const T* init_h, const T* init_c, const TW* forward_w_x, const TW* forward_w_h, const TW* forward_b_x,
        const TW* forward_b_h, const TW* backward_w_x, const TW* backward_w_h, const TW* backward_b_x,
        const TW* backward_b_h, T* y, T* last_h, T* last_c, int batch_size, int xdim, int hdim, int seq_len,
        const std::vector<int>& seq_len_tensor, int layer_num, const float* x_maxptr, const float* h_maxptr,
        const float* forward_wx_maxptr, const float* forward_wh_maxptr, const float* backward_wx_maxptr,
        const float* backward_wh_maxptr, const Activation_t& act, const Activation_t& recurrent_act);

template<typename T, typename TW, typename TGEMM> DLL_EXPORT int bilstm_train(Context* ctx, const T* x, const T* init_h,
        const T* init_c, const TW* forward_w_x, const TW* forward_w_h, const TW* forward_b_x, const TW* forward_b_h,
        const TW* backward_w_x, const TW* backward_w_h, const TW* backward_b_x, const TW* backward_b_h, T* y, T* last_h,
        T* last_c, int batch_size, int xdim, int hdim, int seq_len, const std::vector<int>& seq_len_tensor,
        int layer_num, const float* x_maxptr, const float* h_maxptr, const float* forward_wx_maxptr,
        const float* forward_wh_maxptr, const float* backward_wx_maxptr, const float* backward_wh_maxptr, T* i_f_g_o,
        T* c, const Activation_t& act, const Activation_t& recurrent_act);

template<typename T, typename TW, typename TGEMM> DLL_EXPORT int bilstm_grad(Context* ctx, const T* x, const T* init_h,
        const T* init_c, const TW* forward_w_x, const TW* forward_w_h, const TW* backward_w_x, const TW* backward_w_h,
        const T* y, const T* y_grad, const T* last_h_grad, const T* last_c_grad, T* x_grad, T* init_h_grad,
        T* init_c_grad, TW* forward_w_x_grad, TW* forward_w_h_grad, TW* forward_b_x_grad, TW* forward_b_h_grad,
        TW* backward_w_x_grad, TW* backward_w_h_grad, TW* backward_b_x_grad, TW* backward_b_h_grad, int batch_size,
        int xdim, int hdim, int seq_len, const std::vector<int>& seq_len_tensor, int layer_num, const float* x_maxptr,
        const float* h_maxptr, const float* forward_wx_maxptr, const float* forward_wh_maxptr,
        const float* backward_wx_maxptr, const float* backward_wh_maxptr, const T* i_f_g_o, const T* c,
        const Activation_t& act, const Activation_t& recurrent_act);

// GRU
template<typename TX, typename TW, typename TY, typename TGEMM> DLL_EXPORT int gru_cell(Context* ctx,
        const TX* x, const TX* hidden_prev/*can be nullptr*/, const TW* weight_ih/*[3 * hdim, xdim]*/, const TW* weight_hh,
        TY* y,
        int batch, int seq_len, int xdim, int hdim,
        const float* x_maxptr, const float* hidden_prev_maxptr/*can be nullptr*/, const float* w_ih_maxptr,
        const float* w_hh_maxptr/*float[8]*/, float* y_maxptr/*can be nullptr*/,
        const float* bias_ih/*can be nullptr*/, const float* bias_hh/*can be nullptr*/,
        const Activation_t& act = Activation_t::TANH,
        const Activation_t& gate_act = Activation_t::SIGMOID,
        bool origin_mode = false,
        bool is_reverse = false,
        bool reset_after = false);

template <typename T, typename TW, typename TID, typename TGEMM> DLL_EXPORT
int grnn_cell(Context* ctx, const T* x, const T* h_prev, const std::vector<const TW*>& weight_x,
        const std::vector<const TW*>& weight_h, T* y, int cap_e, int cap_h, const VectorParam<TID>& lod,
        const float* x_maxptr, const float* h_prev_maxptr, const std::vector<const float*>& weight_x_max,
        const std::vector<const float*>& weight_h_max, float* y_maxptr);

template <typename T, typename TW, typename TID> DLL_EXPORT int match_matrix_tensor(Context* ctx, const T* x,
        const T* y,
        const TW* w, T* z, int dim_in, int dim_w, bool w_trans, VectorParam<TID> x_lod, VectorParam<TID> y_lod,
        const float* max_x, const float* max_y, const float* max_w, const Activation_t& act, T* x_dot_w = nullptr);

template<typename T, typename TID> DLL_EXPORT int sorted_softmax_topk(Context* ctx, const T* x, T* y, TID* index,
        const std::vector<int>& xshape, int axis, int k);

template<typename T> DLL_EXPORT int spacial_attention_pool2d(Context* ctx,
        const T* x, T* y, int n, int c, int h, int w, bool is_nchw);

template<typename T, typename TW, typename TGEMM> DLL_EXPORT int squeeze_excitation_block(Context* ctx,
        const T* x, const TW* weight1, const TW* weight2, T* y, int n, int c, int h, int w, int r,
        const float* w1_maxptr, const float* w2_maxptr, const float* bias1, const float* bias2, const T* branch,
        const Activation_t& excitation_act1, const Activation_t& excitation_act2, const Activation_t& block_act);

template<typename T> DLL_EXPORT int batch_norm_fusion(Context* ctx, const T* x, T* y, int n, int c, int h, int w,
        float eps, float momentum, const float* scale, const float* bias,
        float* batch_mean, float* batch_inv_std, float* global_mean, float* global_var, bool is_nchw,
        const T* branch, const Activation_t& act, void* reserve_space, int reserve_space_size);

template<typename T> DLL_EXPORT int batch_norm_fusion_gen_bitmask(Context* ctx, const T* x, T* y, int n, int c, int h,
        int w,
        float eps, float momentum, const float* scale, const float* bias,
        float* batch_mean, float* batch_inv_std, float* global_mean, float* global_var, bool is_nchw,
        const T* branch, const Activation_t& act, void* reserve_space, int reserve_space_size, unsigned int* bitmask);

template<typename T> DLL_EXPORT int batch_norm_grad_fusion(Context* ctx, const T* x, const T* y, const T* dy, T* dx,
        int n, int c, int h, int w, const float* scale, const float* batch_mean,
        const float* batch_inv_std, float* dscale, float* dbias, bool is_nchw,
        T* dbranch, const Activation_t& act, const void* reserve_space, int reserve_space_size,
        unsigned int* bitmask = nullptr);

template<typename TX, typename TY, typename TZ> DLL_EXPORT int add_activation_fusion(Context* ctx, const TX* x,
        const TY* y, TZ* z, int len, const float* max_x, const float* max_y, float* max_z, const Activation_t& act);

template<typename T> DLL_EXPORT int add_activation_grad_fusion(Context* ctx, const T* x, const T* y, const T* z,
        const T* dz,
        const T* inter_out, T* dinter_out, T* dy, T* dx, const Activation_t& act, int len);

template<typename T> DLL_EXPORT int add_layer_norm_fusion(Context* ctx, const T* x,
        const T* y, T* z, int m, int n, float eps, const float* scale, const float* bias);

template<typename T> DLL_EXPORT int slice_add_layer_norm_fusion(Context* ctx, const T* x,
        const T* y, T* z, int n, const VectorParam<int>& mseqs, float eps, const float* scale, const float* bias);

template<typename TT, typename TY, typename TID> DLL_EXPORT int multi_embedding_fusion(Context* ctx,
        const std::vector<const TT*>& table_list, TY* y, const std::vector<VectorParam<TID>>& idx_list,
        const std::vector<TID>& table_len_list, int dim, const std::vector<float>& scale_list,
        const std::vector<TID>& padding_idx_list);

template<typename T> DLL_EXPORT int yolo_box_coord(Context* ctx,
        const T* x, T* y,
        const std::vector<int>& x_shape,
        const float* grid,
        const float* stride,
        const float* anchor_grid,
        const std::vector<int>& grid_shape,
        const std::vector<int>& stride_shape,
        const std::vector<int>& anchor_grid_shape,
        float offset,
        float* x_max,
        float* y_max);


// z = option_trans0213(option_softmax(alpha * batch_matmul(option_trans0213(x), option_trans0213(y)) + mask) * beta)
// the meaning of mask is same as that in qk_attention
template<typename TX, typename TY, typename TZ, typename TGEMM> DLL_EXPORT int attention_fusion(
        Context* ctx, const TX* x, const TY* y, const float* mask, TZ* z,
        int batch_size, int head_num, int m, int n, int k,
        const std::vector<int>& mask_shape,
        bool x_trans, bool y_trans, float alpha, float beta,
        const float* max_x, const float* max_y, float* max_z,
        bool fuse_x_transpose0213, bool fuse_y_transpose0213, bool fuse_z_transpose0213,
        bool fuse_softmax);

template<typename TQ, typename TK, typename TQK, typename TGEMM, typename TZ = float> DLL_EXPORT int qk_attention(
        Context* ctx, const TQ* q, const TK* k, TQK* qk,
        const float* max_q, const float* max_k, float* max_qk, const BaseAttnParam& att, const TZ* mask = nullptr);

template<typename TQK, typename TV, typename TQKV, typename TGEMM, typename TZ = float> DLL_EXPORT int qk_v_attention(
        Context* ctx, const TQK* qk, const TV* v, TQKV* qkv,
        const float* max_qk, const float* max_v, float* max_qkv, const BaseAttnParam& att, const TZ* gate = nullptr);

template<typename TQ, typename TK, typename TV, typename TQKV, typename TGEMM, typename TZ = float> DLL_EXPORT int
qkv_attention(
        Context* ctx, const TQ* q, const TK* k, const TV* v, TQKV* qkv,
        const float* max_q, const float* max_k, const float* max_v, float* max_qkv, const BaseAttnParam& p,
        const TZ* mask = nullptr, const TZ* gate = nullptr);

template<typename T, typename TW, typename TGEMM> DLL_EXPORT int transformer_encoder(
        Context* ctx, const T* x, const std::vector<const TW*>& w_list, T* y,
        const std::vector<const float*>& max_xy_list, const std::vector<const float*>& max_w_list,
        const std::vector<const float*>& bias_list, const std::vector<const float*>& ln_scale_list,
        const std::vector<const float*>& ln_bias_list, const QKVAttnParam& att, const float* mask = nullptr);

template <typename T, typename TW, typename TGEMM> DLL_EXPORT int conv_knrm(
        Context* ctx, const T* query, const T* doc, const T* mask, T* y,
        const float* query_max, const float* doc_max, float* y_max,
        std::vector<const TW*> conv_weight_list, std::vector<const float*> conv_bias_list,
        std::vector<const float*> conv_maxw_list, std::vector<const TW*> fc_weight_list,
        std::vector<const float*> fc_bias_list, std::vector<const float*> fc_maxw_list,
        int batch, int hidden_dim, int query_len, int doc_len, int conv_num, int conv_filters,
        int out_dim, const VectorParam<float>& rbf_mu, const VectorParam<float>& rbf_sigma,
        const Activation_t& conv_act, const Activation_t& fc_act);

struct DLL_EXPORT FTUnifiedDecodingParam {
    const char* decoding_strategy{};
    int beam_size;
    int topk;
    float topp;
    int n_head;
    int size_per_head;
    int num_layer;
    int bos_id;
    int eos_id;
    long max_len;
    float beam_search_diversity_rate;
    int unk_id;
    int mask_id;
    float temperature;
    float len_penalty;
    bool normalize_before;
    bool pos_bias;
    Activation_t hidden_act{Activation_t::RELU};
    bool rel_len;
    bool early_stopping;
    int min_length;
    int vocab_size;
};

template<typename T, typename TW, typename TGEMM>
DLL_EXPORT int fasttransformer_unified_decoding(
        Context* ctx, const int* input_ids, const T* attn_mask, const int* mem_seq_len, const int* type_id,
        const int* decoder_type_id, const T* logits_mask, const T* word_embedding,
        std::vector<const float*>& self_ln_weight, std::vector<const float*>& self_ln_bias,
        std::vector<const TW*>& self_q_weight, std::vector<const float*>& self_q_maxquant,
        std::vector<const float*>& self_q_bias,
        std::vector<const TW*>& self_k_weight, std::vector<const float*>& self_k_maxquant,
        std::vector<const float*>& self_k_bias,
        std::vector<const TW*>& self_v_weight, std::vector<const float*>& self_v_maxquant,
        std::vector<const float*>& self_v_bias,
        std::vector<const TW*>& self_out_weight, std::vector<const float*>& self_out_maxquant,
        std::vector<const float*>& self_out_bias,
        std::vector<const float*>& ffn_ln_weight, std::vector<const float*>& ffn_ln_bias,
        std::vector<const TW*>& ffn_inter_weight, std::vector<const float*>& ffn_inter_maxquant,
        std::vector<const float*>& ffn_inter_bias,
        std::vector<const TW*>& ffn_out_weight, std::vector<const float*>& ffn_out_maxquant,
        std::vector<const float*>& ffn_out_bias,
        const float* decoder_ln_weight, const float* decoder_ln_bias,
        const TW* trans_weight, const float* trans_maxquant, const float* trans_bias,
        const float* lm_ln_weight, const float* lm_ln_bias,
        const TW* embedding_weight, const float* embedding_maxquant, const T* embedding_bias,
        const T* positional_embedding_weight, const T* type_embedding_weight,
        const int* role_id, const int* decoder_role_id, const T* role_embedding_table,
        const int* position_ids, const int* decoder_position_ids,
        int* output_ids, float* output_scores, int* parent_ids, int* sequence_length,
        const int batch_size, const int mem_length, const FTUnifiedDecodingParam& fudparam);

struct DLL_EXPORT DropoutAddLayernormParam {
    bool is_test;
    bool is_upscale_in_train;
    float dropout_prob;
    int seed_val;
    bool is_layernorm;
    float eps;
    int m;
    int n;
};

template<typename T>
DLL_EXPORT int dropout_add_layernorm(Context* ctx, const T* x, const T* bias,
        const float* ln_scale, const float* ln_bias,
        T* dropout, T* dropout_mask, T* y, float* mean, float* var, const DropoutAddLayernormParam& param);

template<typename T>
DLL_EXPORT int dropout_add_layernorm_grad(Context* ctx, const T* dropout,
        const T* dropmask, const T* dy, T* dx, T* d_dropout, const float* scale,
        const float* mean, const float* var, float* dscale, float* dbias, const DropoutAddLayernormParam& param);

// fused op: add bias (1, n) to matrix (m, n), then relu activation
template<typename T> DLL_EXPORT int add_bias_relu(Context* ctx, const T* x, const T* b, T* z, T* relu_out, int m,
        int n);

template <typename T>
DLL_EXPORT int multi_rbf_fusion(Context* ctx, const T* x, T* y, int xlen,
        const VectorParam<float>& mu, const VectorParam<float>& sigma);
}
}
}

#endif
