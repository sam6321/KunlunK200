#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CONTEXT_ATTENTION_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CONTEXT_ATTENTION_H
#include "xpu/dll_export.h"
#include "xpu/refactor/context/newcontext.h"
#include "xpu/xdnn_types.h"
#ifdef _MSC_VER
#include <algorithm>  // std::max, std::min
#endif

typedef enum {
    ATTN_NOVSL = 0x00,           // common situation for Q/K/V when novsl
    ATTN_LVSL = 0x10,            // common situation for Q/K/V when vsl
    ATTN_DVSL = 0x01,
} AttnMatrixVslType_t;

typedef enum {      // 描述Q/K/V矩阵的存储格式。 B = batch维， H = head维, L = seqlen， D = head_dim (size_per_head) 维
    ATTN_BHLD = 0x0001,
    ATTN_BLHD = 0x0010,
    ATTN_LBHD = 0x0011,
} AttnQKVLayout_t;

typedef enum {      // 描述attention probs(QK)的存储格式。 B = 第一个batch， H = 第二个batch， Q = qseqlen 维， K = kseqlen 维
    ATTN_BHQK = 0x0001,
    ATTN_BHKQ = 0x0010,
} AttnProbsLayout_t;

// if ATTN_ADD*, we use mac_acc to add, but not ew, decide by choosed kernel
typedef enum {
    ATTN_BYPASS = 0x0,                   // just do A * B, for qk and qk_v
    ATTN_SOFTMAX = 0x10,                 // for qk
    ATTN_ADD = 0x100,
    ATTN_ADD_SOFTMAX = 0x110,            // for qk
    ATTN_HADAMARD = 0x200,               // for qk_v
    // do not consider these two for now
    ATTN_HADAMARD_SOFTMAX = 0x210,       // for qk
    ATTN_HADAMARD_ADD_SOFTMAX = 0x310,   // for qk, can not use mac to add, so we need a new kernel,
    // and it will need a new param in wrapper level
} AttnPostType_t;

// some attention variants we will support
typedef enum {
    BASE_ATTENTION = 1,
    QKV_ATTENTION = 2,              // wrapper for old implementation
    DIF_SEQ_ATTENTION = 4,          // common situation for Q's seqlen differs from K's seqlen
    DECODE_ATTENTION = 8,           // common situation for decoder, qslice, cal step but not max_seqlen. only novsl
    USER_DEF_ATTENTION = 16,        // user can define an attention variant by himself (only when our kernel support)
} AttnType_t;

typedef enum {
    ATTN_WHOLE_BATCH = 1,
    ATTN_PER_BATCH = 2,                  // in kl2, we use local quant to suport this
} AttnMacMaxPtrType_t;

typedef enum {
    ATTN_NO_LIMITS = 0,
    ATTN_MID_SEQ = 1,
    ATTN_MIN_SEQ = 2,

    ATTN_OLD_KERNEL = 10,
} AttnKernel_t;

#pragma pack (4)
struct DLL_EXPORT AttnDesc {
    AttnMatrixVslType_t q_vsl_type;             // q's vsl type.
    AttnMatrixVslType_t kv_vsl_type;            // k/v 's vsl type. k/v's type may differ from q, for example when qslice
    AttnQKVLayout_t qkv_layout;                 // q/k/v's layout must be same, context's layout will keep same with q/k/v
    AttnProbsLayout_t
    attn_probs_layout;        // check me, only ATTN_BHQK or ATTN_BHKQ, indicates whether do transpose on attn probs

    int batch;
    int head_num;
    int head_dim;                       // size_per_head

    int max_q_seq;                      // max seqlen of q.
    int max_kv_seq;                     // max seqlen of k/v

    int valid_qseq;                     // valid q's seqlen when cal q * k
    int valid_kvseq;                    // valid k/v's seqlen when cal q * k

    int* qlod;                          // lod info of q when q is vsl
    int* kvlod;                         // lod info of k/v when k/v is vsl

    float alpha;
    int ld_factor;                      // ld  = last_dimention * ld_factor

    AttnPostType_t qk_post_type;        // to support more flexible config, postprocess after q*k
    AttnPostType_t qk_v_post_type;      // post process after qk * v

    // indicate which dimension is reduced. such as 0x1110, means {1,1,1,c}, 0x0000 means {b0, b1, max_r, max_c}
    int reduce_type;                    // mask / gate {1, 1, 1, seqlen}, 目前假定所有除q/k/v 的输入shape相同，如果要改，前16位给qk_v, 后16位给qk
    int ldz;                            // ld of mask/gate/..., to support asr mask(for compatibility with kl1)
    int vbatch;                         // valid batch，
    int* vbmap;                         // indicate which bath is valid. vbmap[i] = j, then it indicates i's valid batch is j.

    AttnKernel_t qk_kernel;
    AttnKernel_t qk_v_kernel;

public:
    // for common
    AttnDesc(AttnMatrixVslType_t _q_vsl_type, AttnMatrixVslType_t _kv_vsl_type, AttnQKVLayout_t _layout, int _batch,
            int _head_num, int _head_dim,
            int _max_q_seq, int _max_kv_seq, int _valid_qseq, int _valid_kvseq, int* _qlod = nullptr, int* _kvlod = nullptr,
            float _alpha = -1.0f, int _ld_factor = 1, AttnPostType_t _qk_post_type = ATTN_SOFTMAX,
            AttnPostType_t _qk_v_post_type = ATTN_BYPASS, int _reduce_type = 0x0000,
            int _ldz = -1, int _vbatch = -1, int* _vbmap = nullptr, AttnProbsLayout_t _attn_probs_layout = ATTN_BHQK) :
        q_vsl_type(_q_vsl_type), kv_vsl_type(_kv_vsl_type), qkv_layout(_layout), attn_probs_layout(_attn_probs_layout),
        batch(_batch), head_num(_head_num), head_dim(_head_dim),
        max_q_seq(_max_q_seq), max_kv_seq(_max_kv_seq), valid_qseq(_valid_qseq), valid_kvseq(_valid_kvseq), qlod(_qlod),
        kvlod(_kvlod),
        alpha(_alpha), ld_factor(_ld_factor), qk_post_type(_qk_post_type), qk_v_post_type(_qk_v_post_type),
        reduce_type(_reduce_type),
        ldz(_ldz), vbatch(_vbatch), vbmap(_vbmap) {
        // check all param is valid
        qk_kernel = AttnKernel_t::ATTN_NO_LIMITS;             // 默认兜底方案，防止可能出现的kernel限制等错误
        qk_v_kernel =
                AttnKernel_t::ATTN_NO_LIMITS;             // 默认兜底方案，防止可能出现的kernel限制等错误
    };
    AttnDesc() {};
};

namespace baidu {
namespace xpu {
namespace api {

class DLL_EXPORT BaseAttnParam {
public:
    const bool is_vsl;
    bool do_fc_qkv_fusion;
    int batch;
    int max_seqlen;
    const int head_num;
    const int head_dim;
    VectorParam<int> lod;
    const std::vector<int> zshape;              // normally, it means mask
    float alpha;
    AttnDesc attnDesc;
public:
    BaseAttnParam(bool _do_fc_qkv_fusion, int _batch, int _head_num, int _head_dim, VectorParam<int> _lod,
            const std::vector<int>& _zshape = {}):
        is_vsl(true), do_fc_qkv_fusion(_do_fc_qkv_fusion), batch(_batch),
        head_num(_head_num), head_dim(_head_dim), lod(_lod) {
        alpha = 1.0f / std::sqrt(1.0f * head_dim);
        max_seqlen = -1;
        if (lod.cpu[0] == 0 && batch >= 1) {
            max_seqlen = lod.cpu[1];
            int min_seqlen = lod.cpu[1];
            for (int i = 1; i < batch; i++) {
                int seqlen = lod.cpu[i + 1] - lod.cpu[i];
                max_seqlen = std::max<int>(max_seqlen, seqlen);
                min_seqlen = std::min<int>(min_seqlen, seqlen);
            }
            if (min_seqlen <= 0) {
                max_seqlen = -1;
            }
        }
    };
    BaseAttnParam(bool _do_fc_qkv_fusion, int _batch, int _seqlen, int _head_num, int _head_dim,
            const std::vector<int>& _zshape):
        is_vsl(false), do_fc_qkv_fusion(_do_fc_qkv_fusion), batch(_batch), max_seqlen(_seqlen),
        head_num(_head_num), head_dim(_head_dim), zshape(_zshape) {
        lod.len = 0;
        alpha = 1.0f / std::sqrt(1.0f * head_dim);
    };
    virtual ~BaseAttnParam() {};
    virtual void setAttnDesc() = 0;
    virtual AttnDesc genAttnDesc() const = 0;
    virtual const VectorParam<int> get_qlod() const  = 0;
    virtual const VectorParam<int> get_kvlod() const  = 0;
    virtual const VectorParam<int> get_vbmap() const  = 0;
    virtual Error_t selfcheck(Context* ctx) const = 0;
    virtual void attn_lens_info(int* qlen, int* klen, int* vlen, int* qklen, int* qkvlen) const = 0;
    virtual std::string to_string() const = 0;
    virtual AttnType_t get_attn_type() const = 0;
    virtual int get_mask_len() const = 0;
    virtual AttnMacMaxPtrType_t get_mac_max_ptr_type() const = 0;
};

enum class QuantType : int {
    QUANT_INT8,
    QUANT_INT16,
    NOT_QUANT
};

class DLL_EXPORT QKVAttnParam : public BaseAttnParam {
public:
    int mask_ld;
    const bool is_pre_norm;
    const Activation_t act;
    int last_slice_seq;
    int pad_seqlen;
    int hidden_dim;
    bool is_perchannel;
    std::vector<QuantType> quant_type_;
    int qkv_shape = 0;
    AttnMacMaxPtrType_t max_ptr_type;
    int ldz = -1;
    QKVAttnParam(VectorParam<int> _lod, int _head_num, int _head_dim, const Activation_t& _act = Activation_t::RELU,
            int _last_slice_seq = -1, bool _do_fc_qkv_fusion = false, int _pad_seqlen = -1, int _hidden_dim = -1,
            bool _is_pre_norm = false, bool _is_perchannel = false,
            int _qkv_shape = 0, const std::vector<int>& _zshape = {},
            AttnMacMaxPtrType_t _max_ptr_type = ATTN_WHOLE_BATCH, int _ldz = -1):
        BaseAttnParam(_do_fc_qkv_fusion, (_lod.len - 1), _head_num, _head_dim, _lod, _zshape),
        is_pre_norm(_is_pre_norm), act(_act), last_slice_seq(_last_slice_seq), pad_seqlen(_pad_seqlen),
        is_perchannel(_is_perchannel),
        max_ptr_type(_max_ptr_type), ldz(_ldz) {
        hidden_dim = (_hidden_dim == -1 ? head_num* head_dim : _hidden_dim);
        qkv_shape = _qkv_shape;
        if (zshape.size() == 4) {
            ldz = (ldz == -1 ? zshape[3] : ldz);
        }
    }

    // zshape = {a, b, c, d},
    //      a == 1 or batch
    //      b == 1 or head_num
    //      c == 1 or max_seqlen
    //      d == max_seqlen
    QKVAttnParam(int _batch, int _max_seqlen, int _head_num, int _head_dim, const std::vector<int>& _mask_shape,
            const Activation_t& _act = Activation_t::RELU, int _last_slice_seq = -1, bool _do_fc_qkv_fusion = false,
            int _hidden_dim = -1, bool _is_pre_norm = false, bool _is_perchannel = false, int _qkv_shape = 0,
            AttnMacMaxPtrType_t _max_ptr_type = ATTN_WHOLE_BATCH, int _ldz = -1):
        BaseAttnParam(_do_fc_qkv_fusion, _batch, _max_seqlen, _head_num, _head_dim, _mask_shape),
        is_pre_norm(_is_pre_norm), act(_act), last_slice_seq(_last_slice_seq), pad_seqlen(-1), is_perchannel(_is_perchannel),
        max_ptr_type(_max_ptr_type), ldz(_ldz) {
        hidden_dim = (_hidden_dim == -1 ? head_num* head_dim : _hidden_dim);
        qkv_shape = _qkv_shape;
        bool valid_mask_shape = (zshape.size() == 4);
        valid_mask_shape = valid_mask_shape && (zshape[0] == 1 || zshape[0] == batch);
        valid_mask_shape = valid_mask_shape && (zshape[1] == 1 || zshape[1] == head_num);
        valid_mask_shape = valid_mask_shape && (zshape[2] == 1 || zshape[2] == max_seqlen);
        valid_mask_shape = valid_mask_shape && (zshape[3] == max_seqlen);
        hidden_dim = ((_hidden_dim == -1) ? (head_num * head_dim) : _hidden_dim);
        if (valid_mask_shape) {
            mask_ld = zshape[1] * zshape[2] * zshape[3];
            if (zshape[0] == 1) {
                mask_ld = 0;
            }
        }
        if (zshape.size() == 4) {
            ldz = (ldz == -1 ? zshape[3] : ldz);
        }
    }
    ~QKVAttnParam() {};
    void setAttnDesc();
    AttnDesc genAttnDesc() const;
    const VectorParam<int> get_qlod() const {
        return lod;
    };
    const VectorParam<int> get_kvlod() const {
        return lod;
    };
    const VectorParam<int> get_vbmap() const {
        return {nullptr, 0, nullptr};
    };
    Error_t selfcheck(Context* ctx) const;
    void attn_lens_info(int* qlen, int* klen, int* vlen, int* qklen, int* qkvlen) const;
    std::string to_string() const;
    AttnType_t get_attn_type() const {
        return QKV_ATTENTION;
    };
    int get_mask_len() const {
        return zshape.size() == 4 ? zshape[0] * zshape[1] * zshape[2] * ldz : -1;
    };
    AttnMacMaxPtrType_t get_mac_max_ptr_type() const {
        return max_ptr_type;
    };
};

// in this attention, lod was used for both q and k/v
// lod[0:batch] : q lod
// lod[batch + 1: 2batch + 1]: klod
class DLL_EXPORT DifSeqAttnParam : public BaseAttnParam {
public:
    AttnType_t attn_type = DIF_SEQ_ATTENTION;
    int max_q_seq;
    int max_kv_seq;
    int do_softmax;                 // 1 do softmax
    int attn_probs_trans;           // 1 tranposed
    DifSeqAttnParam(VectorParam<int> _lod, int _head_num, int _head_dim, int _do_softmax = 1, int _attn_probs_trans = 0,
            const std::vector<int>& _zshape = {}):
        BaseAttnParam(false, (_lod.len / 2 - 1), _head_num, _head_dim, _lod, _zshape),
        do_softmax(_do_softmax), attn_probs_trans(_attn_probs_trans) {
        max_q_seq = -1;
        max_kv_seq = -1;
        if (lod.cpu[0] == 0 && batch >= 1) {
            max_q_seq = lod.cpu[1];
            int min_seqlen = lod.cpu[1];
            for (int i = 1; i < batch; i++) {
                int seqlen = lod.cpu[i + 1] - lod.cpu[i];
                max_q_seq = std::max<int>(max_q_seq, seqlen);
                min_seqlen = std::min<int>(min_seqlen, seqlen);
            }
            if (min_seqlen <= 0) {
                max_q_seq = -1;
            }
            min_seqlen = lod.cpu[batch + 2];;
            max_kv_seq = lod.cpu[batch + 2];
            for (int i = batch + 2; i < 2 * batch + 1; i++) {
                int seqlen = lod.cpu[i + 1] - lod.cpu[i];
                max_kv_seq = std::max<int>(max_kv_seq, seqlen);
                min_seqlen = std::min<int>(min_seqlen, seqlen);
            }
            if (min_seqlen <= 0) {
                max_kv_seq = -1;
            }
        }
        max_seqlen = std::max(max_q_seq, max_kv_seq);
    }
    DifSeqAttnParam(int _batch, int _max_q_seq, int _max_kv_seq, int _head_num, int _head_dim,
            const std::vector<int>& _zshape,
            int _do_softmax = 1, int _attn_probs_trans = 0):
        BaseAttnParam(false, _batch, std::max(_max_q_seq, _max_kv_seq), _head_num, _head_dim, _zshape),
        max_q_seq(_max_q_seq), max_kv_seq(_max_kv_seq), do_softmax(_do_softmax), attn_probs_trans(_attn_probs_trans) {
        lod.len = 0;
    }
    ~DifSeqAttnParam() {};
    void setAttnDesc();
    AttnDesc genAttnDesc() const;
    const VectorParam<int> get_qlod() const {
        VectorParam<int> ret{lod.cpu, lod.len / 2, lod.xpu};
        return ret;
    };
    const VectorParam<int> get_kvlod() const {
        VectorParam<int> ret{lod.cpu + lod.len / 2, lod.len / 2, lod.xpu + lod.len / 2};
        return ret;
    };
    const VectorParam<int> get_vbmap() const {
        return {nullptr, 0, nullptr};
    };
    Error_t selfcheck(Context* ctx) const;
    void attn_lens_info(int* qlen, int* klen, int* vlen, int* qklen, int* qkvlen) const;
    std::string to_string() const;
    AttnType_t get_attn_type() const {
        return DIF_SEQ_ATTENTION;
    };
    int get_mask_len() const {
        return zshape.size() == 4 ? zshape[0] * zshape[1] * zshape[2] * zshape[3] : -1;
    };
    AttnMacMaxPtrType_t get_mac_max_ptr_type() const {
        return ATTN_WHOLE_BATCH;
    };
};

// In decode, we suppose q is sliced
class DLL_EXPORT DecodeAttnParam : public BaseAttnParam{
public:
    AttnType_t attn_type = DECODE_ATTENTION;
    int step;
    const int qkv_shape;
    int vbatch;
    VectorParam<int> vb_map;

    // DecodeAttnParam(VectorParam<int> _lod, int _head_num, int _head_dim,  int _step = 1, int _qkv_shape = 0, 
    //         int _vbatch = -1, VectorParam<int> _vb_map = {nullptr, 0, nullptr}):
    //         BaseAttnParam(false, (_lod.len - 1), _head_num, _head_dim, _lod, {}),
    //         step(_step), qkv_shape(_qkv_shape), vbatch(_vbatch), vb_map(_vb_map){step = (step == -1 ? 1 : step);};
    DecodeAttnParam(int _batch, int _max_seqlen, int _head_num, int _head_dim, int _step = 1, int _qkv_shape = 0, 
            int _vbatch = -1, VectorParam<int> _vb_map = {nullptr, 0, nullptr}, const std::vector<int>& _rel_pos_bias_shape = {}):
            BaseAttnParam(false, _batch, _max_seqlen, _head_num, _head_dim, _rel_pos_bias_shape),
            step(_step), qkv_shape(_qkv_shape), vbatch(_vbatch), vb_map(_vb_map){
        vbatch = (vbatch == -1 ? batch : vbatch);
    };
    ~DecodeAttnParam(){};
    void setAttnDesc();
    AttnDesc genAttnDesc() const;
    const VectorParam<int> get_qlod() const {return lod;};
    const VectorParam<int> get_kvlod() const {return lod;};
    const VectorParam<int> get_vbmap() const {return vb_map;};
    Error_t selfcheck(Context* ctx) const;
    void attn_lens_info(int* qlen, int* klen, int* vlen, int* qklen, int* qkvlen) const;
    std::string to_string() const;
    AttnType_t get_attn_type() const {return DECODE_ATTENTION;};
    int get_mask_len() const {return zshape.size() == 4 ? zshape[0] * zshape[1] * zshape[2] * zshape[3] : -1;};
    AttnMacMaxPtrType_t get_mac_max_ptr_type() const{return ATTN_WHOLE_BATCH;};
};

#pragma pack () 

}
}
}
#endif