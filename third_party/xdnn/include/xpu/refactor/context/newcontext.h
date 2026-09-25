#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CONTEXT_NEWCONTEXT_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CONTEXT_NEWCONTEXT_H
#include <cstdint>
#include <vector>
#include <string>
#include <xpu/runtime.h>
#include "xpu/dll_export.h"
#include "xpu/refactor/core/device.h"
#include "xpu/refactor/core/dtype.h"
#include "xpu/refactor/context/buffer_mgr.h"
#include "xpu/refactor/context/xpu_act_type.h"

namespace baidu {
namespace xpu {
namespace api {

class ContextImpl;
struct DLL_EXPORT Context {
private:
    static constexpr int _nsdnn_list[4] = {-1, 4, 6, 12};
    static constexpr int _ncluster_list[4] = {-1, 4, 8, 12};
    static constexpr int _max_ptr_size_list[4] = {1, 4, 6, 12};
    static constexpr int _default_gm_mb[4] = {0, 64, 128, 128};
public:
    Context(Device in_dev);
    ~Context();
    Device dev() const {
        return _dev;
    }
    int debug_level() const {
        return _debug_level;
    }
    int ncluster() const {
        return _ncluster;
    }
    int set_ncluster(int n) {
        int list_index = (int)(_dev.type());
        int max_ncluster = _ncluster_list[list_index];
        if (n >= 1 && n <= max_ncluster) {
            _ncluster = n;
            return 0;
        } else {
            return -1;
        }
    }
    int nsdnn() const {
        return _nsdnn;
    }
    int set_nsdnn(int n) {
        int list_index = (int)(_dev.type());
        int max_nsdnn = _nsdnn_list[list_index];
        if (n >= 1 && n <= max_nsdnn) {
            _nsdnn = n;
            return 0;
        } else {
            return -1;
        }
    }
    int max_ptr_size() const {
        return _max_ptr_size;
    }
    void set_stream(XPUStream s) {
        xpu_stream = s;
    }
    XPUStream get_stream() {
        return xpu_stream;
    }
    ContextImpl* impl;
    BufferMgr _gm_mgr;
    BufferMgr _l3_mgr;
private:
    Device _dev;
    int _debug_level;
    int _nsdnn;
    int _ncluster;
    int _max_ptr_size;
public:
    XPUStream xpu_stream = nullptr;
    // deprecated fields
    bool enable_multi_stream = false;
    int batch_split_type = -1;
    bool qkv_fusion = false;
};

DLL_EXPORT Context* create_context();
DLL_EXPORT void destroy_context(Context* ctx);

class DLL_EXPORT ctx_guard {
private:
    Context* _ctx;
    template<class T> friend class VectorParam;
    std::vector<void*> cpu_vec;
    template <typename T> T* alloc_cpu(int cnt) {
        void* ptr = malloc(sizeof(T) * cnt);
        if (ptr != nullptr) {
            cpu_vec.push_back(ptr);
        } else {
            return nullptr;
            //throw std::bad_alloc();
        }
        return (T*)(ptr);
    }
public:
    ctx_guard(const ctx_guard&) = delete;
    ctx_guard& operator=(const ctx_guard&) = delete;
    ctx_guard(Context* ctx) {
        _ctx = ctx;
        if (_ctx) {
            _ctx->_gm_mgr.save();
            _ctx->_l3_mgr.save();
        }
    }
    ~ctx_guard() {
        if (_ctx) {
            _ctx->_gm_mgr.restore(_ctx->xpu_stream);
            _ctx->_l3_mgr.restore(_ctx->xpu_stream);
        }
        for (auto ptr : cpu_vec) {
            free(ptr);
        }
    }
    template<typename T> T* alloc_l3(int cnt) {
        if (_ctx) {
            if (_ctx->dev().type() == api::kCPU) {
                return alloc_cpu<T>(cnt);
            }
            return static_cast<T*>(_ctx->_l3_mgr.alloc(_ctx->impl, sizeof(T) * cnt, _ctx->xpu_stream));
        }
        return nullptr;
    }
    template<typename T> T* alloc(int cnt) {
        if (_ctx) {
            if (_ctx->dev().type() == api::kCPU) {
                return alloc_cpu<T>(cnt);
            }
            return static_cast<T*>(_ctx->_gm_mgr.alloc(_ctx->impl, sizeof(T) * cnt, _ctx->xpu_stream));
        }
        return nullptr;
    }
    template<typename T> T* alloc_l3_or_gm(int cnt) {
        if (_ctx) {
            if (_ctx->dev().type() == api::kCPU) {
                return alloc_cpu<T>(cnt);
            }
            T* ptr = static_cast<T*>(_ctx->_l3_mgr.alloc(_ctx->impl, sizeof(T) * cnt, _ctx->xpu_stream));
            if (ptr != nullptr) {
                return ptr;
            } else {
                return static_cast<T*>(_ctx->_gm_mgr.alloc(_ctx->impl, sizeof(T) * cnt, _ctx->xpu_stream));
            }
        }
        return nullptr;
    }
};

}
}
}
#endif
