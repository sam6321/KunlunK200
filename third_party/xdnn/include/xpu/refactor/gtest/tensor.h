#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CORE_TENSOR_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CORE_TENSOR_H

#include <xpu/runtime.h>
#include <xpu/refactor/core/device.h>
#include <xpu/refactor/core/dtype.h>
#include <vector>
#include <string>
#include <cstring> // memset
#include <cstdlib> // malloc
#include <random> // random
#include <stdexcept>    // std::invalid_argument()
#include <new>          // std::bad_alloc()
#ifdef _MSC_VER
#else
#include <sys/time.h>
#endif
#include <time.h>

#define TENSOR_THROW(msg)                \
        throw std::invalid_argument(msg);
#define TENSOR_ALLOC_ASSERT_WITHRET(ret) \
        if (ret != 0) {                  \
            do_free();                   \
            throw std::bad_alloc();      \
        }
#define TENSOR_ALLOC_ASSERT_WITHPTR(ptr) \
        if (ptr == nullptr) {            \
            do_free();                   \
            throw std::bad_alloc();      \
        }

//gtest init varible
static int gtest_base_seed = -1;
static int gtest_early_stop_time = 0;
static int gtest_run_time = 0;

//gtest init function
void init_seed(const char* str_used_by_hash);
int xpu_seed();
int get_gtest_early_stop_time();
int get_gtest_cpu_runloop();
int get_gtest_xpu_runloop();
int get_gtest_ncluster_num();
int get_gtest_nsdnn_num();
int get_gtest_run_time();

bool get_gtest_print_all_diff();
bool get_gtest_print_perf();
bool get_gtest_print_tensor_seed();

namespace baidu {
namespace xpu {
namespace api {
// http://agroup.baidu.com/xdnn/md/article/4301359
class Tensor {
public:
    inline Tensor():
        _dev(kCPU),
        _dtype(kFLOAT32),
        _ptr(nullptr),
        _numel(1),
        _mem_kind(XPU_MEM_MAIN) {
        _shape.push_back(1);
        do_malloc();
    }
    inline Tensor(const Tensor& t);                                 // copy constructor
    inline Tensor(int len, Dtype in_dtype);                         // constructor using len. 1-D tensor created.
    inline Tensor(const std::vector<int>& shape, Dtype in_type);    // constructor using shape.
    inline Tensor(Tensor&& src);
    inline Tensor& operator=(const Tensor& t);
    inline ~Tensor() {
        do_free();
    }
    // get attributes
    Device dev() const {
        return _dev;
    }
    Dtype dtype() const {
        return _dtype;
    }
    template <typename T> T* data() {
        return reinterpret_cast<T*>(_ptr);
    }
    const std::vector<int>& shape() const {
        return _shape;
    }
    inline int numel() const {
        return _numel;
    }
    inline XPUMemoryKind mem_kind() const {
        return _mem_kind;
    }

    // get a modified tensor
    DLL_EXPORT Tensor reshape(const std::vector<int>& in_shape) const;
    DLL_EXPORT Tensor astype(Dtype in_dtype) const;
    DLL_EXPORT Tensor to(Device in_dev) const;
    DLL_EXPORT void to_l3(Device in_dev);
private:
    inline void do_malloc();            // malloc _ptr, fail to exit
    inline void do_memcpy(const Tensor& t);             // cpy t's data to this, fail to exit
    inline void relocate_mem(Device dev, XPUMemoryKind mem_kind);
    inline void do_free() noexcept;                     // free _ptr and _ptr = nullptr;
    bool is_nullptr() const {
        return _ptr ==
                nullptr;   // users will never use this function, because in their view, all tensors' ptr is not nullptr.
    }
    // nullptr tensor will cause exception and exit before passing to users
    Device _dev;                        // device type: CPU, XPU1, XPU2
    Dtype _dtype;                       // data type
    void* _ptr;                         // pointer to data.
    std::vector<int> _shape;            // tensor shape.
    int _numel;                         // tensor numel = shape[0] * shape[1] *...*shape[shape.size()-1]
    XPUMemoryKind _mem_kind;            // memory kind: MAIN_MEM/L3, if dev == cpu, then _mem_kind = XPU_MAIN_MEM

    // friend functions to use private ctor
    friend DLL_EXPORT Tensor tensor(int);
    friend DLL_EXPORT Tensor tensor(float);
    friend DLL_EXPORT Tensor tensor(const std::vector<int>& vec);
    friend DLL_EXPORT Tensor tensor(const std::vector<float>& vec);
};

void Tensor::do_malloc() {
    int64_t mem_sz = static_cast<int64_t>(numel()) * Dtype_size(dtype());
    int64_t max_memory_size = static_cast<int64_t>(4 * 1024 * 1024) * 1024;
    if (mem_sz <= 0 || (dev().type() != kCPU && mem_sz > max_memory_size)) {
        TENSOR_THROW("Tensor memory size should be smaller than 4GB");
    }
    do_free();                          // to avoid memory leakage
    if (dev().type() == kCPU) {
        _ptr = malloc(mem_sz);
    }
    if (dev().type() != kCPU) {
        xpu_malloc((void**)(&_ptr), mem_sz, _mem_kind);
    }
    TENSOR_ALLOC_ASSERT_WITHPTR(_ptr);
    return ;
}

void Tensor::do_memcpy(const Tensor& t) {
    if ((is_nullptr()) || (t.is_nullptr())) {
        TENSOR_THROW("Invalid tensor when do_memcpy...");
    }
    if (numel() != t.numel()) {
        TENSOR_THROW("Differnet numel() when do_memcpy...");
    }
    if (dtype() != t.dtype()) {
        TENSOR_THROW("Different data type when do_memcpy...");
    }
    int64_t mem_sz = static_cast<int64_t>(numel()) * Dtype_size(dtype());
    int ret = -1;
    int ret1 = -1;
    if ((dev().type() == kCPU) && (t.dev().type() == kCPU)) {
        std::memcpy(_ptr, t._ptr, mem_sz);                          // std::memcpy do exception management itself
        ret = 0;
    }
    if ((dev().type() == kCPU) && (t.dev().type() != kCPU)) {
        ret = xpu_memcpy((void*)_ptr, (const void*)t._ptr, mem_sz, XPUMemcpyKind::XPU_DEVICE_TO_HOST);
    }
    if ((dev().type() != kCPU) && (t.dev().type() == kCPU)) {
        ret = xpu_memcpy((void*)_ptr, (const void*)t._ptr, mem_sz, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
    }
    if ((dev().type() != kCPU) && (t.dev().type() != kCPU)) {
        // ret = xpu_memcpy((void*)_ptr, (const void*)t._ptr, mem_sz, XPUMemcpyKind::XPU_DEVICE_TO_DEVICE);
        // not supported by XPU2 now
        std::vector<char> middle(mem_sz);
        ret = xpu_memcpy((void*)(middle.data()), (const void*)t._ptr, mem_sz, XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        ret1 = xpu_memcpy((void*)_ptr, (void*)(middle.data()), mem_sz, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        TENSOR_ALLOC_ASSERT_WITHRET(ret1);
    }
    TENSOR_ALLOC_ASSERT_WITHRET(ret);
    return;
}

void Tensor::relocate_mem(Device dev, XPUMemoryKind mem_kind = XPU_MEM_MAIN) {
    // 1. same device and same mem_kind
    // if _dev == cpu, _mem_kind will always be same
    if (_dev == dev && _mem_kind == mem_kind) {
        return;                                 // do nothing
    }
    // others situations need do memory copy
    int64_t mem_sz = numel() * Dtype_size(dtype());
    int ret = -1;
    // middle data
    std::vector<char> middle(mem_sz);
    if (_dev == kCPU) {
        std::memcpy((void*)(middle.data()), (const void*)_ptr, mem_sz);
    } else {
        ret = xpu_memcpy((void*)(middle.data()), (const void*)_ptr, mem_sz, XPUMemcpyKind::XPU_DEVICE_TO_HOST);
        // check middle data copy success
        TENSOR_ALLOC_ASSERT_WITHRET(ret);
    }

    // other situations
    // first do free, then reset device status, finnally copy middle data to dst
    do_free();
    _dev = dev;
    _mem_kind = mem_kind;
    do_malloc();
    if (dev == kCPU) {
        std::memcpy((void*)_ptr, (void*)(middle.data()), mem_sz);
    } else {
        ret = xpu_memcpy((void*)_ptr, (void*)(middle.data()), mem_sz, XPUMemcpyKind::XPU_HOST_TO_DEVICE);
        // check relocation success
        TENSOR_ALLOC_ASSERT_WITHRET(ret);
    }
    return;
}
void Tensor::do_free() noexcept {
    if (_ptr != nullptr) {
        if (dev().type() == kCPU) {
            free(_ptr);
        }
        if (dev().type() != kCPU) {
            xpu_free(_ptr);
            xpu_wait();
        }
    }
    _ptr = nullptr;                         // to avoid double free
    return;
}

Tensor::Tensor(int len, Dtype in_dtype) {
    _ptr = nullptr;
    if (len <= 0) {
        TENSOR_THROW("Tensor numel should be positive.");
    }
    _dtype = in_dtype;
    _dev = {kCPU, 0};
    _shape.push_back(len);
    _numel = len;
    _mem_kind = XPU_MEM_MAIN;
    do_malloc();
}

Tensor::Tensor(const std::vector<int>& shape, Dtype in_type) {
    _ptr = nullptr;
    _dtype = in_type;
    _numel = 1;
    for (size_t i = 0; i < shape.size(); i++) {
        _numel = _numel * shape[i];
        if (shape[i] < 0) {
            TENSOR_THROW("Invalid shape size");
        }
    }
    if (_numel <= 0) {
        TENSOR_THROW("Tensor numel should be positive.");
    }
    _dev = {kCPU, 0};
    _shape = shape;
    _mem_kind = XPU_MEM_MAIN;
    do_malloc();
}
Tensor::Tensor(const Tensor& t) {
    _ptr = nullptr;
    // do not construct a tensor with another tensor with nulltpr.
    if (t.is_nullptr()) {
        TENSOR_THROW("Invalid Tensor");        // early exit, to avoid useless do_malloc() when t's ptr is nullptr
    }
    _dev = t.dev();
    _dtype = t.dtype();
    _shape = t.shape();
    _numel = t.numel();
    _mem_kind = t.mem_kind();       // temporary
    do_malloc();                    // if failed, throw exception and directly exit
    do_memcpy(t);
}

Tensor& Tensor::operator=(const Tensor& t) {
    if (this == &t) {
        return *this;
    }
    do_free();
    // do not assign a tensor with nullptr to another one.
    if (t.is_nullptr()) {
        TENSOR_THROW("Invalid Tensor");
    }
    _dev = t._dev;
    _dtype = t._dtype;              // will do data type transformation defaultly.
    _shape = t._shape;
    _numel = t._numel;              // for support tensor reuse.
    _mem_kind = t.mem_kind();       // temporary
    do_malloc();                    // if failed, throw exception and directly exit
    do_memcpy(t);                   // if do_memcpy() success, then numel is same naturaly.
    return *this;
}

Tensor::Tensor(Tensor&& src) {
    _dev = src.dev();
    _dtype = src.dtype();
    _shape = src.shape();
    _numel = src.numel();
    _mem_kind = src.mem_kind();
    _ptr = src._ptr;
    src._ptr = nullptr;
}

// Creation
Tensor tensor(int val);
Tensor tensor(float val);
Tensor tensor(const std::vector<int8_t>& vec);
Tensor tensor(const std::vector<int16_t>& vec);
Tensor tensor(const std::vector<int>& vec);
Tensor tensor(const std::vector<int64_t>& vec);
Tensor tensor(const std::vector<float16>& vec);
Tensor tensor(const std::vector<float>& vec);
DLL_EXPORT Tensor tensor(const std::string& fname, Dtype dtype);
DLL_EXPORT Tensor tensor(const std::string& fname);
DLL_EXPORT Tensor arange(int start, int end, int step);
DLL_EXPORT Tensor arange(float start, float end, float step);
DLL_EXPORT Tensor randint(int minval, int maxval, int size, int seed = xpu_seed());
DLL_EXPORT Tensor randfloat(float minval, float maxval, int size, int seed = xpu_seed());
DLL_EXPORT Tensor randint(int minval, int maxval, const std::vector<int>& shape, int seed = xpu_seed());
DLL_EXPORT Tensor randfloat(float minval, float maxval, const std::vector<int>& shape, int seed = xpu_seed());

DLL_EXPORT std::string to_string(const std::vector<int>& vec);
DLL_EXPORT int count_allclose(const Tensor& t0, const Tensor& t1, float rtol, float atol);
DLL_EXPORT bool abs_max_close(const Tensor& t0, const Tensor& t1, float rtol, float atol);
DLL_EXPORT int print_tensor_diff(const Tensor& t0, const Tensor& t1);
DLL_EXPORT int print_tensor(const Tensor& t0);
DLL_EXPORT int debug_level();

}
}
}
#endif
