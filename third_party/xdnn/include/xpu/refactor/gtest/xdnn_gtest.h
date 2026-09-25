#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_IMPL_XDNN_GTEST_H

#include "gtest/gtest.h"
#include "xpu/refactor/gtest/tensor.h"
#include "xpu/refactor/gtest/quant_tensor.h"
#include "xpu/refactor/impl/xdnn_time.h"
#include "xpu/refactor/gtest/gtest_util.h"
#include <map>
//#include <algorithm>
namespace xdnn = baidu::xpu::api;
namespace api = baidu::xpu::api;
static const std::string GM = "GM";
static const std::string L3 = "L3";
static const std::string NIL = "NULL";

#include <time.h>

#define GTEST_INIT(ctx)                                                                                \
    init_seed(__FUNCTION__);                                                                           \
    gtest_run_time = get_gtest_run_time();                                                             \
    gtest_early_stop_time = get_gtest_early_stop_time();                                               \
    int _gtest_ncluster_num = get_gtest_ncluster_num();                                                \
    int _gtest_nsdnn_num = get_gtest_nsdnn_num();                                                      \
    if (gtest_early_stop_time !=0 && gtest_run_time > gtest_early_stop_time) {                         \
        return;                                                                                        \
    }                                                                                                  \
    if (_gtest_ncluster_num > 0) {                                                                     \
        if (_gtest_ncluster_num <= 4 && (ctx)->dev().type() == api::kXPU1) {                           \
            (ctx)->set_ncluster(_gtest_ncluster_num);                                                  \
        } else if (_gtest_ncluster_num <= 8 && (ctx)->dev().type() == api::kXPU2) {                    \
            (ctx)->set_ncluster(_gtest_ncluster_num);                                                  \
        } else if (_gtest_ncluster_num <= 12 && (ctx)->dev().type() == api::kXPU3) {                   \
            (ctx)->set_ncluster(_gtest_ncluster_num);                                                  \
        } else {                                                                                       \
            std::cout << "The number of cluster is wrong!" << std::endl;                               \
            return;                                                                                    \
        }                                                                                              \
    }                                                                                                  \
    if (_gtest_nsdnn_num > 0) {                                                                        \
        if (_gtest_nsdnn_num <= 4 && (ctx)->dev().type() == api::kXPU1) {                              \
                (ctx)->set_nsdnn(_gtest_nsdnn_num);                                                    \
            } else if (_gtest_nsdnn_num <= 6 && (ctx)->dev().type() == api::kXPU2) {                   \
                (ctx)->set_nsdnn(_gtest_nsdnn_num);                                                    \
            } else if (_gtest_nsdnn_num <= 12 && (ctx)->dev().type() == api::kXPU3) {                  \
                (ctx)->set_nsdnn(_gtest_nsdnn_num);                                                    \
            } else {                                                                                   \
                std::cout << "The number of sdnn is wrong!" << std::endl;                              \
                return;                                                                                \
            }                                                                                          \
    }                                                                                                  \

#define GTEST_XPU_SET_L3(ctx, l3size)                                                                  \
    void* __internal_l3ptr = nullptr;                                                                  \
    if (l3size > 0) {                                                                                  \
        xpu_malloc((void**)&__internal_l3ptr, l3size, XPU_MEM_L3);                                     \
        ASSERT_NE(__internal_l3ptr, nullptr);                                                          \
        (ctx)->_l3_mgr.set(__internal_l3ptr, l3size);                                                  \
    }

#define GTEST_XPU_START(ctx)                                                                           \
    int _gtest_internal_runloop = get_gtest_xpu_runloop();                                             \
    timeval _gtest_internal_time_start;                                                                \
    gettimeofday(&_gtest_internal_time_start, NULL);                                                   \
    for (int _gtest_internal_iter = 0; _gtest_internal_iter < _gtest_internal_runloop; _gtest_internal_iter++) {

#define GTEST_XPU_END(ctx)                                                                             \
    }                                                                                                  \
    xpu_wait((ctx)->xpu_stream);                                                                       \
    timeval _gtest_internal_time_end;                                                                  \
    gettimeofday(&_gtest_internal_time_end, NULL);                                                     \
    uint32_t _gtest_internal_time_diff = 1000000 * (_gtest_internal_time_end.tv_sec - _gtest_internal_time_start.tv_sec); \
    _gtest_internal_time_diff += _gtest_internal_time_end.tv_usec - _gtest_internal_time_start.tv_usec;\
    _gtest_internal_time_diff = _gtest_internal_time_diff / _gtest_internal_runloop;                   \
    if (get_gtest_print_perf()) {                                                                      \
        std::cout << "avg-time: " << _gtest_internal_time_diff << " us" << std::endl;                  \
    }

#define GTEST_XPU_END_DMA_FMT(ctx, total_rw, fmt, arg...)                                              \
    }                                                                                                  \
    xpu_wait((ctx)->xpu_stream);                                                                       \
    timeval _gtest_internal_time_end;                                                                  \
    gettimeofday(&_gtest_internal_time_end, NULL);                                                     \
    uint32_t _gtest_internal_time_diff = 1000000 * (_gtest_internal_time_end.tv_sec - _gtest_internal_time_start.tv_sec); \
    _gtest_internal_time_diff += _gtest_internal_time_end.tv_usec - _gtest_internal_time_start.tv_usec;\
    float _gtest_internal_us = _gtest_internal_time_diff / (float)_gtest_internal_runloop;             \
    float _gtest_internal_gbps = (total_rw) / 1.024 / 1.024 / 1024 *                                   \
            _gtest_internal_runloop / _gtest_internal_time_diff;                                       \
    if (get_gtest_print_perf()) {                                                                      \
        printf(fmt, ## arg);                                                                           \
        printf(": %.1f us, avg-dma = %f GB/s\n", _gtest_internal_us, _gtest_internal_gbps);            \
    }

#define GTEST_XPU_END_PROF(ctx, dmarw, macops, ewops)                                                  \
    }                                                                                                  \
    xpu_wait((ctx)->xpu_stream);                                                                       \
    timeval _gtest_internal_time_end;                                                                  \
    gettimeofday(&_gtest_internal_time_end, NULL);                                                     \
    uint32_t _gtest_internal_time_diff = 1000000 * (_gtest_internal_time_end.tv_sec - _gtest_internal_time_start.tv_sec); \
    _gtest_internal_time_diff += _gtest_internal_time_end.tv_usec - _gtest_internal_time_start.tv_usec;\
    _gtest_internal_time_diff = _gtest_internal_time_diff / _gtest_internal_runloop;                   \
    float _gtest_internal_gbps = (dmarw) / 1.024 / 1.024 / 1024 / _gtest_internal_time_diff;           \
    float _gtest_internal_mac_tops = (macops) / 1024.f / 1024.f / _gtest_internal_time_diff;           \
    float _gtest_internal_ew_tops = (ewops) / 1300.0 / _gtest_internal_time_diff;                      \
    if (get_gtest_print_perf()) {                                                                      \
        std::cout << "avg-dma: " << _gtest_internal_gbps << " GB/s; " ;                                \
        std::cout << "avg-mac: " << _gtest_internal_mac_tops << " TOPS; ";                             \
        std::cout << "avg-ew: " << _gtest_internal_ew_tops << " /cycle " << std::endl;                 \
    }

#define GTEST_XPU_END_PROF_FMT(ctx, dmarw, macops, ewops, fmt, arg...)                                 \
    }                                                                                                  \
    xpu_wait((ctx)->xpu_stream);                                                                       \
    timeval _gtest_internal_time_end;                                                                  \
    gettimeofday(&_gtest_internal_time_end, NULL);                                                     \
    uint32_t _gtest_internal_time_diff = 1000000 * (_gtest_internal_time_end.tv_sec - _gtest_internal_time_start.tv_sec); \
    _gtest_internal_time_diff += _gtest_internal_time_end.tv_usec - _gtest_internal_time_start.tv_usec;\
    _gtest_internal_time_diff = _gtest_internal_time_diff / _gtest_internal_runloop;                   \
    float _gtest_internal_gbps = (dmarw) / 1.024 / 1.024 / 1024 / _gtest_internal_time_diff;           \
    float _gtest_internal_mac_tops = (macops) / 1000000.0 / _gtest_internal_time_diff;                 \
    float _gtest_internal_ew_tops = (ewops) / 1300.0 / _gtest_internal_time_diff;                      \
    if (get_gtest_print_perf()) {                                                                      \
        printf(fmt, ## arg);                                                                           \
        printf(": avg-dma = %f GB/s; ", _gtest_internal_gbps);                                         \
        printf(": avg-mac = %f TOPS; ", _gtest_internal_mac_tops);                                     \
        printf(": avg-ew = %f /cycle\n", _gtest_internal_ew_tops);                                     \
    }

#define GTEST_CPU_START(ctx)                                                                           \
    int _gtest_internal_cpu = get_gtest_cpu_runloop();                                                 \
    if (_gtest_internal_cpu == 1) {                                                                    \

#define GTEST_CPU_END(ctx)                                                                             \
    }


#define GTEST_DEFINE_PTR(TYPE, str, cpu_tensor, xpu_tensor, cpu_ptr_name, xpu_ptr_name)                \
    auto xpu_tensor = cpu_tensor.to(dev);                                                              \
    TYPE* cpu_ptr_name = cpu_tensor.template data<TYPE>();                                             \
    TYPE* xpu_ptr_name = xpu_tensor.template data<TYPE>();                                             \
    if (str == "NULL") {                                                                               \
        cpu_ptr_name = nullptr;                                                                        \
        xpu_ptr_name = nullptr;                                                                        \
    }                                                                                                  \
    if (str[0] == 'L' && str[1] == '3') {                                                              \
        xpu_tensor.to_l3(dev);                                                                         \
        xpu_ptr_name = xpu_tensor.template data<TYPE>();                                               \
    }

#define GTEST_DEFINE_QUANT_PTR(TYPE, str, max_str, cpu_tensor, xpu_tensor, cpu_ptr_name,               \
        xpu_ptr_name, cpu_maxptr_name, xpu_maxptr_name)                                                \
    auto xpu_tensor = cpu_tensor.to(dev);                                                              \
    TYPE* cpu_ptr_name = cpu_tensor.val().data<TYPE>();                                                \
    TYPE* xpu_ptr_name = xpu_tensor.val().data<TYPE>();                                                \
    if (str == "NULL") {                                                                               \
        cpu_ptr_name = nullptr;                                                                        \
        xpu_ptr_name = nullptr;                                                                        \
    }                                                                                                  \
    if (str[0] == 'L' && str[1] == '3') {                                                              \
        xpu_tensor.to_l3(dev);                                                                         \
        xpu_ptr_name = xpu_tensor.val().data<TYPE>();                                                  \
    }                                                                                                  \
    float* cpu_maxptr_name = cpu_tensor.max().data<float>();                                           \
    float* xpu_maxptr_name = xpu_tensor.max().data<float>();                                           \
    if (max_str == "NULL") {                                                                           \
        cpu_maxptr_name = nullptr;                                                                     \
        xpu_maxptr_name = nullptr;                                                                     \
    }

#define GTEST_DEFINE_PTR_IN_LIST(TYPE, str, cpu_tensor, xpu_tensor, cpu_ptr,                           \
        xpu_ptr, start, end, len)                                                                      \
    if (str == "NULL") {                                                                               \
        cpu_ptr = nullptr;                                                                             \
        xpu_ptr = nullptr;                                                                             \
    } else {                                                                                           \
        cpu_tensor = api::randfloat(start, end, len).astype(api::CPPTypeToDtype<TYPE>());              \
        xpu_tensor = cpu_tensor.to(dev);                                                               \
        cpu_ptr = cpu_tensor.template data<TYPE>();                                                    \
        xpu_ptr = xpu_tensor.template data<TYPE>();                                                    \
        if (str[0] == 'L' && str[1] == '3') {                                                          \
            xpu_tensor.to_l3(dev);                                                                     \
            xpu_ptr = xpu_tensor.template data<TYPE>();                                                \
        }                                                                                              \
    }

#define GTEST_DEFINE_PTR_LIST(TYPE, PTR_TYPE, str_list, cpu_tensor_list, xpu_tensor_list,              \
        cpu_ptr_list, xpu_ptr_list, start, end, len_list)                                              \
    std::vector<api::Tensor> cpu_tensor_list(str_list.size());                                         \
    std::vector<api::Tensor> xpu_tensor_list(str_list.size());                                         \
    std::vector<PTR_TYPE> cpu_ptr_list(str_list.size(), nullptr);                                      \
    std::vector<PTR_TYPE> xpu_ptr_list(str_list.size(), nullptr);                                      \
    for (int i = 0; i < str_list.size(); i++) {                                                        \
        GTEST_DEFINE_PTR_IN_LIST(TYPE, str_list[i], cpu_tensor_list[i], xpu_tensor_list[i],            \
                cpu_ptr_list[i], xpu_ptr_list[i], start, end, len_list[i]);                            \
    }

#define GTEST_DEFINE_QUANT_PTR_IN_LIST(TYPE, str, max_str, cpu_tensor, xpu_tensor, cpu_ptr,            \
        xpu_ptr, cpu_maxptr, xpu_maxptr, start, end, len)                                              \
    if (str == "NULL") {                                                                               \
        cpu_ptr = nullptr;                                                                             \
        xpu_ptr = nullptr;                                                                             \
        cpu_maxptr = nullptr;                                                                          \
        xpu_maxptr = nullptr;                                                                          \
    } else {                                                                                           \
        cpu_tensor = Quant(api::randfloat(start, end, len), api::CPPTypeToDtype<TYPE>());              \
        xpu_tensor = cpu_tensor.to(dev);                                                               \
        cpu_ptr = cpu_tensor.val().data<TYPE>();                                                       \
        xpu_ptr = xpu_tensor.val().data<TYPE>();                                                       \
        if (str[0] == 'L' && str[1] == '3') {                                                          \
            xpu_tensor.to_l3(dev);                                                                     \
            xpu_ptr = xpu_tensor.val().data<TYPE>();                                                   \
        }                                                                                              \
        cpu_maxptr = cpu_tensor.max().data<float>();                                                   \
        xpu_maxptr = xpu_tensor.max().data<float>();                                                   \
        if (max_str == "NULL") {                                                                       \
            cpu_maxptr = nullptr;                                                                      \
            xpu_maxptr = nullptr;                                                                      \
        }                                                                                              \
    }

#define GTEST_DEFINE_QUANT_PTR_LIST(TYPE, PTR_TYPE, MAX_PTR_TYPE, str_list, max_str_list,              \
        cpu_tensor_list, xpu_tensor_list, cpu_ptr_list, xpu_ptr_list, cpu_maxptr_list,                 \
        xpu_maxptr_list, start, end, len_list)                                                         \
    std::vector<api::QuantTensor> cpu_tensor_list(str_list.size(), Quant(api::Tensor(), api::kINT16)); \
    std::vector<api::QuantTensor> xpu_tensor_list(str_list.size(), Quant(api::Tensor(), api::kINT16)); \
    std::vector<PTR_TYPE> cpu_ptr_list(str_list.size(), nullptr);                                      \
    std::vector<PTR_TYPE> xpu_ptr_list(str_list.size(), nullptr);                                      \
    std::vector<MAX_PTR_TYPE> cpu_maxptr_list(str_list.size(), nullptr);                               \
    std::vector<MAX_PTR_TYPE> xpu_maxptr_list(str_list.size(), nullptr);                               \
    for (int i = 0; i < str_list.size(); i++) {                                                        \
        GTEST_DEFINE_QUANT_PTR_IN_LIST(TYPE, str_list[i], max_str_list[i], cpu_tensor_list[i],         \
                xpu_tensor_list[i], cpu_ptr_list[i], xpu_ptr_list[i], cpu_maxptr_list[i],              \
                xpu_maxptr_list[i], start, end, len_list[i]);                                          \
    }

namespace baidu {
namespace xpu {
namespace api {
class TensorPosMapping {
public:
    TensorPosMapping() {};
    std::map<std::string, void*> map_cpu;
    std::map<std::string, void*> map_xpu;
    template <typename TYPE>
    int add(bool is_write, api::DeviceType dev, std::string str,
            api::Tensor* xpu_tensor_ptr, TYPE** cpu_ptr_ptr, TYPE** xpu_ptr_ptr) {
        if (map_cpu.find(str) != map_cpu.end()) { // already-exist
            if (is_write) {// write can only be first tensor
                return 2;
            }
            *cpu_ptr_ptr = (TYPE*)map_cpu[str];
            *xpu_ptr_ptr = (TYPE*)map_xpu[str];
            return 0;
        }
        // NULL-cases
        if (str == "NULL") {
            *cpu_ptr_ptr = nullptr;
            *xpu_ptr_ptr = nullptr;
            return 0;
        }
        if (str.size() < 2) {
            return 1; // wrong-name
        }
        bool is_gm = (str[0] == 'G' && str[1] == 'M');
        bool is_l3 = (str[0] == 'L' && str[1] == '3');
        if (is_l3) {
            xpu_tensor_ptr->to_l3(dev);
            *xpu_ptr_ptr = xpu_tensor_ptr->data<TYPE>();
        }
        if (is_gm || is_l3) {
            if (str.size() > 2) { // set map
                map_cpu[str] = (void*)(*cpu_ptr_ptr);
                map_xpu[str] = (void*)(*xpu_ptr_ptr);
            }
            return 0;
        } else {
            return 1; // wrong-name
        }
    }

    template <typename TYPE> int add(bool is_write, api::DeviceType dev, std::string str,
            api::QuantTensor* xpu_tensor_ptr, TYPE** cpu_ptr_ptr, TYPE** xpu_ptr_ptr) {
        if (map_cpu.find(str) != map_cpu.end()) { // already-exist
            if (is_write) {// write can only be first tensor
                return 2;
            }
            *cpu_ptr_ptr = (TYPE*)map_cpu[str];
            *xpu_ptr_ptr = (TYPE*)map_xpu[str];
            return 0;
        }
        // NULL-cases
        if (str == "NULL") {
            *cpu_ptr_ptr = nullptr;
            *xpu_ptr_ptr = nullptr;
            return 0;
        }
        if (str.size() < 2) {
            return 1; // wrong-name
        }
        bool is_gm = (str[0] == 'G' && str[1] == 'M');
        bool is_l3 = (str[0] == 'L' && str[1] == '3');
        if (is_l3) {
            xpu_tensor_ptr->to_l3(dev);
            *xpu_ptr_ptr = xpu_tensor_ptr->val().data<TYPE>();
        }
        if (is_gm || is_l3) {
            if (str.size() > 2) { // set map
                map_cpu[str] = (void*)(*cpu_ptr_ptr);
                map_xpu[str] = (void*)(*xpu_ptr_ptr);
            }
            return 0;
        } else {
            return 1; // wrong-name
        }
    }
};
}
}
}

#define GTEST_REUSE_PTR_DEFINE()                                                                       \
    api::TensorPosMapping gtest_tensor_pos_mapping;

#define GTEST_DEFINE_PTR_RO(TYPE, str, cpu_tensor, xpu_tensor, cpu_ptr_name, xpu_ptr_name)             \
    auto xpu_tensor = cpu_tensor.to(dev);                                                              \
    TYPE* cpu_ptr_name = cpu_tensor.data<TYPE>();                                                      \
    TYPE* xpu_ptr_name = xpu_tensor.data<TYPE>();                                                      \
    ASSERT_EQ(0,  gtest_tensor_pos_mapping.add<TYPE>(false, dev, str, &xpu_tensor, &cpu_ptr_name, &xpu_ptr_name));

#define GTEST_DEFINE_PTR_WO(TYPE, str, cpu_tensor, xpu_tensor, cpu_ptr_name, xpu_ptr_name)             \
    auto xpu_tensor = cpu_tensor.to(dev);                                                              \
    TYPE* cpu_ptr_name = cpu_tensor.data<TYPE>();                                                      \
    TYPE* xpu_ptr_name = xpu_tensor.data<TYPE>();                                                      \
    ASSERT_EQ(0,  gtest_tensor_pos_mapping.add<TYPE>(true, dev, str, &xpu_tensor, &cpu_ptr_name, &xpu_ptr_name));

#define GTEST_DEFINE_QUANT_PTR_WO(TYPE, str, max_str, cpu_tensor, xpu_tensor, cpu_ptr_name,            \
        xpu_ptr_name, cpu_maxptr_name, xpu_maxptr_name)                                                \
    auto xpu_tensor = cpu_tensor.to(dev);                                                              \
    TYPE* cpu_ptr_name = cpu_tensor.val().data<TYPE>();                                                \
    TYPE* xpu_ptr_name = xpu_tensor.val().data<TYPE>();                                                \
    ASSERT_EQ(0,  gtest_tensor_pos_mapping.add<TYPE>(true, dev, str, &xpu_tensor, &cpu_ptr_name, &xpu_ptr_name)); \
    float* cpu_maxptr_name = cpu_tensor.max().data<float>();                                           \
    float* xpu_maxptr_name = xpu_tensor.max().data<float>();                                           \
    if (max_str == "NULL") {                                                                           \
        cpu_maxptr_name = nullptr;                                                                     \
        xpu_maxptr_name = nullptr;                                                                     \
    }

#define GTEST_DEFINE_QUANT_PTR_RO(TYPE, str, max_str, cpu_tensor, xpu_tensor, cpu_ptr_name,            \
        xpu_ptr_name, cpu_maxptr_name, xpu_maxptr_name)                                                \
    auto xpu_tensor = cpu_tensor.to(dev);                                                              \
    TYPE* cpu_ptr_name = cpu_tensor.val().data<TYPE>();                                                \
    TYPE* xpu_ptr_name = xpu_tensor.val().data<TYPE>();                                                \
    ASSERT_EQ(0,  gtest_tensor_pos_mapping.add<TYPE>(false, dev, str, &xpu_tensor, &cpu_ptr_name, &xpu_ptr_name)); \
    float* cpu_maxptr_name = cpu_tensor.max().data<float>();                                           \
    float* xpu_maxptr_name = xpu_tensor.max().data<float>();                                           \
    if (max_str == "NULL") {                                                                           \
        cpu_maxptr_name = nullptr;                                                                     \
        xpu_maxptr_name = nullptr;                                                                     \
    }

#endif
