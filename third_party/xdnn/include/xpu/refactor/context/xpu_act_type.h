#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CONTEXT_XPU_ACT_TYPE_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CONTEXT_XPU_ACT_TYPE_H
#include <cstdint>
#include <string>
#include <tuple>
#include <map>
#include "xpu/dll_export.h"

namespace baidu {
namespace xpu {
namespace api {

struct DLL_EXPORT Activation_t {
public:
    enum act_enum {
        LINEAR = 0,
        RELU = 1,
        SIGMOID = 2,
        TANH = 3,
        GELU = 4,
        LEAKY_RELU = 5,
        EXP = 12,
        HARD_SWISH = 14,
        HARD_SIGMOID = 15,
        SWISH = 16,
        RELU6 = 17,
        APPROXIMATE_GELU = 18,
        TANHSOFT = 22,
    };
    act_enum type;
    union {
        float leaky_alpha;
        float hard_sigmoid_slope;
    };
    Activation_t(act_enum t): type(t), leaky_alpha(0.0f) {}
    Activation_t(act_enum t, float value) {
        type = t;
        if (t == LEAKY_RELU) {
            leaky_alpha = value;
        } else if (t == HARD_SIGMOID) {
            hard_sigmoid_slope = value;
        }
    }
    bool operator==(const Activation_t& other) const {
        if (type == other.type) {
            if (type == LEAKY_RELU) {
                return leaky_alpha == other.leaky_alpha;
            } else if (type == HARD_SIGMOID) {
                return hard_sigmoid_slope == other.hard_sigmoid_slope;
            }
            return true;
        } else {
            return false;
        }
    }
    std::string decode() const {
        int tmp = type;
        return std::to_string(tmp) + "," + std::to_string(leaky_alpha);
    }
    std::string to_string() const {
        std::string str_type = "";
        switch (type) {
        case Activation_t::LINEAR:
            str_type = "LINEAR";
            break;
        case Activation_t::RELU:
            str_type = "RELU";
            break;
        case Activation_t::SIGMOID:
            str_type = "SIGMOID";
            break;
        case Activation_t::TANH:
            str_type = "TANH";
            break;
        case Activation_t::GELU:
            str_type = "GELU";
            break;
        case Activation_t::LEAKY_RELU:
            str_type = "LEAKY_RELU";
            break;
        case Activation_t::EXP:
            str_type = "EXP";
            break;
        case Activation_t::HARD_SWISH:
            str_type = "HARD_SWISH";
            break;
        case Activation_t::HARD_SIGMOID:
            str_type = "HARD_SIGMOID";
            break;
        case Activation_t::SWISH:
            str_type = "SWISH";
            break;
        case Activation_t::RELU6:
            str_type = "RELU6";
            break;
        case Activation_t::APPROXIMATE_GELU:
            str_type = "APPROXIMATE_GELU";
            break;
        case Activation_t::TANHSOFT:
            str_type = "TANHSOFT";
            break;
        default:
            break;// do nothing
        }
        return "{api::Activation_t::" + str_type + ", " + std::to_string(leaky_alpha) + "}";
    }
};

}
}
}
#endif
