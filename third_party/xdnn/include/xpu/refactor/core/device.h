#ifndef BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CORE_DEVICE_H
#define BAIDU_XPU_API_INCLUDE_XPU_REFACTOR_CORE_DEVICE_H
#include <string>

namespace baidu {
namespace xpu {
namespace api {
enum class DeviceType {
    CPU = 0,
    XPU1 = 1,
    XPU2 = 2,
    XPU3 = 3,
};
constexpr DeviceType kCPU = DeviceType::CPU;
constexpr DeviceType kXPU1 = DeviceType::XPU1;
constexpr DeviceType kXPU2 = DeviceType::XPU2;
constexpr DeviceType kXPU3 = DeviceType::XPU3;

struct Device {
    Device(DeviceType type = kCPU, int id = 0) : type_(type), id_(id) {
        if (type == kCPU) {
            id = 0;
        }
    }
    bool operator==(const Device& other) const noexcept {
        return this->type_ == other.type_ && this->id_ == other.id_;
    }
    bool operator!=(const Device& other) const noexcept {
        return !(*this == other);
    }
    DeviceType type() const noexcept {
        return type_;
    }
    int id() const noexcept {
        return id_;
    }
private:
    DeviceType type_;
    int id_;
};

inline std::string to_string(DeviceType dt) {
    std::string arr[4] = {
        std::string("kCPU"),
        std::string("kXPU1"),
        std::string("kXPU2"),
        std::string("kXPU3"),
    };
    return arr[static_cast<int>(dt)];
}

inline std::string to_string(Device dev) {
    return std::string("{") + api::to_string(dev.type()) + "," + std::to_string(dev.id()) + "}";
}

}
}
}
#endif
