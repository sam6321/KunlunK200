#ifndef BAIDU_XPU_API_INCLUDE_XPU_ENV_H
#define BAIDU_XPU_API_INCLUDE_XPU_ENV_H

#include <string>
#include <cstring>
#include <unordered_map>
#include <vector>
#include <cstdio>
#include <xpu/runtime.h>


/*
 * ------------ key ------------ -------------- value ---------------
 * XPUSIM_DEVICE_MODEL          | string  |  [KUNLUN1, KUNLUN2]
 * XPUAPI_DEBUG                 | int     |  [0, 1]
 * XPU_CONV_AUTOTUNE_FILE       | string  |  "AnyNameYouLike"
 * XPU_CONV_AUTOTUNE            | int     |  [1, 2, 3, 4, 5, ...]
 * XPUAPI_DEFAULT_SIZE          | int     |  [1, 2, 3, 4, 5, ...]
 * ----------------------------- ------------------------------------
 */

class XPUEnv {
public:
    static char* getenv(const char* name) {
        static XPUEnv xpu_env;
        return xpu_env.get_env(name);
    }

    XPUEnv(const XPUEnv&) = delete;
    XPUEnv& operator=(const XPUEnv&) = delete;
    ~XPUEnv() = default;

private:
    XPUEnv() : kl2_env("KUNLUN2") {
        init_env();
    }

    void init_env() {
        std::vector<const char*> env_names = {
            // please add the name of xpu environment variable here if necessary
            "XPUSIM_DEVICE_MODEL",
            "XPUAPI_DEBUG",
            "XPU_CONV_AUTOTUNE_FILE",
            "XPU_CONV_AUTOTUNE",
            "XPUAPI_DEFAULT_SIZE",
            "XPU_AUTOTUNE_WRITEBACK",
            "XDNN_LOG_FILE",
        };

        for (auto name : env_names) {
            std::string n(name);
            env_caches.emplace(n, std::getenv(name));
        }

#ifndef _MSC_VER
        // TODO: Remove All pre-refactor wrapper in Paddle-Lite
        auto dev_env = env_caches["XPUSIM_DEVICE_MODEL"];
        if (dev_env == NULL || std::strcmp(dev_env, "KUNLUN2") != 0) {
            int cur_dev_idx = 0;
            uint64_t cur_dev_attr = 0;
            int ret = xpu_current_device(&cur_dev_idx);
            if (ret != 0) {
                fprintf(stderr, "[INFO][XPUAPI][No XPU Device Found]");
            }
            ret = xpu_device_get_attr(&cur_dev_attr, XPUATTR_MODEL, cur_dev_idx);
            if (ret != 0) {
                fprintf(stderr, "[INFO][XPUAPI][Invalid Device Attr 'XPUATTR_MODEL']");
            }
            if (cur_dev_attr == R200) {
                env_caches["XPUSIM_DEVICE_MODEL"] = const_cast<char*>(kl2_env.c_str());
            }
        }
#endif
    }

    char* get_env(const char* name) {
        if (name == nullptr) {
            return nullptr;
        }

        std::string n(name);

        auto res = env_caches.find(n);
        if (res == env_caches.end()) {
            return nullptr;
        } else {
            return env_caches[n];
        }
    }

    std::unordered_map<std::string, char*> env_caches;
    const std::string kl2_env;
};

#endif // BAIDU_XPU_API_INCLUDE_XPU_ENV_H
