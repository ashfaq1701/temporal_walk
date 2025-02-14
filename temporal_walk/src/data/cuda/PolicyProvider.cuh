// PolicyProvider.cuh
#ifndef POLICY_PROVIDER_H
#define POLICY_PROVIDER_H

#ifdef HAS_CUDA
#include "../../config.cuh"
#endif

template<GPUUsageMode GPUUsage>
class PolicyProvider {
public:
    virtual ~PolicyProvider() = default;

    #ifdef HAS_CUDA
    static constexpr auto get_policy() {
        if constexpr (GPUUsage == GPUUsageMode::DATA_ON_GPU) {
            return DEVICE_POLICY;
        } else {
            return HOST_POLICY;
        }
    }
    #endif
};



#endif //POLICY_PROVIDER_H
