// PolicyProvider.cuh
#ifndef POLICY_PROVIDER_H
#define POLICY_PROVIDER_H

#include "../core/structs.h"
#ifdef HAS_CUDA
#include "cuda_common/config.cuh"
#endif

template<GPUUsageMode GPUUsage>
class PolicyProvider {
public:
    virtual ~PolicyProvider() = default;

    #ifdef HAS_CUDA
    static constexpr auto get_policy() {
        if constexpr (GPUUsage == GPUUsageMode::ON_GPU) {
            return DEVICE_POLICY;
        } else {
            return HOST_POLICY;
        }
    }
    #endif
};



#endif //POLICY_PROVIDER_H
