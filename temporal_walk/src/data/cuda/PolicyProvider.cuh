// PolicyProvider.cuh
#ifndef POLICY_PROVIDER_H
#define POLICY_PROVIDER_H

#include "../../cuda_common/config.cuh"
#include <thrust/execution_policy.h>

template<GPUUsageMode GPUUsage>
class PolicyProvider {
public:
    virtual ~PolicyProvider() = default;

    static constexpr auto get_policy() {
        if constexpr (GPUUsage == GPUUsageMode::DATA_ON_GPU) {
            return thrust::device;
        } else {
            return thrust::host;
        }
    }
};

#endif //POLICY_PROVIDER_H
