#ifndef NODEMAPPING_CUDA_H
#define NODEMAPPING_CUDA_H

#include "../cpu/NodeMapping.cuh"
#include "PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA final : public NodeMapping<GPUUsage>, public PolicyProvider<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEMAPPING_CUDA_H
