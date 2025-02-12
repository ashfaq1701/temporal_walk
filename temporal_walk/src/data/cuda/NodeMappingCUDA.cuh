#ifndef NODEMAPPING_CUDA_H
#define NODEMAPPING_CUDA_H

#include "../cpu/NodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA : public NodeMapping<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEMAPPING_CUDA_H
