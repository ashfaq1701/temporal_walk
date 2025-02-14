#ifndef TEMPORALGRAPH_CUDA_H
#define TEMPORALGRAPH_CUDA_H

#include "../cpu/TemporalGraph.cuh"
#include "PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA final : public TemporalGraph<GPUUsage>, public PolicyProvider<GPUUsage> {
public:
    // Inherit constructors from base class
    using TemporalGraph<GPUUsage>::TemporalGraph;

#ifdef HAS_CUDA

#endif
};

#endif //TEMPORALGRAPH_CUDA_H
