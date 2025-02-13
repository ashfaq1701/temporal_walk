#ifndef TEMPORALGRAPH_CUDA_H
#define TEMPORALGRAPH_CUDA_H

#include "../cpu/TemporalGraph.cuh"
#include "../../cuda_common/config.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA final : public TemporalGraph<GPUUsage> {
public:
    // Inherit constructors from base class
    using TemporalGraph<GPUUsage>::TemporalGraph;

#ifdef HAS_CUDA

#endif
};

#endif //TEMPORALGRAPH_CUDA_H
