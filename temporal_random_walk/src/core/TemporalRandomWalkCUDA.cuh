#ifndef TEMPORAL_RANDOM_WALK_CUDA_CUH
#define TEMPORAL_RANDOM_WALK_CUDA_CUH

#include "TemporalRandomWalk.cuh"

template<GPUUsageMode GPUUsage>
class TemporalRandomWalkCUDA : public TemporalRandomWalk<GPUUsage> {
    static_assert(GPUUsage != GPUUsageMode::ON_CPU, "TemporalGraphThrust cannot be used with ON_CPU mode");

public:
    // Inherit constructors from base class
    using TemporalRandomWalk<GPUUsage>::TemporalRandomWalk;

    #ifdef HAS_CUDA

    #endif
};



#endif //TEMPORAL_RANDOM_WALK_CUDA_CUH
