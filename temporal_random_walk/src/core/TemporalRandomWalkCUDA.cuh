#ifndef TEMPORAL_RANDOM_WALK_CUDA_CUH
#define TEMPORAL_RANDOM_WALK_CUDA_CUH

#include "TemporalRandomWalk.cuh"
#include "../structs/enums.h"

template<GPUUsageMode GPUUsage>
class TemporalRandomWalkCUDA : public TemporalRandomWalk<GPUUsage> {
    static_assert(GPUUsage != GPUUsageMode::ON_CPU, "TemporalGraphThrust cannot be used with ON_CPU mode");

public:

    #ifdef HAS_CUDA

    #endif
};



#endif //TEMPORAL_RANDOM_WALK_CUDA_CUH
