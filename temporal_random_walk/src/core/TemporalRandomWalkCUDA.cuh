#ifndef TEMPORAL_RANDOM_WALK_CUDA_CUH
#define TEMPORAL_RANDOM_WALK_CUDA_CUH

#include "ITemporalRandomWalk.cuh"
#include "../data/enums.h"

template<GPUUsageMode GPUUsage>
class TemporalRandomWalkCUDA : public ITemporalRandomWalk<GPUUsage> {
public:

    explicit HOST TemporalRandomWalkCUDA(
        bool is_directed,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND);

    HOST void clear() override;
};

#endif //TEMPORAL_RANDOM_WALK_CUDA_CUH
