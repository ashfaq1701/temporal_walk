#ifndef TEMPORALGRAPHCUDA_H
#define TEMPORALGRAPHCUDA_H

#include <data/interfaces/TemporalGraph.cuh>

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA : public TemporalGraph<GPUUsage> {
    static_assert(GPUUsage != GPUUsageMode::ON_CPU, "TemporalGraphCUDA cannot be used with ON_CPU mode");

public:
    ~TemporalGraphCUDA() override = default;

    DEVICE explicit TemporalGraphCUDA(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1);

    #ifdef HAS_CUDA

    #endif
};

#endif //TEMPORALGRAPHCUDA_H
