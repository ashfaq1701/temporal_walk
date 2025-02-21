#ifndef TEMPORALGRAPHCUDA_H
#define TEMPORALGRAPHCUDA_H

#include <data/thrust/TemporalGraphThrust.cuh>

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA : public TemporalGraphThrust<GPUUsage> {
    static_assert(GPUUsage != GPUUsageMode::ON_CPU, "TemporalGraphCUDA cannot be used with ON_CPU mode");

public:
    // Inherit constructors from base class
    using TemporalGraph<GPUUsage>::TemporalGraph;

    #ifdef HAS_CUDA

    #endif
};

#endif //TEMPORALGRAPHCUDA_H
