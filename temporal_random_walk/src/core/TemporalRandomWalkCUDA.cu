#include "TemporalRandomWalkCUDA.cuh"

#include <stores/cuda/TemporalGraphCUDA.cuh>

template<GPUUsageMode GPUUsage>
HOST TemporalRandomWalkCUDA<GPUUsage>::TemporalRandomWalkCUDA(
    bool is_directed,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound):
    ITemporalRandomWalk<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound)
{
    this->temporal_graph = new TemporalGraphCUDA<GPUUsage>(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
}

template <GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCUDA<GPUUsage>::clear() {
    this->temporal_graph = new TemporalGraphCUDA<GPUUsage>(
        this->is_directed, this->max_time_capacity,
        this->enable_weight_computation, this->timescale_bound);
}

#ifdef HAS_CUDA
template class TemporalRandomWalkCUDA<GPUUsageMode::ON_GPU>;
#endif
