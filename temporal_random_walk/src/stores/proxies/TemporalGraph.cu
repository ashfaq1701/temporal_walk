#include "TemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
TemporalGraph<GPUUsage>::TemporalGraph(
    bool directed,
    int64_t window,
    bool enable_weight_computation,
    double timescale_bound)
        : temporal_graph(TemporalGraphType(directed, window, enable_weight_computation, timescale_bound)) {}

template class TemporalGraph<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalGraph<GPUUsageMode::ON_GPU>;
#endif
