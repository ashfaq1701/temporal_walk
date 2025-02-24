#include "TemporalGraphCUDA.cuh"

#include "EdgeDataCUDA.cuh"
#include "NodeEdgeIndexCUDA.cuh"
#include "NodeMappingCUDA.cuh"

template<GPUUsageMode GPUUsage>
HOST TemporalGraphCUDA<GPUUsage>::TemporalGraphCUDA(
    const bool directed,
    const int64_t window,
    const bool enable_weight_computation,
    const double timescale_bound)
    : TemporalGraph<GPUUsage>::is_directed(directed)
    , TemporalGraph<GPUUsage>::time_window(window)
    , TemporalGraph<GPUUsage>::enable_weight_computation(enable_weight_computation)
    , TemporalGraph<GPUUsage>::timescale_bound(timescale_bound)
    , TemporalGraph<GPUUsage>::latest_timestamp(0)
    , TemporalGraph<GPUUsage>::node_index(NodeEdgeIndexCUDA<GPUUsage>())
    , TemporalGraph<GPUUsage>::edges(EdgeDataCUDA<GPUUsage>())
    , TemporalGraph<GPUUsage>::node_mapping(NodeMappingCUDA<GPUUsage>())
{}

#ifdef HAS_CUDA

template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
