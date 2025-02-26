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
    : ITemporalGraph<GPUUsage>(directed, window, enable_weight_computation, timescale_bound)
    , ITemporalGraph<GPUUsage>::node_index(NodeEdgeIndexCUDA<GPUUsage>())
    , ITemporalGraph<GPUUsage>::edges(EdgeDataCUDA<GPUUsage>())
    , ITemporalGraph<GPUUsage>::node_mapping(NodeMappingCUDA<GPUUsage>())
{}

#ifdef HAS_CUDA

template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
