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
    : ITemporalGraph<GPUUsage>(directed, window, enable_weight_computation, timescale_bound) {
    this->node_index = new NodeEdgeIndexCUDA<GPUUsage>();
    this->edges = new EdgeDataCUDA<GPUUsage>();
    this->node_mapping = new NodeMappingCUDA<GPUUsage>();
}

#ifdef HAS_CUDA
template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
