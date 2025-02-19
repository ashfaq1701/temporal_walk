#ifndef TEMPORALGRAPH_CUDA_H
#define TEMPORALGRAPH_CUDA_H

#include "../cpu/TemporalGraph.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA final : public TemporalGraph<GPUUsage>, public PolicyProvider<GPUUsage> {
public:
    // Inherit constructors from base class
    using TemporalGraph<GPUUsage>::TemporalGraph;

    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const override;
    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const override;
    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const override;
    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const override;

#ifdef HAS_CUDA

#endif
};

#endif //TEMPORALGRAPH_CUDA_H
