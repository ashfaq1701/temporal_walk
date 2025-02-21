#ifndef TEMPORALGRAPH_CUDA_H
#define TEMPORALGRAPH_CUDA_H

#include "../cpu/TemporalGraph.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCUDA final : public TemporalGraph<GPUUsage>, public PolicyProvider<GPUUsage> {

    static_assert(GPUUsage != GPUUsageMode::ON_CPU, "TemporalGraphCUDA cannot be used with ON_CPU mode");

public:
    // Inherit constructors from base class
    using TemporalGraph<GPUUsage>::TemporalGraph;

#ifdef HAS_CUDA
    void sort_and_merge_edges(size_t start_idx) override;
    void delete_old_edges() override;

    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const override;
    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const override;
    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const override;
    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const override;

    [[nodiscard]] std::tuple<int, int, int64_t> get_node_edge_at(int node_id,
                                                                 RandomPicker& picker,
                                                                 int64_t timestamp,
                                                                 bool forward) const override;

#endif
};

#endif //TEMPORALGRAPH_CUDA_H
