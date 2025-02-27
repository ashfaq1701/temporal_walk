#ifndef TEMPORALGRAPH_CPU_H
#define TEMPORALGRAPH_CPU_H

#include <cstdint>

#include "../../data/enums.h"
#include "../../random/RandomPicker.h"

#include "../interfaces/ITemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraphCPU : public ITemporalGraph<GPUUsage>
{
public:
    ~TemporalGraphCPU() override = default;

    HOST explicit TemporalGraphCPU(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1);

    HOST void sort_and_merge_edges_host(size_t start_idx) override;

    // Edge addition
    HOST void add_multiple_edges_host(const typename ITemporalGraph<GPUUsage>::EdgeVector& new_edges) override;

    HOST void update_temporal_weights_host() override;

    HOST void delete_old_edges_host() override;

    // Timestamp group counting
    [[nodiscard]] HOST size_t count_timestamps_less_than_host(int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_timestamps_greater_than_host(int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_node_timestamps_less_than_host(int node_id, int64_t timestamp) const override;
    [[nodiscard]] HOST size_t count_node_timestamps_greater_than_host(int node_id, int64_t timestamp) const override;

    // Edge selection
    [[nodiscard]] HOST Edge get_edge_at_host(RandomPicker<GPUUsage>* picker, int64_t timestamp = -1,
                                            bool forward = true) const override;

    [[nodiscard]] HOST Edge get_node_edge_at_host(int node_id,
                                                RandomPicker<GPUUsage>* picker,
                                                int64_t timestamp = -1,
                                                bool forward = true) const override;

    // Utility methods
    [[nodiscard]] HOST size_t get_total_edges_host() const override { return this->edges->size_host(); }
    [[nodiscard]] HOST size_t get_node_count_host() const override { return this->node_mapping->active_size_host(); }
    [[nodiscard]] HOST int64_t get_latest_timestamp_host() override { return this->latest_timestamp; }
    [[nodiscard]] typename ITemporalGraph<GPUUsage>::IntVector get_node_ids_host() const override;
    [[nodiscard]] typename ITemporalGraph<GPUUsage>::EdgeVector get_edges_host() override;
};

#endif //TEMPORALGRAPH_CPU_H
