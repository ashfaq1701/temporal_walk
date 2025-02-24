#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <vector>
#include <cstdint>
#include <tuple>

#include "../../common/types.cuh"
#include "../../structs/structs.cuh"
#include "../../structs/enums.h"

#include "../../random/RandomPicker.h"

#include "../interfaces/NodeMapping.cuh"
#include "../interfaces/EdgeData.cuh"
#include "../interfaces/NodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraph
{
public:
    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64TVector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using BoolVector = typename SelectVectorType<bool, GPUUsage>::type;

    using EdgeVector = typename SelectVectorType<Edge, GPUUsage>::type;

    int64_t time_window = -1; // Time duration to keep edges (-1 means keep all)
    bool enable_weight_computation = false;
    double timescale_bound = -1;
    int64_t latest_timestamp = 0; // Track latest edge timestamp

    virtual ~TemporalGraph() = default;

    bool is_directed = false;

    NodeEdgeIndex<GPUUsage> node_index; // Node to edge mappings
    EdgeData<GPUUsage> edges; // Main edge storage
    NodeMapping<GPUUsage> node_mapping; // Sparse to dense node ID mapping

    /**
    * HOST METHODS
    */
    virtual HOST void sort_and_merge_edges_host(size_t start_idx) {}

    // Edge addition
    virtual HOST void add_multiple_edges_host(const EdgeVector new_edges) {}

    virtual HOST void update_temporal_weights_host() {}

    virtual HOST void delete_old_edges_host() {}

    // Timestamp group counting
    [[nodiscard]] virtual HOST size_t count_timestamps_less_than_host(int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual HOST size_t count_timestamps_greater_than_host(int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual HOST size_t count_node_timestamps_less_than_host(int node_id, int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual HOST size_t count_node_timestamps_greater_than_host(int node_id, int64_t timestamp) const { return 0; }

    // Edge selection
    [[nodiscard]] virtual HOST Edge get_edge_at_host(
        RandomPicker& picker, int64_t timestamp = -1,
        bool forward = true) const { return {}; }

    [[nodiscard]] virtual HOST Edge get_node_edge_at_host(int node_id,
                                                                 RandomPicker& picker,
                                                                 int64_t timestamp = -1,
                                                                 bool forward = true) const { return {}; }

    // Utility methods
    [[nodiscard]] virtual HOST size_t get_total_edges_host() const { return 0; }
    [[nodiscard]] virtual HOST size_t get_node_count_host() const { return 0; }
    [[nodiscard]] virtual HOST int64_t get_latest_timestamp_host() { return latest_timestamp; }
    [[nodiscard]] virtual HOST IntVector get_node_ids_host() const { return IntVector(); }
    [[nodiscard]] virtual HOST EdgeVector get_edges_host() { return EdgeVector(); }


    /**
    * DEVICE METHODS
    */
    virtual DEVICE void sort_and_merge_edges_device(size_t start_idx) {}

    // Edge addition
    virtual DEVICE void add_multiple_edges_device(const std::vector<std::tuple<int, int, int64_t>>& new_edges) {}

    virtual DEVICE void update_temporal_weights_device() {}

    virtual DEVICE void delete_old_edges_device() {}

    // Timestamp group counting
    [[nodiscard]] virtual DEVICE size_t count_timestamps_less_than_device(int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual DEVICE size_t count_timestamps_greater_than_device(int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual DEVICE size_t count_node_timestamps_less_than_device(int node_id, int64_t timestamp) const { return 0; }
    [[nodiscard]] virtual DEVICE size_t count_node_timestamps_greater_than_device(int node_id, int64_t timestamp) const { return 0; }

    // Edge selection
    [[nodiscard]] virtual DEVICE Edge get_edge_at_device(
        RandomPicker& picker, int64_t timestamp = -1,
        bool forward = true) const { return {}; }

    [[nodiscard]] virtual DEVICE Edge get_node_edge_at_device(int node_id,
                                                                 RandomPicker& picker,
                                                                 int64_t timestamp = -1,
                                                                 bool forward = true) const { return {}; }

    // Utility methods
    [[nodiscard]] virtual DEVICE size_t get_total_edges_device() const { return 0; }
    [[nodiscard]] virtual DEVICE size_t get_node_count_device() const { return 0; }
    [[nodiscard]] virtual DEVICE int64_t get_latest_timestamp_device() const { return latest_timestamp; }
    [[nodiscard]] virtual DEVICE IntVector get_node_ids_device() const { return IntVector(); }
    [[nodiscard]] virtual DEVICE EdgeVector get_edges_device() { return EdgeVector(); }
};

#endif //TEMPORALGRAPH_H
