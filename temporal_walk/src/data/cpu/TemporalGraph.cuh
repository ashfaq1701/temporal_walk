#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <vector>
#include <cstdint>
#include <tuple>
#include <functional>

#include "../../core/structs.h"
#include "../../random/RandomPicker.h"

#include "NodeMapping.cuh"
#include "EdgeData.cuh"
#include "NodeEdgeIndex.cuh"

#include "../cuda/NodeMappingCUDA.cuh"
#include "../cuda/EdgeDataCUDA.cuh"
#include "../cuda/NodeEdgeIndexCUDA.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraph
{
private:
    #ifdef HAS_CUDA
    using NodeIndexType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeEdgeIndex<GPUUsage>, NodeEdgeIndexCUDA<GPUUsage>>;

    using EdgeDataType = std::conditional_t<
        (GPUUsage == ON_CPU), EdgeData<GPUUsage>, EdgeDataCUDA<GPUUsage>>;

    using NodeMappingType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeMapping<GPUUsage>, NodeMappingCUDA<GPUUsage>>;
    #else
    using NodeIndexType = NodeEdgeIndex<GPUUsage>;
    using EdgeDataType = EdgeData<GPUUsage>;
    using NodeMappingType = NodeMapping<GPUUsage>;
    #endif

    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64TVector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using BoolVector = typename SelectVectorType<bool, GPUUsage>::type;

    int64_t time_window; // Time duration to keep edges (-1 means keep all)
    bool enable_weight_computation;
    double timescale_bound;
    int64_t latest_timestamp; // Track latest edge timestamp

    void delete_old_edges();

public:
    virtual ~TemporalGraph() = default;

    bool is_directed;

    NodeIndexType node_index; // Node to edge mappings
    EdgeDataType edges; // Main edge storage
    NodeMappingType node_mapping; // Sparse to dense node ID mapping

    explicit TemporalGraph(
        bool directed,
        int64_t window = -1,
        bool enable_weight_computation = false,
        double timescale_bound=-1);

    void sort_and_merge_edges(size_t start_idx);

    // Edge addition
    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& new_edges);

    void update_temporal_weights();

    // Timestamp group counting
    [[nodiscard]] virtual size_t count_timestamps_less_than(int64_t timestamp) const;
    [[nodiscard]] virtual size_t count_timestamps_greater_than(int64_t timestamp) const;
    [[nodiscard]] virtual size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;
    [[nodiscard]] virtual size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    // Edge selection
    [[nodiscard]] std::tuple<int, int, int64_t> get_edge_at(
        RandomPicker& picker, int64_t timestamp = -1,
        bool forward = true) const;

    [[nodiscard]] std::tuple<int, int, int64_t> get_node_edge_at(int node_id,
                                                                 RandomPicker& picker,
                                                                 int64_t timestamp = -1,
                                                                 bool forward = true) const;

    // Utility methods
    [[nodiscard]] size_t get_total_edges() const { return edges.size(); }
    [[nodiscard]] size_t get_node_count() const { return node_mapping.active_size(); }
    [[nodiscard]] int64_t get_latest_timestamp() const { return latest_timestamp; }
    [[nodiscard]] std::vector<int> get_node_ids() const;
    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges();
};

#endif //TEMPORALGRAPH_H
