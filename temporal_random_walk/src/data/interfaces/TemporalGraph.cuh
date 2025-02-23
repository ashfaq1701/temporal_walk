#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <vector>
#include <cstdint>
#include <tuple>

#include "../cpu/NodeEdgeIndexCPU.cuh"
#include "../cpu/EdgeDataCPU.cuh"
#include "../cpu/NodeMappingCPU.cuh"

#include "../cuda/NodeEdgeIndexCUDA.cuh"
#include "../cuda/EdgeDataCUDA.cuh"
#include "../cuda/NodeMappingCUDA.cuh"

#include "../../core/structs.h"
#include "../../random/RandomPicker.h"

#include "../interfaces/NodeMapping.cuh"
#include "../interfaces/EdgeData.cuh"
#include "../interfaces/NodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class TemporalGraph
{
protected:
    #ifdef HAS_CUDA
    using NodeIndexType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeEdgeIndexCPU<GPUUsage>, NodeEdgeIndexCUDA<GPUUsage>>;

    using EdgeDataType = std::conditional_t<
        (GPUUsage == ON_CPU), EdgeDataCPU<GPUUsage>, EdgeDataCUDA<GPUUsage>>;

    using NodeMappingType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeMappingCPU<GPUUsage>, NodeMappingCUDA<GPUUsage>>;
    #else
    using NodeIndexType = NodeEdgeIndexCPU<GPUUsage>;
    using EdgeDataType = EdgeDataCPU<GPUUsage>;
    using NodeMappingType = NodeMappingCPU<GPUUsage>;
    #endif

    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64TVector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using BoolVector = typename SelectVectorType<bool, GPUUsage>::type;

    int64_t time_window; // Time duration to keep edges (-1 means keep all)
    bool enable_weight_computation;
    double timescale_bound;
    int64_t latest_timestamp; // Track latest edge timestamp

public:
    virtual ~TemporalGraph() = default;

    bool is_directed;

    NodeIndexType node_index; // Node to edge mappings
    EdgeDataType edges; // Main edge storage
    NodeMappingType node_mapping; // Sparse to dense node ID mapping

    /**
    * HOST METHODS
    */
    virtual HOST void sort_and_merge_edges_host(size_t start_idx);

    // Edge addition
    virtual HOST void add_multiple_edges_host(const std::vector<std::tuple<int, int, int64_t>>& new_edges);

    virtual HOST void update_temporal_weights_host();

    virtual HOST void delete_old_edges_host();

    // Timestamp group counting
    [[nodiscard]] virtual HOST size_t count_timestamps_less_than_host(int64_t timestamp) const;
    [[nodiscard]] virtual HOST size_t count_timestamps_greater_than_host(int64_t timestamp) const;
    [[nodiscard]] virtual HOST size_t count_node_timestamps_less_than_host(int node_id, int64_t timestamp) const;
    [[nodiscard]] virtual HOST size_t count_node_timestamps_greater_than_host(int node_id, int64_t timestamp) const;

    // Edge selection
    [[nodiscard]] virtual HOST std::tuple<int, int, int64_t> get_edge_at_host(
        RandomPicker& picker, int64_t timestamp = -1,
        bool forward = true) const;

    [[nodiscard]] virtual HOST std::tuple<int, int, int64_t> get_node_edge_at_host(int node_id,
                                                                 RandomPicker& picker,
                                                                 int64_t timestamp = -1,
                                                                 bool forward = true) const;

    // Utility methods
    [[nodiscard]] virtual HOST size_t get_total_edges_host() const;
    [[nodiscard]] virtual HOST size_t get_node_count_host() const;
    [[nodiscard]] virtual HOST int64_t get_latest_timestamp_host();
    [[nodiscard]] virtual HOST std::vector<int> get_node_ids_host() const;
    [[nodiscard]] virtual HOST std::vector<std::tuple<int, int, int64_t>> get_edges_host();


    /**
    * DEVICE METHODS
    */
    virtual DEVICE void sort_and_merge_edges_device(size_t start_idx);

    // Edge addition
    virtual DEVICE void add_multiple_edges_device(const std::vector<std::tuple<int, int, int64_t>>& new_edges);

    virtual DEVICE void update_temporal_weights_device();

    virtual DEVICE void delete_old_edges_device();

    // Timestamp group counting
    [[nodiscard]] virtual DEVICE size_t count_timestamps_less_than_device(int64_t timestamp) const;
    [[nodiscard]] virtual DEVICE size_t count_timestamps_greater_than_device(int64_t timestamp) const;
    [[nodiscard]] virtual DEVICE size_t count_node_timestamps_less_than_device(int node_id, int64_t timestamp) const;
    [[nodiscard]] virtual DEVICE size_t count_node_timestamps_greater_than_device(int node_id, int64_t timestamp) const;

    // Edge selection
    [[nodiscard]] virtual DEVICE std::tuple<int, int, int64_t> get_edge_at_device(
        RandomPicker& picker, int64_t timestamp = -1,
        bool forward = true) const;

    [[nodiscard]] virtual DEVICE std::tuple<int, int, int64_t> get_node_edge_at_device(int node_id,
                                                                 RandomPicker& picker,
                                                                 int64_t timestamp = -1,
                                                                 bool forward = true) const;

    // Utility methods
    [[nodiscard]] virtual DEVICE size_t get_total_edges_device() const;
    [[nodiscard]] virtual DEVICE size_t get_node_count_device() const;
    [[nodiscard]] virtual DEVICE int64_t get_latest_timestamp_device() const;
    [[nodiscard]] virtual DEVICE std::vector<int> get_node_ids_device() const;
    [[nodiscard]] virtual DEVICE std::vector<std::tuple<int, int, int64_t>> get_edges_device();
};

#endif //TEMPORALGRAPH_H
