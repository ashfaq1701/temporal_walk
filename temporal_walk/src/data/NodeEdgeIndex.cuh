#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.cuh"

template<bool UseGPU>
struct NodeEdgeIndex
{
public:

    using SizeVector = typename SelectVectorType<size_t, UseGPU>::type;
    using DoubleVector = typename SelectVectorType<double, UseGPU>::type;

    // Base CSR format for edges
    SizeVector outbound_offsets{}; // Size: num_nodes + 1
    SizeVector outbound_indices{}; // Size: num_edges

    // CSR format for timestamp groups
    SizeVector outbound_timestamp_group_offsets{}; // Size: num_nodes + 1
    SizeVector outbound_timestamp_group_indices{}; // Each group's start position in outbound_indices

    // Mirror structures for directed graphs
    SizeVector inbound_offsets{};
    SizeVector inbound_indices{};
    SizeVector inbound_timestamp_group_offsets{};
    SizeVector inbound_timestamp_group_indices{};

    DoubleVector outbound_forward_cumulative_weights_exponential{};   // For all forward walks
    DoubleVector outbound_backward_cumulative_weights_exponential{};  // For undirected backward walks
    DoubleVector inbound_backward_cumulative_weights_exponential{};   // For directed backward walks

    void clear();
    void rebuild(const EdgeData<UseGPU>& edges, const NodeMapping<UseGPU>& mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;

    void update_temporal_weights(const EdgeData<UseGPU>& edges, double timescale_bound);

private:
    SizeVector get_timestamp_offset_vector(bool forward, bool directed) const;
};

#endif //NODEEDGEINDEX_H
