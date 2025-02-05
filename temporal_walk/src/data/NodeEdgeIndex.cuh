#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.cuh"
#include "../cuda/dual_vector.cuh"

struct NodeEdgeIndex
{
    bool use_gpu;

    // Base CSR format for edges
    DualVector<size_t> outbound_offsets; // Size: num_nodes + 1
    DualVector<size_t> outbound_indices; // Size: num_edges

    // CSR format for timestamp groups
    DualVector<size_t> outbound_timestamp_group_offsets; // Size: num_nodes + 1
    DualVector<size_t> outbound_timestamp_group_indices; // Each group's start position in outbound_indices

    // Mirror structures for directed graphs
    DualVector<size_t> inbound_offsets;
    DualVector<size_t> inbound_indices;
    DualVector<size_t> inbound_timestamp_group_offsets;
    DualVector<size_t> inbound_timestamp_group_indices;

    DualVector<double> outbound_forward_cumulative_weights_exponential;   // For all forward walks
    DualVector<double> outbound_backward_cumulative_weights_exponential;  // For undirected backward walks
    DualVector<double> inbound_backward_cumulative_weights_exponential;   // For directed backward walks

    explicit NodeEdgeIndex(bool use_gpu);

    void clear();
    void rebuild(const EdgeData& edges, const NodeMapping& mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;

    void update_temporal_weights(const EdgeData& edges, double timescale_bound);
};

#endif //NODEEDGEINDEX_H
