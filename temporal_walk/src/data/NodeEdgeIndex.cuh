#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.cuh"

class NodeEdgeIndex
{
public:
    bool use_gpu;

    explicit NodeEdgeIndex(bool use_gpu);

    // Base CSR format for edges
    VectorTypes<size_t>::Vector outbound_offsets; // Size: num_nodes + 1
    VectorTypes<size_t>::Vector outbound_indices; // Size: num_edges

    // CSR format for timestamp groups
    VectorTypes<size_t>::Vector outbound_timestamp_group_offsets; // Size: num_nodes + 1
    VectorTypes<size_t>::Vector outbound_timestamp_group_indices; // Each group's start position in outbound_indices

    // Mirror structures for directed graphs
    VectorTypes<size_t>::Vector inbound_offsets;
    VectorTypes<size_t>::Vector inbound_indices;
    VectorTypes<size_t>::Vector inbound_timestamp_group_offsets;
    VectorTypes<size_t>::Vector inbound_timestamp_group_indices;

    VectorTypes<double>::Vector outbound_forward_cumulative_weights_exponential;   // For all forward walks
    VectorTypes<double>::Vector outbound_backward_cumulative_weights_exponential;  // For undirected backward walks
    VectorTypes<double>::Vector inbound_backward_cumulative_weights_exponential;   // For directed backward walks

    void clear();
    void rebuild(const EdgeData& edges, const NodeMapping& mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;

    void update_temporal_weights(const EdgeData& edges, double timescale_bound);

private:
    [[nodiscard]] VectorTypes<size_t>::Vector get_timestamp_offset_vector(bool forward, bool directed) const;
};

#endif //NODEEDGEINDEX_H
