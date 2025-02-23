#ifndef NODEEDGEINDEX_CPU_H
#define NODEEDGEINDEX_CPU_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "../interfaces/NodeEdgeIndex.cuh"
#include "../interfaces/NodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCPU
{
public:
    virtual ~NodeEdgeIndexCPU() = default;

    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using DoubleVector = typename SelectVectorType<double, GPUUsage>::type;

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
    virtual void rebuild(const EdgeData<GPUUsage>& edges, const NodeMapping<GPUUsage>& mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] virtual std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] virtual std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] virtual size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;

    virtual void update_temporal_weights(const EdgeData<GPUUsage>& edges, double timescale_bound);

protected:
    SizeVector get_timestamp_offset_vector(bool forward, bool directed) const;
};

#endif //NODEEDGEINDEX_CPU_H
