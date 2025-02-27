#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <functional>
#include "../cpu/NodeEdgeIndexCPU.cuh"
#include "../cuda/NodeEdgeIndexCUDA.cuh"

#include "../../data/enums.h"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndex {

protected:
    using BaseType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeEdgeIndexCPU<GPUUsage>, NodeEdgeIndexCUDA<GPUUsage>>;

public:

    INodeEdgeIndex<GPUUsage>* node_edge_index;

    NodeEdgeIndex();
    explicit NodeEdgeIndex(INodeEdgeIndex<GPUUsage>* node_edge_index);

    // Accessors for outbound structures
    auto& outbound_offsets() { return node_edge_index->outbound_offsets; }
    const auto& outbound_offsets() const { return node_edge_index->outbound_offsets; }

    auto& outbound_indices() { return node_edge_index->outbound_indices; }
    const auto& outbound_indices() const { return node_edge_index->outbound_indices; }

    auto& outbound_timestamp_group_offsets() { return node_edge_index->outbound_timestamp_group_offsets; }
    const auto& outbound_timestamp_group_offsets() const { return node_edge_index->outbound_timestamp_group_offsets; }

    auto& outbound_timestamp_group_indices() { return node_edge_index->outbound_timestamp_group_indices; }
    const auto& outbound_timestamp_group_indices() const { return node_edge_index->outbound_timestamp_group_indices; }

    // Accessors for inbound structures
    auto& inbound_offsets() { return node_edge_index->inbound_offsets; }
    const auto& inbound_offsets() const { return node_edge_index->inbound_offsets; }

    auto& inbound_indices() { return node_edge_index->inbound_indices; }
    const auto& inbound_indices() const { return node_edge_index->inbound_indices; }

    auto& inbound_timestamp_group_offsets() { return node_edge_index->inbound_timestamp_group_offsets; }
    const auto& inbound_timestamp_group_offsets() const { return node_edge_index->inbound_timestamp_group_offsets; }

    auto& inbound_timestamp_group_indices() { return node_edge_index->inbound_timestamp_group_indices; }
    const auto& inbound_timestamp_group_indices() const { return node_edge_index->inbound_timestamp_group_indices; }

    // Accessors for cumulative weights
    auto& outbound_forward_cumulative_weights_exponential() { return node_edge_index->outbound_forward_cumulative_weights_exponential; }
    const auto& outbound_forward_cumulative_weights_exponential() const { return node_edge_index->outbound_forward_cumulative_weights_exponential; }

    auto& outbound_backward_cumulative_weights_exponential() { return node_edge_index->outbound_backward_cumulative_weights_exponential; }
    const auto& outbound_backward_cumulative_weights_exponential() const { return node_edge_index->outbound_backward_cumulative_weights_exponential; }

    auto& inbound_backward_cumulative_weights_exponential() { return node_edge_index->inbound_backward_cumulative_weights_exponential; }
    const auto& inbound_backward_cumulative_weights_exponential() const { return node_edge_index->inbound_backward_cumulative_weights_exponential; }


    void clear();
    void rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;

    void update_temporal_weights(const IEdgeData<GPUUsage>* edges, double timescale_bound);

};

#endif //NODEEDGEINDEX_H