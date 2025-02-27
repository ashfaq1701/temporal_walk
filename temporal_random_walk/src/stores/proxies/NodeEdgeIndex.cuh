#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <functional>
#include "../cpu/NodeEdgeIndexCPU.cuh"
#include "../cuda/NodeEdgeIndexCUDA.cuh"

#include "../../data/enums.h"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndex : protected std::conditional_t<
    (GPUUsage == GPUUsageMode::ON_CPU), NodeEdgeIndexCPU<GPUUsage>, NodeEdgeIndexCUDA<GPUUsage>> {

protected:
    using BaseType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeEdgeIndexCPU<GPUUsage>, NodeEdgeIndexCUDA<GPUUsage>>;

    BaseType node_edge_index;

public:
    using BaseType::outbound_offsets;
    using BaseType::outbound_indices;

    using BaseType::outbound_timestamp_group_offsets;
    using BaseType::outbound_timestamp_group_indices;

    using BaseType::inbound_offsets;
    using BaseType::inbound_indices;
    using BaseType::inbound_timestamp_group_offsets;
    using BaseType::inbound_timestamp_group_indices;

    using BaseType::outbound_forward_cumulative_weights_exponential;
    using BaseType::outbound_backward_cumulative_weights_exponential;
    using BaseType::inbound_backward_cumulative_weights_exponential;

    void clear();
    void rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;
};

#endif //NODEEDGEINDEX_H