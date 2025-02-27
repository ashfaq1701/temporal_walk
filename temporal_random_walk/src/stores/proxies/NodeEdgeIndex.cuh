#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <functional>
#include "../cpu/NodeEdgeIndexCPU.cuh"
#include "../cuda/NodeEdgeIndexCUDA.cuh"

#include "../../data/enums.h"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndex {
protected:
    using NodeEdgeIndexType = std::conditional_t<
        (GPUUsage == ON_CPU), NodeEdgeIndexCPU<GPUUsage>, NodeEdgeIndexCUDA<GPUUsage>>;

    NodeEdgeIndexType node_edge_index;

public:
    void clear();
    void rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool directed) const;
};

#endif //NODEEDGEINDEX_H