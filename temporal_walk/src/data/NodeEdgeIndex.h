#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.h"

struct NodeEdgeIndex {
    // For directed graphs:
    // outbound: edges where node is source
    // inbound: edges where node is target
    std::vector<size_t> outbound_offsets;
    std::vector<size_t> outbound_indices;
    std::vector<size_t> inbound_offsets;
    std::vector<size_t> inbound_indices;

    void rebuild(const EdgeData& edges, const NodeMapping& mapping, bool is_directed);
    void clear();

    // Helper methods for edge access
    std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
};

#endif //NODEEDGEINDEX_H
