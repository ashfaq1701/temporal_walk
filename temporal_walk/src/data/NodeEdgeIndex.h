#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.h"

// In header file:
struct NodeEdgeIndex {
   // Base CSR format for outbound edges
   std::vector<size_t> outbound_offsets;  // Per node edge ranges
   std::vector<size_t> outbound_indices;  // Edge indices into main edge array
   std::vector<std::vector<size_t>> outbound_timestamp_groups;  // Per node timestamp groups

   // Mirror for inbound edges (for directed graphs)
   std::vector<size_t> inbound_offsets;
   std::vector<size_t> inbound_indices;
   std::vector<std::vector<size_t>> inbound_timestamp_groups;

   void clear();
   void rebuild(const EdgeData& edges, const NodeMapping& mapping, bool is_directed);

   // Get edge range for a node
   std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;

   // Get range for a specific timestamp group
   std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const;

   // Get number of timestamp groups for a node
   size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;
};

#endif //NODEEDGEINDEX_H
