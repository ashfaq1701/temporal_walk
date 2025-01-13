#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.h"

struct NodeEdgeIndex {
   // Base CSR format for edges
   std::vector<size_t> outbound_offsets;    // Size: num_nodes + 1
   std::vector<size_t> outbound_indices;    // Size: num_edges

   // CSR format for timestamp groups
   std::vector<size_t> outbound_group_offsets;  // Size: num_nodes + 1
   std::vector<size_t> outbound_group_indices;  // Each group's start position in outbound_indices

   // Mirror structures for directed graphs
   std::vector<size_t> inbound_offsets;
   std::vector<size_t> inbound_indices;
   std::vector<size_t> inbound_group_offsets;
   std::vector<size_t> inbound_group_indices;

   void clear();
   void rebuild(const EdgeData& edges, const NodeMapping& mapping, bool is_directed);

   // Core access methods
   std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
   std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward, bool is_directed) const;
   size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;
};

#endif //NODEEDGEINDEX_H
