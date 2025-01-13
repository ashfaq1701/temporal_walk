#include "NodeEdgeIndex.h"

void NodeEdgeIndex::clear() {
   outbound_offsets.clear();
   outbound_indices.clear();
   outbound_timestamp_groups.clear();

   inbound_offsets.clear();
   inbound_indices.clear();
   inbound_timestamp_groups.clear();
}

void NodeEdgeIndex::rebuild(
   const EdgeData& edges,
   const NodeMapping& mapping,
   bool is_directed) {

   const size_t num_nodes = mapping.size();

   // Initialize structures
   outbound_offsets.assign(num_nodes + 1, 0);
   outbound_timestamp_groups.resize(num_nodes);

   if (is_directed) {
       inbound_offsets.assign(num_nodes + 1, 0);
       inbound_timestamp_groups.resize(num_nodes);
   }

   // First pass: count edges per node
   for (size_t i = 0; i < edges.size(); i++) {
       int src_idx = mapping.to_dense(edges.sources[i]);
       int tgt_idx = mapping.to_dense(edges.targets[i]);

       outbound_offsets[src_idx + 1]++;
       if (is_directed) {
           inbound_offsets[tgt_idx + 1]++;
       } else {
           outbound_offsets[tgt_idx + 1]++;
       }
   }

   // Calculate prefix sums for offsets
   for (size_t i = 1; i <= num_nodes; i++) {
       outbound_offsets[i] += outbound_offsets[i-1];
       if (is_directed) {
           inbound_offsets[i] += inbound_offsets[i-1];
       }
   }

   // Allocate index arrays
   outbound_indices.resize(outbound_offsets.back());
   if (is_directed) {
       inbound_indices.resize(inbound_offsets.back());
   }

   // Second pass: fill indices
   std::vector<size_t> outbound_current(num_nodes, 0);
   std::vector<size_t> inbound_current;
   if (is_directed) {
       inbound_current.resize(num_nodes, 0);
   }

   for (size_t i = 0; i < edges.size(); i++) {
       int src_idx = mapping.to_dense(edges.sources[i]);
       int tgt_idx = mapping.to_dense(edges.targets[i]);

       // Place in outbound edges
       size_t out_pos = outbound_offsets[src_idx] + outbound_current[src_idx]++;
       outbound_indices[out_pos] = i;

       if (is_directed) {
           // Place in inbound edges
           size_t in_pos = inbound_offsets[tgt_idx] + inbound_current[tgt_idx]++;
           inbound_indices[in_pos] = i;
       } else {
           // For undirected, also place in target's outbound
           size_t out_pos2 = outbound_offsets[tgt_idx] + outbound_current[tgt_idx]++;
           outbound_indices[out_pos2] = i;
       }
   }

   // Build timestamp groups for each node
   for (size_t node = 0; node < num_nodes; node++) {
       // Handle outbound timestamps
       size_t start = outbound_offsets[node];
       size_t end = outbound_offsets[node + 1];

       if (start < end) {
           auto& groups = outbound_timestamp_groups[node];
           groups.push_back(start);

           for (size_t i = start + 1; i < end; i++) {
               if (edges.timestamps[outbound_indices[i]] !=
                   edges.timestamps[outbound_indices[i-1]]) {
                   groups.push_back(i);
               }
           }
           groups.push_back(end);
       }

       // Handle inbound timestamps for directed graphs
       if (is_directed) {
           start = inbound_offsets[node];
           end = inbound_offsets[node + 1];

           if (start < end) {
               auto& groups = inbound_timestamp_groups[node];
               groups.push_back(start);

               for (size_t i = start + 1; i < end; i++) {
                   if (edges.timestamps[inbound_indices[i]] !=
                       edges.timestamps[inbound_indices[i-1]]) {
                       groups.push_back(i);
                   }
               }
               groups.push_back(end);
           }
       }
   }
}

std::pair<size_t, size_t> NodeEdgeIndex::get_edge_range(
   int dense_node_id,
   bool forward,
   bool is_directed) const {

   if (is_directed) {
       const auto& offsets = forward ? outbound_offsets : inbound_offsets;
       if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
           return {0, 0};
       }
       return {offsets[dense_node_id], offsets[dense_node_id + 1]};
   } else {
       if (dense_node_id < 0 || dense_node_id >= outbound_offsets.size() - 1) {
           return {0, 0};
       }
       return {outbound_offsets[dense_node_id], outbound_offsets[dense_node_id + 1]};
   }
}

std::pair<size_t, size_t> NodeEdgeIndex::get_timestamp_group_range(
   int dense_node_id,
   size_t group_idx,
   bool forward,
   bool is_directed) const {

   const auto& groups = (is_directed && !forward) ?
       inbound_timestamp_groups[dense_node_id] :
       outbound_timestamp_groups[dense_node_id];

   if (group_idx >= groups.size() - 1) {
       return {0, 0};
   }

   return {groups[group_idx], groups[group_idx + 1]};
}

size_t NodeEdgeIndex::get_timestamp_group_count(
   int dense_node_id,
   bool forward,
   bool is_directed) const {

   const auto& groups = (is_directed && !forward) ?
       inbound_timestamp_groups[dense_node_id] :
       outbound_timestamp_groups[dense_node_id];

   return groups.empty() ? 0 : groups.size() - 1;
}
