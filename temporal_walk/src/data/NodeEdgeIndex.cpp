#include "NodeEdgeIndex.h"

#include <iostream>

void NodeEdgeIndex::clear() {
   // Clear edge CSR structures
   outbound_offsets.clear();
   outbound_indices.clear();
   outbound_timestamp_group_offsets.clear();
   outbound_timestamp_group_indices.clear();

   // Clear inbound structures
   inbound_offsets.clear();
   inbound_indices.clear();
   inbound_timestamp_group_offsets.clear();
   inbound_timestamp_group_indices.clear();
}

void NodeEdgeIndex::rebuild(
   const EdgeData& edges,
   const NodeMapping& mapping,
   bool is_directed) {

   const size_t num_nodes = mapping.size();

   // Initialize base CSR structures
   outbound_offsets.assign(num_nodes + 1, 0);
   outbound_timestamp_group_offsets.assign(num_nodes + 1, 0);

   if (is_directed) {
       inbound_offsets.assign(num_nodes + 1, 0);
       inbound_timestamp_group_offsets.assign(num_nodes + 1, 0);
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

   // Calculate prefix sums for edge offsets
   for (size_t i = 1; i <= num_nodes; i++) {
       outbound_offsets[i] += outbound_offsets[i-1];
       if (is_directed) {
           inbound_offsets[i] += inbound_offsets[i-1];
       }
   }

   // Allocate edge index arrays
   outbound_indices.resize(outbound_offsets.back());
   if (is_directed) {
       inbound_indices.resize(inbound_offsets.back());
   }

   // Second pass: fill edge indices
   std::vector<size_t> outbound_current(num_nodes, 0);
   std::vector<size_t> inbound_current;
   if (is_directed) {
       inbound_current.resize(num_nodes, 0);
   }

   for (size_t i = 0; i < edges.size(); i++) {
       int src_idx = mapping.to_dense(edges.sources[i]);
       int tgt_idx = mapping.to_dense(edges.targets[i]);

       size_t out_pos = outbound_offsets[src_idx] + outbound_current[src_idx]++;
       outbound_indices[out_pos] = i;

       if (is_directed) {
           size_t in_pos = inbound_offsets[tgt_idx] + inbound_current[tgt_idx]++;
           inbound_indices[in_pos] = i;
       } else {
           size_t out_pos2 = outbound_offsets[tgt_idx] + outbound_current[tgt_idx]++;
           outbound_indices[out_pos2] = i;
       }
   }

   // Third pass: count timestamp groups
   std::vector<size_t> outbound_group_count(num_nodes, 0);
   std::vector<size_t> inbound_group_count;
   if (is_directed) {
       inbound_group_count.resize(num_nodes, 0);
   }

   for (size_t node = 0; node < num_nodes; node++) {
       size_t start = outbound_offsets[node];
       size_t end = outbound_offsets[node + 1];

       if (start < end) {
           outbound_group_count[node] = 1;  // First group
           for (size_t i = start + 1; i < end; i++) {
               if (edges.timestamps[outbound_indices[i]] !=
                   edges.timestamps[outbound_indices[i-1]]) {
                   outbound_group_count[node]++;
               }
           }
       }

       if (is_directed) {
           start = inbound_offsets[node];
           end = inbound_offsets[node + 1];

           if (start < end) {
               inbound_group_count[node] = 1;  // First group
               for (size_t i = start + 1; i < end; i++) {
                   if (edges.timestamps[inbound_indices[i]] !=
                       edges.timestamps[inbound_indices[i-1]]) {
                       inbound_group_count[node]++;
                   }
               }
           }
       }
   }

   // Calculate prefix sums for group offsets
   for (size_t i = 0; i < num_nodes; i++) {
       outbound_timestamp_group_offsets[i + 1] = outbound_timestamp_group_offsets[i] + outbound_group_count[i];
       if (is_directed) {
           inbound_timestamp_group_offsets[i + 1] = inbound_timestamp_group_offsets[i] + inbound_group_count[i];
       }
   }

   // Allocate and fill group indices
   outbound_timestamp_group_indices.resize(outbound_timestamp_group_offsets.back());
   if (is_directed) {
       inbound_timestamp_group_indices.resize(inbound_timestamp_group_offsets.back());
   }

   // Final pass: fill group indices
   for (size_t node = 0; node < num_nodes; node++) {
       size_t start = outbound_offsets[node];
       size_t end = outbound_offsets[node + 1];
       size_t group_pos = outbound_timestamp_group_offsets[node];

       if (start < end) {
           outbound_timestamp_group_indices[group_pos++] = start;
           for (size_t i = start + 1; i < end; i++) {
               if (edges.timestamps[outbound_indices[i]] !=
                   edges.timestamps[outbound_indices[i-1]]) {
                   outbound_timestamp_group_indices[group_pos++] = i;
               }
           }
       }

       if (is_directed) {
           start = inbound_offsets[node];
           end = inbound_offsets[node + 1];
           group_pos = inbound_timestamp_group_offsets[node];

           if (start < end) {
               inbound_timestamp_group_indices[group_pos++] = start;
               for (size_t i = start + 1; i < end; i++) {
                   if (edges.timestamps[inbound_indices[i]] !=
                       edges.timestamps[inbound_indices[i-1]]) {
                       inbound_timestamp_group_indices[group_pos++] = i;
                   }
               }
           }
       }
   }
}

void NodeEdgeIndex::update_temporal_weights(const EdgeData& edges) {

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

   const auto& group_offsets = (is_directed && !forward) ?
       inbound_timestamp_group_offsets : outbound_timestamp_group_offsets;
   const auto& group_indices = (is_directed && !forward) ?
       inbound_timestamp_group_indices : outbound_timestamp_group_indices;
   const auto& edge_offsets = (is_directed && !forward) ?
       inbound_offsets : outbound_offsets;

   if (dense_node_id < 0 || dense_node_id >= group_offsets.size() - 1) {
       return {0, 0};
   }

   size_t num_groups = group_offsets[dense_node_id + 1] - group_offsets[dense_node_id];
   if (group_idx >= num_groups) {
       return {0, 0};
   }

   size_t group_start_idx = group_offsets[dense_node_id] + group_idx;
   size_t group_start = group_indices[group_start_idx];

   // Group end is either next group's start or node's edge range end
   size_t group_end;
   if (group_idx == num_groups - 1) {
       group_end = edge_offsets[dense_node_id + 1];
   } else {
       group_end = group_indices[group_start_idx + 1];
   }

   return {group_start, group_end};
}

size_t NodeEdgeIndex::get_timestamp_group_count(
   int dense_node_id,
   bool forward,
   bool is_directed) const {

   const auto& offsets = (is_directed && !forward) ?
       inbound_timestamp_group_offsets : outbound_timestamp_group_offsets;

   if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
       return 0;
   }

   return offsets[dense_node_id + 1] - offsets[dense_node_id];
}
