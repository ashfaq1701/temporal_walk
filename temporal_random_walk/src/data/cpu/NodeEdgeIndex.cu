#include "NodeEdgeIndex.cuh"

#include <iostream>

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::clear() {
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

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::rebuild(
   const EdgeData<GPUUsage>& edges,
   const NodeMapping<GPUUsage>& mapping,
   const bool is_directed) {

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
       const int src_idx = mapping.to_dense(edges.sources[i]);
       const int tgt_idx = mapping.to_dense(edges.targets[i]);

       ++outbound_offsets[src_idx + 1];
       if (is_directed) {
           ++inbound_offsets[tgt_idx + 1];
       } else {
           ++outbound_offsets[tgt_idx + 1];
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
   SizeVector outbound_current(num_nodes, 0);
   SizeVector inbound_current;
   if (is_directed) {
       inbound_current.resize(num_nodes, 0);
   }

   for (size_t i = 0; i < edges.size(); i++) {
       const int src_idx = mapping.to_dense(edges.sources[i]);
       const int tgt_idx = mapping.to_dense(edges.targets[i]);

       const size_t out_pos = outbound_offsets[src_idx] + outbound_current[src_idx]++;
       outbound_indices[out_pos] = i;

       if (is_directed) {
           const size_t in_pos = inbound_offsets[tgt_idx] + inbound_current[tgt_idx]++;
           inbound_indices[in_pos] = i;
       } else {
           const size_t out_pos2 = outbound_offsets[tgt_idx] + outbound_current[tgt_idx]++;
           outbound_indices[out_pos2] = i;
       }
   }

   // Third pass: count timestamp groups
   SizeVector outbound_group_count(num_nodes, 0);
   SizeVector inbound_group_count;
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
                   ++outbound_group_count[node];
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
                       ++inbound_group_count[node];
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

template<GPUUsageMode GPUUsage>
void NodeEdgeIndex<GPUUsage>::update_temporal_weights(const EdgeData<GPUUsage>& edges, double timescale_bound) {
    const size_t num_nodes = outbound_offsets.size() - 1;

    outbound_forward_cumulative_weights_exponential.resize(outbound_timestamp_group_indices.size());
    outbound_backward_cumulative_weights_exponential.resize(outbound_timestamp_group_indices.size());
    if (!inbound_offsets.empty()) {
        inbound_backward_cumulative_weights_exponential.resize(inbound_timestamp_group_indices.size());
    }

    // Process each node
    for (size_t node = 0; node < num_nodes; node++) {
        // Outbound weights
        const auto& outbound_offsets = get_timestamp_offset_vector(true, false);
        const size_t out_start = outbound_offsets[node];
        const size_t out_end = outbound_offsets[node + 1];

        if (out_start < out_end) {
            const size_t first_group_start = outbound_timestamp_group_indices[out_start];
            const size_t last_group_start = outbound_timestamp_group_indices[out_end - 1];
            const int64_t min_ts = edges.timestamps[outbound_indices[first_group_start]];
            const int64_t max_ts = edges.timestamps[outbound_indices[last_group_start]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                timescale_bound / time_diff : 1.0;

            double forward_sum = 0.0;
            double backward_sum = 0.0;

            // Calculate weights and sums
            for (size_t pos = out_start; pos < out_end; ++pos) {
                const size_t edge_start = outbound_timestamp_group_indices[pos];
                const int64_t group_ts = edges.timestamps[outbound_indices[edge_start]];

                const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                const double forward_scaled = timescale_bound > 0 ?
                    time_diff_forward * time_scale : time_diff_forward;
                const double backward_scaled = timescale_bound > 0 ?
                    time_diff_backward * time_scale : time_diff_backward;

                const double forward_weight = exp(forward_scaled);
                outbound_forward_cumulative_weights_exponential[pos] = forward_weight;
                forward_sum += forward_weight;

                const double backward_weight = exp(backward_scaled);
                outbound_backward_cumulative_weights_exponential[pos] = backward_weight;
                backward_sum += backward_weight;
            }

            // Normalize and compute cumulative sums
            double forward_cumsum = 0.0, backward_cumsum = 0.0;
            for (size_t pos = out_start; pos < out_end; ++pos) {
                outbound_forward_cumulative_weights_exponential[pos] /= forward_sum;
                outbound_backward_cumulative_weights_exponential[pos] /= backward_sum;

                forward_cumsum += outbound_forward_cumulative_weights_exponential[pos];
                backward_cumsum += outbound_backward_cumulative_weights_exponential[pos];

                outbound_forward_cumulative_weights_exponential[pos] = forward_cumsum;
                outbound_backward_cumulative_weights_exponential[pos] = backward_cumsum;
            }
        }

        // Inbound weights
        if (!inbound_offsets.empty()) {
            const auto& inbound_group_offsets = get_timestamp_offset_vector(false, true);
            const size_t in_start = inbound_group_offsets[node];
            const size_t in_end = inbound_group_offsets[node + 1];

            if (in_start < in_end) {
                const size_t first_group_start = inbound_timestamp_group_indices[in_start];
                const size_t last_group_start = inbound_timestamp_group_indices[in_end - 1];
                const int64_t min_ts = edges.timestamps[inbound_indices[first_group_start]];
                const int64_t max_ts = edges.timestamps[inbound_indices[last_group_start]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                    timescale_bound / time_diff : 1.0;

                double backward_sum = 0.0;

                // Calculate weights and sum
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    const size_t edge_start = inbound_timestamp_group_indices[pos];
                    const int64_t group_ts = edges.timestamps[inbound_indices[edge_start]];

                    const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                    const double backward_scaled = timescale_bound > 0 ?
                        time_diff_backward * time_scale : time_diff_backward;

                    const double backward_weight = exp(backward_scaled);
                    inbound_backward_cumulative_weights_exponential[pos] = backward_weight;
                    backward_sum += backward_weight;
                }

                // Normalize and compute cumulative sum
                double backward_cumsum = 0.0;
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    inbound_backward_cumulative_weights_exponential[pos] /= backward_sum;
                    backward_cumsum += inbound_backward_cumulative_weights_exponential[pos];
                    inbound_backward_cumulative_weights_exponential[pos] = backward_cumsum;
                }
            }
        }
    }
}

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> NodeEdgeIndex<GPUUsage>::get_edge_range(
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

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> NodeEdgeIndex<GPUUsage>::get_timestamp_group_range(
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

template<GPUUsageMode GPUUsage>
size_t NodeEdgeIndex<GPUUsage>::get_timestamp_group_count(
   int dense_node_id,
   bool forward,
   bool directed) const {

   const auto& offsets = get_timestamp_offset_vector(forward, directed);

   if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
       return 0;
   }

   return offsets[dense_node_id + 1] - offsets[dense_node_id];
}

template<GPUUsageMode GPUUsage>
[[nodiscard]] typename NodeEdgeIndex<GPUUsage>::SizeVector NodeEdgeIndex<GPUUsage>::get_timestamp_offset_vector(
    const bool forward,
    const bool directed) const {
    return (directed && !forward) ? inbound_timestamp_group_offsets : outbound_timestamp_group_offsets;
}

template class NodeEdgeIndex<GPUUsageMode::ON_CPU>;
