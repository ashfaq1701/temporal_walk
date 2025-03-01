#include "NodeEdgeIndexCPU.cuh"

#include <iostream>

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::clear_host() {
   // Clear edge CSR structures
   this->outbound_offsets.clear();
   this->outbound_indices.clear();
   this->outbound_timestamp_group_offsets.clear();
   this->outbound_timestamp_group_indices.clear();

   // Clear inbound structures
   this->inbound_offsets.clear();
   this->inbound_indices.clear();
   this->inbound_timestamp_group_offsets.clear();
   this->inbound_timestamp_group_indices.clear();
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::populate_dense_ids_host(
        const IEdgeData<GPUUsage>* edges,
        const INodeMapping<GPUUsage>* mapping,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets)
{
    for (size_t i = 0; i < edges->size_host(); i++)
    {
        dense_sources[i] = mapping->to_dense_host(edges->sources[i]);
        dense_targets[i] = mapping->to_dense_host(edges->targets[i]);
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::allocate_node_edge_offsets(size_t num_nodes, bool is_directed)
{
    // Initialize base CSR structures
    this->outbound_offsets.assign(num_nodes + 1, 0);
    this->outbound_timestamp_group_offsets.assign(num_nodes + 1, 0);

    if (is_directed) {
        this->inbound_offsets.assign(num_nodes + 1, 0);
        this->inbound_timestamp_group_offsets.assign(num_nodes + 1, 0);
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_edge_offsets_host(
    const IEdgeData<GPUUsage>* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
    size_t num_nodes,
    bool is_directed)
{
    // First pass: count edges per node
    for (size_t i = 0; i < edges->size_host(); i++) {
        int src_idx = dense_sources[i];
        int tgt_idx = dense_targets[i];

        ++this->outbound_offsets[src_idx + 1];
        if (is_directed) {
            ++this->inbound_offsets[tgt_idx + 1];
        } else {
            ++this->outbound_offsets[tgt_idx + 1];
        }
    }

    // Calculate prefix sums for edge offsets
    for (size_t i = 1; i <= num_nodes; i++) {
        this->outbound_offsets[i] += this->outbound_offsets[i-1];
        if (is_directed) {
            this->inbound_offsets[i] += this->inbound_offsets[i-1];
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::allocate_node_edge_indices(bool is_directed)
{
    this->outbound_indices.resize(this->outbound_offsets.back());
    if (is_directed) {
        this->inbound_indices.resize(this->inbound_offsets.back());
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_edge_indices_host(
    const IEdgeData<GPUUsage>* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
    typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
    typename INodeEdgeIndex<GPUUsage>::SizeVector& outbound_running_index,
    typename INodeEdgeIndex<GPUUsage>::SizeVector& inbound_running_index,
    bool is_directed)
{
    auto edges_size = edges->size_host();
    for (size_t i = 0; i < edges->size_host(); ++i) {
        int src_idx = dense_sources[i];
        int tgt_idx = dense_targets[i];

        const size_t out_pos = this->outbound_offsets[src_idx] + outbound_running_index[src_idx]++;
        this->outbound_indices[out_pos] = i;

        if (is_directed) {
            const size_t in_pos = this->inbound_offsets[tgt_idx] + inbound_running_index[tgt_idx]++;
            this->inbound_indices[in_pos] = i;
        } else {
            const size_t out_pos2 = this->outbound_offsets[tgt_idx] + outbound_running_index[tgt_idx]++;
            this->outbound_indices[out_pos2] = i;
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_timestamp_offsets_host(
    const IEdgeData<GPUUsage>* edges,
    size_t num_nodes,
    bool is_directed)
{
    // Third pass: count timestamp groups
    typename INodeEdgeIndex<GPUUsage>::SizeVector outbound_group_count(num_nodes);
    typename INodeEdgeIndex<GPUUsage>::SizeVector inbound_group_count;
    if (is_directed) {
        inbound_group_count.resize(num_nodes);
    }

    for (size_t node = 0; node < num_nodes; node++) {
        size_t start = this->outbound_offsets[node];
        size_t end = this->outbound_offsets[node + 1];

        if (start < end) {
            outbound_group_count[node] = 1;  // First group
            for (size_t i = start + 1; i < end; ++i) {
                if (edges->timestamps[this->outbound_indices[i]] !=
                    edges->timestamps[this->outbound_indices[i-1]]) {
                    ++outbound_group_count[node];
                    }
            }
        }

        if (is_directed) {
            start = this->inbound_offsets[node];
            end = this->inbound_offsets[node + 1];

            if (start < end) {
                inbound_group_count[node] = 1;  // First group
                for (size_t i = start + 1; i < end; ++i) {
                    if (edges->timestamps[this->inbound_indices[i]] !=
                        edges->timestamps[this->inbound_indices[i-1]]) {
                        ++inbound_group_count[node];
                        }
                }
            }
        }
    }

    // Calculate prefix sums for group offsets
    for (size_t i = 0; i < num_nodes; i++) {
        this->outbound_timestamp_group_offsets[i + 1] = this->outbound_timestamp_group_offsets[i] + outbound_group_count[i];
        if (is_directed) {
            this->inbound_timestamp_group_offsets[i + 1] = this->inbound_timestamp_group_offsets[i] + inbound_group_count[i];
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::allocate_node_timestamp_indices(bool is_directed)
{
    this->outbound_timestamp_group_indices.resize(this->outbound_timestamp_group_offsets.back());
    if (is_directed) {
        this->inbound_timestamp_group_indices.resize(this->inbound_timestamp_group_offsets.back());
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_timestamp_indices_host(
    const IEdgeData<GPUUsage>* edges,
    size_t num_nodes,
    bool is_directed)
{
    // Final pass: fill group indices
    for (size_t node = 0; node < num_nodes; node++) {
        size_t start = this->outbound_offsets[node];
        size_t end = this->outbound_offsets[node + 1];
        size_t group_pos = this->outbound_timestamp_group_offsets[node];

        if (start < end) {
            this->outbound_timestamp_group_indices[group_pos++] = start;
            for (size_t i = start + 1; i < end; ++i) {
                if (edges->timestamps[this->outbound_indices[i]] !=
                    edges->timestamps[this->outbound_indices[i-1]]) {
                    this->outbound_timestamp_group_indices[group_pos++] = i;
                    }
            }
        }

        if (is_directed) {
            start = this->inbound_offsets[node];
            end = this->inbound_offsets[node + 1];
            group_pos = this->inbound_timestamp_group_offsets[node];

            if (start < end) {
                this->inbound_timestamp_group_indices[group_pos++] = start;
                for (size_t i = start + 1; i < end; ++i) {
                    if (edges->timestamps[this->inbound_indices[i]] !=
                        edges->timestamps[this->inbound_indices[i-1]]) {
                        this->inbound_timestamp_group_indices[group_pos++] = i;
                        }
                }
            }
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::rebuild(
   const IEdgeData<GPUUsage>* edges,
   const INodeMapping<GPUUsage>* mapping,
   const bool is_directed) {

   const size_t num_nodes = mapping->size_host();
    const size_t num_edges = edges->size_host();

    typename INodeEdgeIndex<GPUUsage>::IntVector dense_sources(num_edges);
    typename INodeEdgeIndex<GPUUsage>::IntVector dense_targets(num_edges);
    populate_dense_ids_host(edges, mapping, dense_sources, dense_targets);

    allocate_node_edge_offsets(num_nodes, is_directed);
    compute_node_edge_offsets_host(edges, dense_sources, dense_targets, num_nodes, is_directed);

   allocate_node_edge_indices(is_directed);

   typename INodeEdgeIndex<GPUUsage>::SizeVector outbound_running_index(num_nodes);
   typename INodeEdgeIndex<GPUUsage>::SizeVector inbound_running_index;
   if (is_directed) {
       inbound_running_index.resize(num_nodes);
   }

    compute_node_edge_indices_host(edges, dense_sources, dense_targets, outbound_running_index, inbound_running_index, is_directed);

    compute_node_timestamp_offsets_host(edges, num_nodes, is_directed);

    allocate_node_timestamp_indices(is_directed);

    compute_node_timestamp_indices_host(edges, num_nodes, is_directed);
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_temporal_weights_host(
    const IEdgeData<GPUUsage>* edges,
    double timescale_bound,
    size_t num_nodes)
{
    // Process each node
    for (size_t node = 0; node < num_nodes; node++) {
        // Outbound weights
        const auto& outbound_offsets = get_timestamp_offset_vector_host(true, false);
        const size_t out_start = outbound_offsets[node];
        const size_t out_end = outbound_offsets[node + 1];

        if (out_start < out_end) {
            const size_t first_group_start = this->outbound_timestamp_group_indices[out_start];
            const size_t last_group_start = this->outbound_timestamp_group_indices[out_end - 1];
            const int64_t min_ts = edges->timestamps[this->outbound_indices[first_group_start]];
            const int64_t max_ts = edges->timestamps[this->outbound_indices[last_group_start]];
            const auto time_diff = static_cast<double>(max_ts - min_ts);
            const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                timescale_bound / time_diff : 1.0;

            double forward_sum = 0.0;
            double backward_sum = 0.0;

            // Calculate weights and sums
            for (size_t pos = out_start; pos < out_end; ++pos) {
                const size_t edge_start = this->outbound_timestamp_group_indices[pos];
                const int64_t group_ts = edges->timestamps[this->outbound_indices[edge_start]];

                const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                const double forward_scaled = timescale_bound > 0 ?
                    time_diff_forward * time_scale : time_diff_forward;
                const double backward_scaled = timescale_bound > 0 ?
                    time_diff_backward * time_scale : time_diff_backward;

                const double forward_weight = exp(forward_scaled);
                this->outbound_forward_cumulative_weights_exponential[pos] = forward_weight;
                forward_sum += forward_weight;

                const double backward_weight = exp(backward_scaled);
                this->outbound_backward_cumulative_weights_exponential[pos] = backward_weight;
                backward_sum += backward_weight;
            }

            // Normalize and compute cumulative sums
            double forward_cumsum = 0.0, backward_cumsum = 0.0;
            for (size_t pos = out_start; pos < out_end; ++pos) {
                this->outbound_forward_cumulative_weights_exponential[pos] /= forward_sum;
                this->outbound_backward_cumulative_weights_exponential[pos] /= backward_sum;

                forward_cumsum += this->outbound_forward_cumulative_weights_exponential[pos];
                backward_cumsum += this->outbound_backward_cumulative_weights_exponential[pos];

                this->outbound_forward_cumulative_weights_exponential[pos] = forward_cumsum;
                this->outbound_backward_cumulative_weights_exponential[pos] = backward_cumsum;
            }
        }

        // Inbound weights
        if (!this->inbound_offsets.empty()) {
            const auto& inbound_group_offsets = get_timestamp_offset_vector_host(false, true);
            const size_t in_start = inbound_group_offsets[node];
            const size_t in_end = inbound_group_offsets[node + 1];

            if (in_start < in_end) {
                const size_t first_group_start = this->inbound_timestamp_group_indices[in_start];
                const size_t last_group_start = this->inbound_timestamp_group_indices[in_end - 1];
                const int64_t min_ts = edges->timestamps[this->inbound_indices[first_group_start]];
                const int64_t max_ts = edges->timestamps[this->inbound_indices[last_group_start]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                    timescale_bound / time_diff : 1.0;

                double backward_sum = 0.0;

                // Calculate weights and sum
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    const size_t edge_start = this->inbound_timestamp_group_indices[pos];
                    const int64_t group_ts = edges->timestamps[this->inbound_indices[edge_start]];

                    const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                    const double backward_scaled = timescale_bound > 0 ?
                        time_diff_backward * time_scale : time_diff_backward;

                    const double backward_weight = exp(backward_scaled);
                    this->inbound_backward_cumulative_weights_exponential[pos] = backward_weight;
                    backward_sum += backward_weight;
                }

                // Normalize and compute cumulative sum
                double backward_cumsum = 0.0;
                for (size_t pos = in_start; pos < in_end; ++pos) {
                    this->inbound_backward_cumulative_weights_exponential[pos] /= backward_sum;
                    backward_cumsum += this->inbound_backward_cumulative_weights_exponential[pos];
                    this->inbound_backward_cumulative_weights_exponential[pos] = backward_cumsum;
                }
            }
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::update_temporal_weights(const IEdgeData<GPUUsage>* edges, double timescale_bound) {
    const size_t num_nodes = this->outbound_offsets.size() - 1;

    this->outbound_forward_cumulative_weights_exponential.resize(this->outbound_timestamp_group_indices.size());
    this->outbound_backward_cumulative_weights_exponential.resize(this->outbound_timestamp_group_indices.size());
    if (!this->inbound_offsets.empty()) {
        this->inbound_backward_cumulative_weights_exponential.resize(this->inbound_timestamp_group_indices.size());
    }

    compute_temporal_weights_host(edges, timescale_bound, num_nodes);
}

template<GPUUsageMode GPUUsage>
HOST SizeRange NodeEdgeIndexCPU<GPUUsage>::get_edge_range_host(
   int dense_node_id,
   bool forward,
   bool is_directed) const {

   if (is_directed) {
       const auto& offsets = forward ? this->outbound_offsets : this->inbound_offsets;
       if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
           return SizeRange{0, 0};
       }
       return SizeRange{offsets[dense_node_id], offsets[dense_node_id + 1]};
   } else {
       if (dense_node_id < 0 || dense_node_id >= this->outbound_offsets.size() - 1) {
           return SizeRange{0, 0};
       }
       return SizeRange{this->outbound_offsets[dense_node_id], this->outbound_offsets[dense_node_id + 1]};
   }
}

template<GPUUsageMode GPUUsage>
HOST SizeRange NodeEdgeIndexCPU<GPUUsage>::get_timestamp_group_range_host(
   int dense_node_id,
   size_t group_idx,
   bool forward,
   bool is_directed) const {

   const auto& group_offsets = (is_directed && !forward) ?
       this->inbound_timestamp_group_offsets : this->outbound_timestamp_group_offsets;
   const auto& group_indices = (is_directed && !forward) ?
       this->inbound_timestamp_group_indices : this->outbound_timestamp_group_indices;
   const auto& edge_offsets = (is_directed && !forward) ?
       this->inbound_offsets : this->outbound_offsets;

   if (dense_node_id < 0 || dense_node_id >= group_offsets.size() - 1) {
       return SizeRange{0, 0};
   }

   size_t num_groups = group_offsets[dense_node_id + 1] - group_offsets[dense_node_id];
   if (group_idx >= num_groups) {
       return SizeRange{0, 0};
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

   return SizeRange{group_start, group_end};
}

template<GPUUsageMode GPUUsage>
HOST size_t NodeEdgeIndexCPU<GPUUsage>::get_timestamp_group_count_host(
   int dense_node_id,
   bool forward,
   bool directed) const {

   const auto& offsets = get_timestamp_offset_vector_host(forward, directed);

   if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
       return 0;
   }

   return offsets[dense_node_id + 1] - offsets[dense_node_id];
}

template<GPUUsageMode GPUUsage>
[[nodiscard]] HOST typename INodeEdgeIndex<GPUUsage>::SizeVector NodeEdgeIndexCPU<GPUUsage>::get_timestamp_offset_vector_host(
    const bool forward,
    const bool directed) const {
    return (directed && !forward) ? this->inbound_timestamp_group_offsets : this->outbound_timestamp_group_offsets;
}

template class NodeEdgeIndexCPU<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeEdgeIndexCPU<GPUUsageMode::ON_GPU>;
#endif
