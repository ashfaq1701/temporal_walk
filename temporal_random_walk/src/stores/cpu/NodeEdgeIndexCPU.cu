#include "NodeEdgeIndexCPU.cuh"

#include <iostream>
#include <numeric>

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::resize_node_offset_and_indices_vectors(
    size_t num_nodes,
    size_t num_edges,
    bool is_directed)
{
    // Initialize base CSR structures
    this->outbound_offsets.resize(num_nodes + 1);
    this->outbound_timestamp_group_offsets.resize(num_nodes + 1);

    if (is_directed) {
        this->inbound_offsets.resize(num_nodes + 1);
        this->inbound_timestamp_group_offsets.resize(num_nodes + 1);
    }

    this->outbound_indices.resize(num_edges);
    if (is_directed)
    {
        this->inbound_indices.resize(num_edges);
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_dense_vectors_host(
    const IEdgeData<GPUUsage>* edges,
    const INodeMapping<GPUUsage>* mapping,
    typename INodeEdgeIndex<GPUUsage>::IntVector& source_dense_ids,
    typename INodeEdgeIndex<GPUUsage>::IntVector& target_dense_ids)
{
    for (size_t i = 0; i < edges->size_host(); ++i)
    {
        source_dense_ids[i] = mapping->to_dense_host(edges->sources[i]);
        target_dense_ids[i] = mapping->to_dense_host(edges->targets[i]);
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_offsets_and_indices_host(
    const IEdgeData<GPUUsage>* edges,
    typename INodeEdgeIndex<GPUUsage>::IntVector& source_dense_ids,
    typename INodeEdgeIndex<GPUUsage>::IntVector& target_dense_ids,
    const bool is_directed,
    const size_t num_nodes,
    const size_t num_edges) {

    for (size_t i = 0; i < edges->size_host(); i++) {
        const int src_idx = source_dense_ids[i];
        const int tgt_idx = target_dense_ids[i];

        ++this->outbound_offsets[src_idx + 1];
        if (is_directed) {
            ++this->inbound_offsets[tgt_idx + 1];
        } else {
            ++this->outbound_offsets[tgt_idx + 1];
        }
    }

    for (size_t i = 1; i <= num_nodes; i++) {
        this->outbound_offsets[i] += this->outbound_offsets[i-1];
        if (is_directed) {
            this->inbound_offsets[i] += this->inbound_offsets[i-1];
        }
    }

    std::iota(this->outbound_indices.begin(), this->outbound_indices.begin() + num_edges, 0);
    if (is_directed) {
        std::iota(this->inbound_indices.begin(), this->inbound_indices.begin() + num_edges, 0);
    }

    std::sort(this->outbound_indices.begin(), this->outbound_indices.begin() + num_edges,
              [&](size_t a, size_t b) {
                  return source_dense_ids[a] < source_dense_ids[b];
              });

    if (is_directed) {
        std::sort(this->inbound_indices.begin(), this->inbound_indices.begin() + num_edges,
                  [&](size_t a, size_t b) {
                      return target_dense_ids[a] < target_dense_ids[b];
                  });
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_edge_timestamp_group_offsets_host(
    const IEdgeData<GPUUsage>* edges,
    bool is_directed,
    size_t num_nodes,
    typename INodeEdgeIndex<GPUUsage>::SizeVector& outbound_node_timestamp_group_count,
    typename INodeEdgeIndex<GPUUsage>::SizeVector& inbound_node_timestamp_group_count)
{
    for (size_t node = 0; node < num_nodes; node++) {
        size_t start = this->outbound_offsets[node];
        size_t end = this->outbound_offsets[node + 1];

        if (start < end) {
            outbound_node_timestamp_group_count[node] = 1;  // First group
            for (size_t i = start + 1; i < end; ++i) {
                if (edges->timestamps[this->outbound_indices[i]] != edges->timestamps[this->outbound_indices[i-1]]) {
                    ++outbound_node_timestamp_group_count[node];
                }
            }
        }

        if (is_directed) {
            start = this->inbound_offsets[node];
            end = this->inbound_offsets[node + 1];

            if (start < end) {
                inbound_node_timestamp_group_count[node] = 1;  // First group
                for (size_t i = start + 1; i < end; ++i) {
                    if (edges->timestamps[this->inbound_indices[i]] !=
                        edges->timestamps[this->inbound_indices[i-1]])
                    {
                        ++inbound_node_timestamp_group_count[node];
                    }
                }
            }
        }
    }

    // Calculate prefix sums for group offsets
    for (size_t i = 0; i < num_nodes; i++) {
        this->outbound_timestamp_group_offsets[i + 1] = this->outbound_timestamp_group_offsets[i] + outbound_node_timestamp_group_count[i];
        if (is_directed) {
            this->inbound_timestamp_group_offsets[i + 1] = this->inbound_timestamp_group_offsets[i] + inbound_node_timestamp_group_count[i];
        }
    }
}

template<GPUUsageMode GPUUsage>
void NodeEdgeIndexCPU<GPUUsage>::resize_node_timestamp_group_indices(bool is_directed)
{
    // Allocate and fill group indices
    this->outbound_timestamp_group_indices.resize(this->outbound_timestamp_group_offsets.back());
    if (is_directed) {
        this->inbound_timestamp_group_indices.resize(this->inbound_timestamp_group_offsets.back());
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_node_edge_timestamp_group_indices_host(
    const IEdgeData<GPUUsage>* edges,
    bool is_directed,
    size_t num_nodes)
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

    typename INodeEdgeIndex<GPUUsage>::IntVector source_dense_ids(num_edges);
    typename INodeEdgeIndex<GPUUsage>::IntVector target_dense_ids(num_edges);

    compute_dense_vectors_host(edges, mapping, source_dense_ids, target_dense_ids);
    resize_node_offset_and_indices_vectors(num_nodes, num_edges, is_directed);
    compute_node_offsets_and_indices_host(
        edges,
        source_dense_ids,
        target_dense_ids,
        is_directed,
        num_nodes,
        num_edges);

    // Third pass: count timestamp groups
    typename INodeEdgeIndex<GPUUsage>::SizeVector outbound_node_timestamp_group_count(num_nodes);
    typename INodeEdgeIndex<GPUUsage>::SizeVector inbound_node_timestamp_group_count;
    if (is_directed) {
        inbound_node_timestamp_group_count.resize(num_nodes);
    }

    compute_node_edge_timestamp_group_offsets_host(
        edges,
        is_directed,
        num_nodes,
        outbound_node_timestamp_group_count,
        inbound_node_timestamp_group_count);

    resize_node_timestamp_group_indices(is_directed);
    compute_node_edge_timestamp_group_indices_host(edges, is_directed, num_nodes);
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::resize_weight_vectors()
{
    this->outbound_forward_cumulative_weights_exponential.resize(this->outbound_timestamp_group_indices.size());
    this->outbound_backward_cumulative_weights_exponential.resize(this->outbound_timestamp_group_indices.size());
    if (!this->inbound_offsets.empty()) {
        this->inbound_backward_cumulative_weights_exponential.resize(this->inbound_timestamp_group_indices.size());
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCPU<GPUUsage>::compute_temporal_weights_host(const IEdgeData<GPUUsage>* edges, double timescale_bound)
{
    const size_t num_nodes = this->outbound_offsets.size() - 1;

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
    resize_weight_vectors();
    compute_temporal_weights_host(edges, timescale_bound);
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
