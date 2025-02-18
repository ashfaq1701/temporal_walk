#include "NodeEdgeIndexCUDA.cuh"
#include "NodeMappingCUDA.cuh"

#ifdef HAS_CUDA

#include <thrust/gather.h>
#include <thrust/sort.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/scan.h>
#include <thrust/transform.h>

template<GPUUsageMode GPUUsage>
void NodeEdgeIndexCUDA<GPUUsage>::rebuild(
    const EdgeData<GPUUsage>& edges,
    const NodeMapping<GPUUsage>& mapping,
    const bool is_directed)
{
    const size_t num_nodes = mapping.size();
    const size_t num_edges = edges.size();

    // Initialize base CSR structures
    this->outbound_offsets.assign(num_nodes + 1, 0);
    this->outbound_timestamp_group_offsets.assign(num_nodes + 1, 0);

    if (is_directed) {
        this->inbound_offsets.assign(num_nodes + 1, 0);
        this->inbound_timestamp_group_offsets.assign(num_nodes + 1, 0);
    }

    typename SelectVectorType<int, GPUUsage>::type d_src_indices(num_edges);
    typename SelectVectorType<int, GPUUsage>::type d_tgt_indices(num_edges);

    const int* d_sparse_to_dense = thrust::raw_pointer_cast(mapping.sparse_to_dense.data());
    const auto sparse_to_dense_size = mapping.sparse_to_dense.size();

    // Convert sparse node IDs to dense
    thrust::transform(
        this->get_policy(),
        edges.sources.begin(),
        edges.sources.end(),
        d_src_indices.begin(),
        [d_sparse_to_dense, sparse_to_dense_size] __host__ __device__ (const int id) {
            return to_dense(d_sparse_to_dense, id, sparse_to_dense_size);
        });

    thrust::transform(
        this->get_policy(),
        edges.targets.begin(),
        edges.targets.end(),
        d_tgt_indices.begin(),
        [d_sparse_to_dense, sparse_to_dense_size] __host__ __device__ (const int id) {
            return to_dense(d_sparse_to_dense, id, sparse_to_dense_size);
        });

    // Get raw pointers for counter updates
    size_t* d_outbound_ptr = thrust::raw_pointer_cast(this->outbound_offsets.data());
    size_t* d_inbound_ptr = thrust::raw_pointer_cast(this->inbound_offsets.data());
    const int* d_src_ptr = thrust::raw_pointer_cast(d_src_indices.data());
    const int* d_tgt_ptr = thrust::raw_pointer_cast(d_tgt_indices.data());

    // Count edges per node
    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_edges),
        [d_outbound_ptr, d_inbound_ptr, d_src_ptr, d_tgt_ptr, is_directed] __host__ __device__ (const size_t i) {
            const int src_idx = d_src_ptr[i];
            const int tgt_idx = d_tgt_ptr[i];

            ++d_outbound_ptr[src_idx + 1];
            if (is_directed) {
                ++d_inbound_ptr[tgt_idx + 1];
            } else {
                ++d_outbound_ptr[tgt_idx + 1];
            }
        }
    );

    // Calculate prefix sums for edge offsets
    thrust::inclusive_scan(
        this->get_policy(),
        this->outbound_offsets.begin() + 1,
        this->outbound_offsets.end(),
        this->outbound_offsets.begin() + 1
    );

    if (is_directed) {
        thrust::inclusive_scan(
            this->get_policy(),
            this->inbound_offsets.begin() + 1,
            this->inbound_offsets.end(),
            this->inbound_offsets.begin() + 1
        );
    }

    // Allocate edge index arrays
    this->outbound_indices.resize(this->outbound_offsets.back());
    if (is_directed) {
        this->inbound_indices.resize(this->inbound_offsets.back());
    }

    // Create current counters for each node
    typename SelectVectorType<size_t, GPUUsage>::type d_outbound_current(num_nodes, 0);
    typename SelectVectorType<size_t, GPUUsage>::type d_inbound_current;
    if (is_directed) {
        d_inbound_current.resize(num_nodes, 0);
    }

    // Get raw pointers for filling indices
    size_t* d_outbound_current_ptr = thrust::raw_pointer_cast(d_outbound_current.data());
    size_t* d_inbound_current_ptr = is_directed ? thrust::raw_pointer_cast(d_inbound_current.data()) : nullptr;
    size_t* d_outbound_indices_ptr = thrust::raw_pointer_cast(this->outbound_indices.data());
    size_t* d_inbound_indices_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_indices.data()) : nullptr;
    const size_t* d_outbound_offsets_ptr = thrust::raw_pointer_cast(this->outbound_offsets.data());
    const size_t* d_inbound_offsets_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_offsets.data()) : nullptr;

    // Fill edge indices
    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_edges),
        [d_src_ptr, d_tgt_ptr,
            d_outbound_offsets_ptr, d_inbound_offsets_ptr,
            d_outbound_indices_ptr, d_inbound_indices_ptr,
            d_outbound_current_ptr, d_inbound_current_ptr, is_directed] __host__ __device__ (const size_t i) {
            const int src_idx = d_src_ptr[i];
            const int tgt_idx = d_tgt_ptr[i];

            const size_t out_pos = d_outbound_offsets_ptr[src_idx] + d_outbound_current_ptr[src_idx]++;
            d_outbound_indices_ptr[out_pos] = i;

            if (is_directed) {
                const size_t in_pos = d_inbound_offsets_ptr[tgt_idx] + d_inbound_current_ptr[tgt_idx]++;
                d_inbound_indices_ptr[in_pos] = i;
            } else {
                const size_t out_pos2 = d_outbound_offsets_ptr[tgt_idx] + d_outbound_current_ptr[tgt_idx]++;
                d_outbound_indices_ptr[out_pos2] = i;
            }
        }
    );

    // Third pass: count timestamp groups
    typename SelectVectorType<size_t, GPUUsage>::type d_outbound_group_count(num_nodes, 0);
    typename SelectVectorType<size_t, GPUUsage>::type d_inbound_group_count;
    if (is_directed) {
        d_inbound_group_count.resize(num_nodes, 0);
    }

    // Get raw pointers for timestamp comparison
    const int64_t* d_timestamps_ptr = thrust::raw_pointer_cast(edges.timestamps.data());
    size_t* d_outbound_group_count_ptr = thrust::raw_pointer_cast(d_outbound_group_count.data());
    size_t* d_inbound_group_count_ptr = is_directed ? thrust::raw_pointer_cast(d_inbound_group_count.data()) : nullptr;

    // Count timestamp groups for each node
    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_nodes),
        [d_outbound_offsets_ptr, d_inbound_offsets_ptr,
            d_outbound_indices_ptr, d_inbound_indices_ptr,
            d_outbound_group_count_ptr, d_inbound_group_count_ptr,
            d_timestamps_ptr, is_directed] __host__ __device__ (const size_t node) {
            size_t start = d_outbound_offsets_ptr[node];
            size_t end = d_outbound_offsets_ptr[node + 1];

            if (start < end) {
                d_outbound_group_count_ptr[node] = 1;  // First group
                for (size_t i = start + 1; i < end; ++i) {
                    if (d_timestamps_ptr[d_outbound_indices_ptr[i]] !=
                        d_timestamps_ptr[d_outbound_indices_ptr[i-1]]) {
                        ++d_outbound_group_count_ptr[node];
                    }
                }
            }

            if (is_directed) {
                start = d_inbound_offsets_ptr[node];
                end = d_inbound_offsets_ptr[node + 1];

                if (start < end) {
                    d_inbound_group_count_ptr[node] = 1;  // First group
                    for (size_t i = start + 1; i < end; ++i) {
                        if (d_timestamps_ptr[d_inbound_indices_ptr[i]] !=
                            d_timestamps_ptr[d_inbound_indices_ptr[i-1]]) {
                            ++d_inbound_group_count_ptr[node];
                        }
                    }
                }
            }
        }
    );

    // Calculate prefix sums for group offsets
    thrust::inclusive_scan(
        this->get_policy(),
        d_outbound_group_count.begin(),
        d_outbound_group_count.end(),
        thrust::make_permutation_iterator(
            this->outbound_timestamp_group_offsets.begin() + 1,
            thrust::make_counting_iterator<size_t>(0)
        )
    );

    if (is_directed) {
        thrust::inclusive_scan(
            this->get_policy(),
            d_inbound_group_count.begin(),
            d_inbound_group_count.end(),
            thrust::make_permutation_iterator(
                this->inbound_timestamp_group_offsets.begin() + 1,
                thrust::make_counting_iterator<size_t>(0)
            )
        );
    }

    // Allocate and fill group indices
    this->outbound_timestamp_group_indices.resize(this->outbound_timestamp_group_offsets.back());
    if (is_directed) {
        this->inbound_timestamp_group_indices.resize(this->inbound_timestamp_group_offsets.back());
    }

    // Get raw pointers for filling group indices
    size_t* d_outbound_group_indices_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_indices.data());
    size_t* d_inbound_group_indices_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_timestamp_group_indices.data()) : nullptr;
    const size_t* d_outbound_group_offsets_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_offsets.data());
    const size_t* d_inbound_group_offsets_ptr = is_directed ? thrust::raw_pointer_cast(this->inbound_timestamp_group_offsets.data()) : nullptr;

    // Fill group indices
    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_nodes),
        [d_outbound_offsets_ptr, d_inbound_offsets_ptr,
            d_outbound_indices_ptr, d_inbound_indices_ptr,
            d_outbound_group_offsets_ptr, d_inbound_group_offsets_ptr,
            d_outbound_group_indices_ptr, d_inbound_group_indices_ptr,
            d_timestamps_ptr, is_directed] __host__ __device__ (const size_t node) {
            size_t start = d_outbound_offsets_ptr[node];
            size_t end = d_outbound_offsets_ptr[node + 1];
            size_t group_pos = d_outbound_group_offsets_ptr[node];

            if (start < end) {
                d_outbound_group_indices_ptr[group_pos++] = start;
                for (size_t i = start + 1; i < end; ++i) {
                    if (d_timestamps_ptr[d_outbound_indices_ptr[i]] !=
                        d_timestamps_ptr[d_outbound_indices_ptr[i-1]]) {
                        d_outbound_group_indices_ptr[group_pos++] = i;
                    }
                }
            }

            if (is_directed) {
                start = d_inbound_offsets_ptr[node];
                end = d_inbound_offsets_ptr[node + 1];
                group_pos = d_inbound_group_offsets_ptr[node];

                if (start < end) {
                    d_inbound_group_indices_ptr[group_pos++] = start;
                    for (size_t i = start + 1; i < end; ++i) {
                        if (d_timestamps_ptr[d_inbound_indices_ptr[i]] !=
                            d_timestamps_ptr[d_inbound_indices_ptr[i-1]]) {
                            d_inbound_group_indices_ptr[group_pos++] = i;
                        }
                    }
                }
            }
        }
    );
}


template<GPUUsageMode GPUUsage>
void NodeEdgeIndexCUDA<GPUUsage>::update_temporal_weights(
    const EdgeData<GPUUsage>& edges,
    const double timescale_bound)
{
    const size_t num_nodes = this->outbound_offsets.size() - 1;
    if (edges.size() == 0) {
        return;
    }

    // Resize weight vectors
    this->outbound_forward_cumulative_weights_exponential.resize(
        this->outbound_timestamp_group_indices.size());
    this->outbound_backward_cumulative_weights_exponential.resize(
        this->outbound_timestamp_group_indices.size());
    if (!this->inbound_offsets.empty()) {
        this->inbound_backward_cumulative_weights_exponential.resize(
            this->inbound_timestamp_group_indices.size());
    }

    // Process outbound weights
    {
        const auto& outbound_offsets = this->get_timestamp_offset_vector(true, false);
        typename SelectVectorType<double, GPUUsage>::type forward_weights(this->outbound_timestamp_group_indices.size());
        typename SelectVectorType<double, GPUUsage>::type backward_weights(this->outbound_timestamp_group_indices.size());

        auto timestamps_ptr = thrust::raw_pointer_cast(edges.timestamps.data());
        auto indices_ptr = thrust::raw_pointer_cast(this->outbound_indices.data());
        auto group_indices_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_indices.data());
        auto offsets_ptr = thrust::raw_pointer_cast(outbound_offsets.data());
        auto forward_weights_ptr = thrust::raw_pointer_cast(forward_weights.data());
        auto backward_weights_ptr = thrust::raw_pointer_cast(backward_weights.data());

        // Calculate initial weights
        thrust::for_each(
            this->get_policy(),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_nodes),
            [
                timestamps_ptr,
                indices_ptr,
                group_indices_ptr,
                offsets_ptr,
                forward_weights_ptr,
                backward_weights_ptr,
                timescale_bound
            ] __host__ __device__ (size_t node) {
                const size_t out_start = offsets_ptr[node];
                const size_t out_end = offsets_ptr[node + 1];

                if (out_start < out_end) {
                    // Get node's timestamp range
                    const size_t first_group_start = group_indices_ptr[out_start];
                    const size_t last_group_start = group_indices_ptr[out_end - 1];
                    const int64_t min_ts = timestamps_ptr[indices_ptr[first_group_start]];
                    const int64_t max_ts = timestamps_ptr[indices_ptr[last_group_start]];

                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                        timescale_bound / time_diff : 1.0;

                    double forward_sum = 0.0;
                    double backward_sum = 0.0;

                    // Calculate weights for each group
                    for (size_t pos = out_start; pos < out_end; ++pos) {
                        const size_t edge_start = group_indices_ptr[pos];
                        const int64_t group_ts = timestamps_ptr[indices_ptr[edge_start]];

                        const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                        const double forward_scaled = timescale_bound > 0 ?
                            time_diff_forward * time_scale : time_diff_forward;
                        const double backward_scaled = timescale_bound > 0 ?
                            time_diff_backward * time_scale : time_diff_backward;

                        const double forward_weight = exp(forward_scaled);
                        forward_weights_ptr[pos] = forward_weight;
                        forward_sum += forward_weight;

                        const double backward_weight = exp(backward_scaled);
                        backward_weights_ptr[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sums
                    double forward_cumsum = 0.0, backward_cumsum = 0.0;
                    for (size_t pos = out_start; pos < out_end; ++pos) {
                        forward_weights_ptr[pos] /= forward_sum;
                        backward_weights_ptr[pos] /= backward_sum;

                        forward_cumsum += forward_weights_ptr[pos];
                        backward_cumsum += backward_weights_ptr[pos];

                        forward_weights_ptr[pos] = forward_cumsum;
                        backward_weights_ptr[pos] = backward_cumsum;
                    }
                }
            }
        );

        // Copy results back
        this->outbound_forward_cumulative_weights_exponential = forward_weights;
        this->outbound_backward_cumulative_weights_exponential = backward_weights;
    }

    // Process inbound weights if directed
    if (!this->inbound_offsets.empty()) {
        const auto& inbound_offsets = this->get_timestamp_offset_vector(false, true);
        typename SelectVectorType<double, GPUUsage>::type backward_weights(this->inbound_timestamp_group_indices.size());

        auto timestamps_ptr = thrust::raw_pointer_cast(edges.timestamps.data());
        auto indices_ptr = thrust::raw_pointer_cast(this->inbound_indices.data());
        auto group_indices_ptr = thrust::raw_pointer_cast(this->inbound_timestamp_group_indices.data());
        auto offsets_ptr = thrust::raw_pointer_cast(inbound_offsets.data());
        auto weights_ptr = thrust::raw_pointer_cast(backward_weights.data());

        // Calculate weights
        thrust::for_each(
            this->get_policy(),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(num_nodes),
            [
                timestamps_ptr,
                indices_ptr,
                group_indices_ptr,
                offsets_ptr,
                weights_ptr,
                timescale_bound
            ] __host__ __device__ (size_t node) {
                const size_t in_start = offsets_ptr[node];
                const size_t in_end = offsets_ptr[node + 1];

                if (in_start < in_end) {
                    // Get node's timestamp range
                    const size_t first_group_start = group_indices_ptr[in_start];
                    const size_t last_group_start = group_indices_ptr[in_end - 1];
                    const int64_t min_ts = timestamps_ptr[indices_ptr[first_group_start]];
                    const int64_t max_ts = timestamps_ptr[indices_ptr[last_group_start]];

                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                        timescale_bound / time_diff : 1.0;

                    // Calculate weights
                    double backward_sum = 0.0;

                    // Calculate weights and sum in single pass
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        const size_t edge_start = group_indices_ptr[pos];
                        const int64_t group_ts = timestamps_ptr[indices_ptr[edge_start]];

                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                        const double backward_scaled = timescale_bound > 0 ?
                            time_diff_backward * time_scale : time_diff_backward;

                        const double backward_weight = exp(backward_scaled);
                        weights_ptr[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sum
                    double backward_cumsum = 0.0;
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        weights_ptr[pos] /= backward_sum;
                        backward_cumsum += weights_ptr[pos];
                        weights_ptr[pos] = backward_cumsum;
                    }
                }
            }
        );

        // Copy results
        this->inbound_backward_cumulative_weights_exponential = backward_weights;
    }
}

template class NodeEdgeIndexCUDA<GPUUsageMode::DATA_ON_GPU>;
template class NodeEdgeIndexCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
