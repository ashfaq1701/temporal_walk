#include "NodeEdgeIndexCUDA.cuh"

#include "../../cuda_common/cuda_config.cuh"

#ifdef HAS_CUDA

template<GPUUsageMode GPUUsage>
HOST void NodeEdgeIndexCUDA<GPUUsage>::compute_temporal_weights(
    const IEdgeData<GPUUsage>* edges,
    double timescale_bound,
    size_t num_nodes) {

    // Process outbound weights
    {
        const auto& outbound_offsets = this->get_timestamp_offset_vector(true, false);
        typename SelectVectorType<double, GPUUsage>::type forward_weights(this->outbound_timestamp_group_indices.size());
        typename SelectVectorType<double, GPUUsage>::type backward_weights(this->outbound_timestamp_group_indices.size());

        auto timestamps_ptr = thrust::raw_pointer_cast(edges->timestamps.data());
        auto indices_ptr = thrust::raw_pointer_cast(this->outbound_indices.data());
        auto group_indices_ptr = thrust::raw_pointer_cast(this->outbound_timestamp_group_indices.data());
        auto offsets_ptr = thrust::raw_pointer_cast(outbound_offsets.data());
        auto forward_weights_ptr = thrust::raw_pointer_cast(forward_weights.data());
        auto backward_weights_ptr = thrust::raw_pointer_cast(backward_weights.data());

        // Calculate initial weights
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
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

        auto timestamps_ptr = thrust::raw_pointer_cast(edges->timestamps.data());
        auto indices_ptr = thrust::raw_pointer_cast(this->inbound_indices.data());
        auto group_indices_ptr = thrust::raw_pointer_cast(this->inbound_timestamp_group_indices.data());
        auto offsets_ptr = thrust::raw_pointer_cast(inbound_offsets.data());
        auto weights_ptr = thrust::raw_pointer_cast(backward_weights.data());

        // Calculate weights
        thrust::for_each(
            DEVICE_EXECUTION_POLICY,
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

template class NodeEdgeIndexCUDA<GPUUsageMode::ON_GPU>;
#endif
