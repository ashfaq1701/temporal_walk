#include "EdgeDataThrust.cuh"

#ifdef HAS_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>

template <GPUUsageMode GPUUsage>
void EdgeDataThrust<GPUUsage>::update_timestamp_groups() {
    if (this->timestamps.empty()) {
        this->timestamp_group_offsets.clear();
        this->unique_timestamps.clear();
        return;
    }

    const size_t n = this->timestamps.size();

    // Create a temporary vector for flags where timestamps change
    typename SelectVectorType<int, GPUUsage>::type flags(n);

    thrust::transform(
        this->get_policy(),
        this->timestamps.begin() + 1,
        this->timestamps.end(),
        this->timestamps.begin(),
        flags.begin() + 1,
        [] __host__ __device__ (const int64_t curr, const int64_t prev) { return curr != prev ? 1 : 0; });

    // First element is always a group start
    thrust::fill_n(flags.begin(), 1, 1);

    // Count total groups (sum of flags)
    size_t num_groups = thrust::reduce(flags.begin(), flags.end());

    // Resize output vectors
    this->timestamp_group_offsets.resize(num_groups + 1);  // +1 for end offset
    this->unique_timestamps.resize(num_groups);

    // Find positions of group boundaries
    thrust::copy_if(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(n),
        flags.begin(),
        this->timestamp_group_offsets.begin(),
        [] __host__ __device__ (const int flag) { return flag == 1; });

    // Add final offset
    thrust::fill_n(this->timestamp_group_offsets.begin() + num_groups, 1, n);

    // Get unique timestamps at group boundaries
    thrust::copy_if(
        this->get_policy(),
        this->timestamps.begin(),
        this->timestamps.end(),
        flags.begin(),
        this->unique_timestamps.begin(),
        [] __host__ __device__ (const int flag) { return flag == 1; });
}

template <GPUUsageMode GPUUsage>
void EdgeDataThrust<GPUUsage>::update_temporal_weights(double timescale_bound) {
    if (this->timestamps.empty()) {
        this->forward_cumulative_weights_exponential.clear();
        this->backward_cumulative_weights_exponential.clear();
        return;
    }

    const int64_t min_timestamp = this->timestamps[0];
    const int64_t max_timestamp = this->timestamps[this->timestamps.size() - 1];

    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = this->get_timestamp_group_count();
    this->forward_cumulative_weights_exponential.resize(num_groups);
    this->backward_cumulative_weights_exponential.resize(num_groups);

    typename SelectVectorType<double, GPUUsage>::type forward_weights(num_groups);
    typename SelectVectorType<double, GPUUsage>::type backward_weights(num_groups);

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->timestamps.data());
    const size_t* offsets_ptr = thrust::raw_pointer_cast(this->timestamp_group_offsets.data());

    // Calculate weights using device pointers
    thrust::transform(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(num_groups),
        thrust::make_zip_iterator(thrust::make_tuple(
            forward_weights.begin(),
            backward_weights.begin()
        )),
        [offsets_ptr, timestamps_ptr, max_timestamp, min_timestamp, timescale_bound, time_scale]
        __host__ __device__ (const size_t group) {
            const size_t start = offsets_ptr[group];
            const int64_t group_timestamp = timestamps_ptr[start];

            const auto time_diff_forward = static_cast<double>(max_timestamp - group_timestamp);
            const auto time_diff_backward = static_cast<double>(group_timestamp - min_timestamp);

            const double forward_scaled = timescale_bound > 0 ?
                time_diff_forward * time_scale : time_diff_forward;
            const double backward_scaled = timescale_bound > 0 ?
                time_diff_backward * time_scale : time_diff_backward;

            return thrust::make_tuple(exp(forward_scaled), exp(backward_scaled));
        }
    );

    // Calculate sums
    const double forward_sum = thrust::reduce(
        this->get_policy(),
        forward_weights.begin(),
        forward_weights.end()
    );
    const double backward_sum = thrust::reduce(
        this->get_policy(),
        backward_weights.begin(),
        backward_weights.end()
    );

    // Normalize weights
    thrust::transform(
        this->get_policy(),
        forward_weights.begin(),
        forward_weights.end(),
        forward_weights.begin(),
        [=] __host__ __device__ (const double w) { return w / forward_sum; }
    );
    thrust::transform(
        this->get_policy(),
        backward_weights.begin(),
        backward_weights.end(),
        backward_weights.begin(),
        [=] __host__ __device__ (const double w) { return w / backward_sum; }
    );

    // Compute cumulative sums
    thrust::inclusive_scan(
        this->get_policy(),
        forward_weights.begin(),
        forward_weights.end(),
        this->forward_cumulative_weights_exponential.begin()
    );
    thrust::inclusive_scan(
        this->get_policy(),
        backward_weights.begin(),
        backward_weights.end(),
        this->backward_cumulative_weights_exponential.begin()
    );
}

template<GPUUsageMode GPUUsage>
size_t EdgeDataThrust<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    auto it = thrust::upper_bound(
        this->get_policy(),
        this->unique_timestamps.begin(),
        this->unique_timestamps.end(),
        timestamp
    );
    return it - this->unique_timestamps.begin();
}

template<GPUUsageMode GPUUsage>
size_t EdgeDataThrust<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    auto it = thrust::lower_bound(
        this->get_policy(),
        this->unique_timestamps.begin(),
        this->unique_timestamps.end(),
        timestamp
    );
    return (it - this->unique_timestamps.begin()) - 1;
}

template<GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> EdgeDataThrust<GPUUsage>::get_edges() {
    std::vector<std::tuple<int, int, int64_t>> edges;
    edges.reserve(this->sources.size());

    if constexpr (GPUUsage == GPUUsageMode::ON_GPU_USING_CUDA) {
        // Copy data from device to host
        thrust::host_vector<int> h_sources = this->sources;
        thrust::host_vector<int> h_targets = this->targets;
        thrust::host_vector<int64_t> h_timestamps = this->timestamps;

        for (int i = 0; i < h_sources.size(); i++) {
            edges.emplace_back(h_sources[i], h_targets[i], h_timestamps[i]);
        }
    } else {
        for (int i = 0; i < this->sources.size(); i++) {
            edges.emplace_back(this->sources[i], this->targets[i], this->timestamps[i]);
        }
    }

    return edges;
}

template class EdgeDataThrust<GPUUsageMode::ON_GPU_USING_CUDA>;
template class EdgeDataThrust<GPUUsageMode::ON_HOST_USING_THRUST>;
#endif
