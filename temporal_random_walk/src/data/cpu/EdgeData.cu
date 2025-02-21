#include "EdgeData.cuh"
#include <algorithm>
#include <iostream>

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::reserve(size_t size) {
    sources.allocate(size);
    targets.allocate(size);
    timestamps.allocate(size);
    timestamp_group_offsets.allocate(size);  // Estimate for group count
    unique_timestamps.allocate(size);
}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::clear() {
    sources.clear();
    targets.clear();
    timestamps.clear();
    timestamp_group_offsets.clear();
    unique_timestamps.clear();
}

template<GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::size() const {
    return timestamps.size();
}

template<GPUUsageMode GPUUsage>
bool EdgeData<GPUUsage>::empty() const {
    return timestamps.empty();
}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::resize(size_t new_size) {
    sources.resize(new_size);
    targets.resize(new_size);
    timestamps.resize(new_size);
}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::add_edges(int* src, int* tgt, int64_t* ts, size_t size) {
    sources.write_from_pointer(src, size);
    targets.write_from_pointer(tgt, size);
    timestamps.write_from_pointer(ts, size);
}

template<GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> EdgeData<GPUUsage>::get_edges() {
    std::vector<std::tuple<int, int, int64_t>> edges;
    edges.reserve(sources.size());

    for (int i = 0; i < sources.size(); i++) {
        edges.emplace_back(sources[i], targets[i], timestamps[i]);
    }

    return edges;
}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::update_timestamp_groups() {
    if (timestamps.empty()) {
        timestamp_group_offsets.clear();
        unique_timestamps.clear();
        return;
    }

    timestamp_group_offsets.clear();
    unique_timestamps.clear();

    timestamp_group_offsets.push_back(0);
    unique_timestamps.push_back(timestamps[0]);

    for (size_t i = 1; i < timestamps.size(); i++) {
        if (timestamps[i] != timestamps[i-1]) {
            timestamp_group_offsets.push_back(i);
            unique_timestamps.push_back(timestamps[i]);
        }
    }
    timestamp_group_offsets.push_back(timestamps.size());
}

template<GPUUsageMode GPUUsage>
void EdgeData<GPUUsage>::update_temporal_weights(const double timescale_bound) {
    if (timestamps.empty()) {
        forward_cumulative_weights_exponential.clear();
        backward_cumulative_weights_exponential.clear();
        return;
    }

    const int64_t min_timestamp = timestamps[0];
    const int64_t max_timestamp = timestamps.back();
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count();
    forward_cumulative_weights_exponential.resize(num_groups);
    backward_cumulative_weights_exponential.resize(num_groups);

    double forward_sum = 0.0, backward_sum = 0.0;

    // First calculate all weights and total sums
    for (size_t group = 0; group < num_groups; group++) {
        const size_t start = timestamp_group_offsets[group];
        const int64_t group_timestamp = timestamps[start];

        const auto time_diff_forward = static_cast<double>(max_timestamp - group_timestamp);
        const auto time_diff_backward = static_cast<double>(group_timestamp - min_timestamp);

        const double forward_scaled = timescale_bound > 0 ?
            time_diff_forward * time_scale : time_diff_forward;
        const double backward_scaled = timescale_bound > 0 ?
            time_diff_backward * time_scale : time_diff_backward;

        const double forward_weight = exp(forward_scaled);
        const double backward_weight = exp(backward_scaled);

        forward_sum += forward_weight;
        backward_sum += backward_weight;

        forward_cumulative_weights_exponential[group] = forward_weight;
        backward_cumulative_weights_exponential[group] = backward_weight;
    }

    // Then normalize and compute cumulative sums
    double forward_cumsum = 0.0, backward_cumsum = 0.0;
    for (size_t group = 0; group < num_groups; group++) {
        forward_cumulative_weights_exponential[group] /= forward_sum;
        backward_cumulative_weights_exponential[group] /= backward_sum;

        // Update with cumulative sums
        forward_cumsum += forward_cumulative_weights_exponential[group];
        backward_cumsum += backward_cumulative_weights_exponential[group];

        forward_cumulative_weights_exponential[group] = forward_cumsum;
        backward_cumulative_weights_exponential[group] = backward_cumsum;
    }
}

template<GPUUsageMode GPUUsage>
std::pair<size_t, size_t> EdgeData<GPUUsage>::get_timestamp_group_range(size_t group_idx) const {
    if (group_idx >= unique_timestamps.size()) {
        return {0, 0};
    }
    return {timestamp_group_offsets[group_idx], timestamp_group_offsets[group_idx + 1]};
}

template<GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::get_timestamp_group_count() const {
    return unique_timestamps.size();
}

template<GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_after_timestamp(int64_t timestamp) const {
    if (unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::upper_bound directly
    const int64_t* begin = unique_timestamps.data;
    const int64_t* end = unique_timestamps.data + unique_timestamps.size();

    const auto it = std::upper_bound(begin, end, timestamp);
    return it - begin;
}

template<GPUUsageMode GPUUsage>
size_t EdgeData<GPUUsage>::find_group_before_timestamp(int64_t timestamp) const {
    if (unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::lower_bound directly
    const int64_t* begin = unique_timestamps.data;
    const int64_t* end = unique_timestamps.data + unique_timestamps.size();

    auto it = std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

template class EdgeData<GPUUsageMode::ON_CPU>;
