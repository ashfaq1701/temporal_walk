#include "EdgeData.cuh"
#include <algorithm>
#include <iostream>

EdgeData::EdgeData(const bool use_gpu):
    use_gpu(use_gpu), sources(use_gpu), targets(use_gpu), timestamps(use_gpu),
    timestamp_group_offsets(use_gpu), unique_timestamps(use_gpu),
    forward_weights_exponential(use_gpu), backward_weights_exponential(use_gpu) {}

void EdgeData::reserve(const size_t size) {
    sources.reserve(size);
    targets.reserve(size);
    timestamps.reserve(size);
    timestamp_group_offsets.reserve(size/4 + 1);  // Estimate for group count
    unique_timestamps.reserve(size/4);
}

void EdgeData::clear() {
    sources.clear();
    targets.clear();
    timestamps.clear();
    timestamp_group_offsets.clear();
    unique_timestamps.clear();
}

size_t EdgeData::size() const {
    return timestamps.size();
}

bool EdgeData::empty() const {
    return timestamps.empty();
}

void EdgeData::resize(const size_t new_size) {
    sources.resize(new_size);
    targets.resize(new_size);
    timestamps.resize(new_size);
}

void EdgeData::push_back(const int src, const int tgt, const int64_t ts) {
    sources.push_back(src);
    targets.push_back(tgt);
    timestamps.push_back(ts);
}

std::vector<std::tuple<int, int, int64_t>> EdgeData::get_edges() {
    std::vector<std::tuple<int, int, int64_t>> edges;
    edges.reserve(sources.size());

    for (int i = 0; i < sources.size(); i++) {
        edges.emplace_back(sources[i], targets[i], timestamps[i]);
    }

    return edges;
}

void EdgeData::update_timestamp_groups() {
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

void EdgeData::update_temporal_weights(const double timescale_bound) {
    if (timestamps.empty()) {
        forward_weights_exponential.clear();
        backward_weights_exponential.clear();
        return;
    }

    const int64_t min_timestamp = timestamps[0];
    const int64_t max_timestamp = timestamps.back();
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = get_timestamp_group_count();
    forward_weights_exponential.resize(num_groups);
    backward_weights_exponential.resize(num_groups);

    double forward_sum = 0.0, backward_sum = 0.0;
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
        forward_weights_exponential[group] = forward_weight;
        forward_sum += forward_weight;

        const double backward_weight = exp(backward_scaled);
        backward_weights_exponential[group] = backward_weight;
        backward_sum += backward_weight;
    }

    for (size_t group = 0; group < num_groups; group++) {
        forward_weights_exponential[group] /= forward_sum;
        backward_weights_exponential[group] /= backward_sum;
    }
}

std::pair<size_t, size_t> EdgeData::get_timestamp_group_range(const size_t group_idx) const {
    if (group_idx >= unique_timestamps.size()) {
        return {0, 0};
    }
    return {timestamp_group_offsets[group_idx], timestamp_group_offsets[group_idx + 1]};
}

size_t EdgeData::get_timestamp_group_count() const {
    return unique_timestamps.size();
}

size_t EdgeData::find_group_after_timestamp(const int64_t timestamp) const {
    if (unique_timestamps.empty()) return 0;

    if (use_gpu) {
        #ifdef HAS_CUDA
        auto it = thrust::upper_bound(thrust::device,
                                    unique_timestamps.device_begin(),
                                    unique_timestamps.device_end(),
                                    timestamp);
        return thrust::distance(unique_timestamps.device_begin(), it);
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    } else {
        auto it = std::upper_bound(unique_timestamps.host_begin(),
                                 unique_timestamps.host_end(),
                                 timestamp);
        return std::distance(unique_timestamps.host_begin(), it);
    }
}

size_t EdgeData::find_group_before_timestamp(const int64_t timestamp) const {
    if (unique_timestamps.empty()) return 0;

    if (use_gpu) {
        #ifdef HAS_CUDA
        auto it = thrust::lower_bound(thrust::device,
                                    unique_timestamps.device_begin(),
                                    unique_timestamps.device_end(),
                                    timestamp);
        return thrust::distance(unique_timestamps.device_begin(), it) - 1;
        #else
        throw std::runtime_error("GPU support not compiled in");
        #endif
    } else {
        auto it = std::lower_bound(unique_timestamps.host_begin(),
                                 unique_timestamps.host_end(),
                                 timestamp);
        return std::distance(unique_timestamps.host_begin(), it) - 1;
    }
}
