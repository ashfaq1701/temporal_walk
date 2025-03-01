#include "EdgeDataCPU.cuh"
#include <algorithm>
#include <iostream>

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::reserve_host(size_t size) {
    this->sources.resize(size);
    this->targets.reserve(size);
    this->timestamps.reserve(size);
    this->timestamp_group_offsets.reserve(size);  // Estimate for group count
    this->unique_timestamps.reserve(size);
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::clear_host() {
    this->sources.clear();
    this->targets.clear();
    this->timestamps.clear();
    this->timestamp_group_offsets.clear();
    this->unique_timestamps.clear();
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCPU<GPUUsage>::size_host() const {
    return this->timestamps.size();
}

template<GPUUsageMode GPUUsage>
HOST bool EdgeDataCPU<GPUUsage>::empty_host() const {
    return this->timestamps.empty();
}

template<GPUUsageMode GPUUsage>
void EdgeDataCPU<GPUUsage>::resize_host(size_t new_size) {
    this->sources.resize(new_size);
    this->targets.resize(new_size);
    this->timestamps.resize(new_size);
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::add_edges_host(int* src, int* tgt, int64_t* ts, size_t size) {
    this->sources.insert(this->sources.end(), src, src + size);
    this->targets.insert(this->targets.end(), tgt, tgt + size);
    this->timestamps.insert(this->timestamps.end(), ts, ts + size);
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::push_back_host(int src, int tgt, int64_t ts) {
    this->sources.push_back(src);
    this->targets.push_back(tgt);
    this->timestamps.push_back(ts);
}


template<GPUUsageMode GPUUsage>
HOST typename IEdgeData<GPUUsage>::EdgeVector EdgeDataCPU<GPUUsage>::get_edges_host() {
    typename IEdgeData<GPUUsage>::EdgeVector accumulated_edges;
    accumulated_edges.reserve(this->sources.size());

    for (int i = 0; i < this->sources.size(); i++) {
        accumulated_edges.push_back(Edge(this->sources[i], this->targets[i], this->timestamps[i]));
    }

    return accumulated_edges;
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::update_timestamp_groups_host() {
    if (this->timestamps.empty()) {
        this->timestamp_group_offsets.clear();
        this->unique_timestamps.clear();
        return;
    }

    this->timestamp_group_offsets.clear();
    this->unique_timestamps.clear();

    this->timestamp_group_offsets.push_back(0);
    this->unique_timestamps.push_back(this->timestamps[0]);

    for (size_t i = 1; i < this->timestamps.size(); i++) {
        if (this->timestamps[i] != this->timestamps[i-1]) {
            this->timestamp_group_offsets.push_back(i);
            this->unique_timestamps.push_back(this->timestamps[i]);
        }
    }
    this->timestamp_group_offsets.push_back(this->timestamps.size());
}

template<GPUUsageMode GPUUsage>
HOST void EdgeDataCPU<GPUUsage>::update_temporal_weights_host(const double timescale_bound) {
    if (this->timestamps.empty()) {
        this->forward_cumulative_weights_exponential.clear();
        this->backward_cumulative_weights_exponential.clear();
        return;
    }

    const int64_t min_timestamp = this->timestamps[0];
    const int64_t max_timestamp = this->timestamps.back();
    const auto time_diff = static_cast<double>(max_timestamp - min_timestamp);
    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
        timescale_bound / time_diff : 1.0;

    const size_t num_groups = this->get_timestamp_group_count_host();
    this->forward_cumulative_weights_exponential.resize(num_groups);
    this->backward_cumulative_weights_exponential.resize(num_groups);

    double forward_sum = 0.0, backward_sum = 0.0;

    // First calculate all weights and total sums
    for (size_t group = 0; group < num_groups; group++) {
        const size_t start = this->timestamp_group_offsets[group];
        const int64_t group_timestamp = this->timestamps[start];

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

        this->forward_cumulative_weights_exponential[group] = forward_weight;
        this->backward_cumulative_weights_exponential[group] = backward_weight;
    }

    // Then normalize and compute cumulative sums
    double forward_cumsum = 0.0, backward_cumsum = 0.0;
    for (size_t group = 0; group < num_groups; group++) {
        this->forward_cumulative_weights_exponential[group] /= forward_sum;
        this->backward_cumulative_weights_exponential[group] /= backward_sum;

        // Update with cumulative sums
        forward_cumsum += this->forward_cumulative_weights_exponential[group];
        backward_cumsum += this->backward_cumulative_weights_exponential[group];

        this->forward_cumulative_weights_exponential[group] = forward_cumsum;
        this->backward_cumulative_weights_exponential[group] = backward_cumsum;
    }
}

template<GPUUsageMode GPUUsage>
HOST SizeRange EdgeDataCPU<GPUUsage>::get_timestamp_group_range_host(size_t group_idx) const {
    if (group_idx >= this->unique_timestamps.size()) {
        return SizeRange{0, 0};
    }
    return SizeRange{this->timestamp_group_offsets[group_idx], this->timestamp_group_offsets[group_idx + 1]};
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCPU<GPUUsage>::get_timestamp_group_count_host() const {
    return this->unique_timestamps.size();
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCPU<GPUUsage>::find_group_after_timestamp_host(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::upper_bound directly
    const int64_t* begin = this->unique_timestamps.data();
    const int64_t* end = this->unique_timestamps.data() + this->unique_timestamps.size();

    const auto it = std::upper_bound(begin, end, timestamp);
    return it - begin;
}

template<GPUUsageMode GPUUsage>
HOST size_t EdgeDataCPU<GPUUsage>::find_group_before_timestamp_host(int64_t timestamp) const {
    if (this->unique_timestamps.empty()) return 0;

    // Get raw pointer to data and use std::lower_bound directly
    const int64_t* begin = this->unique_timestamps.data();
    const int64_t* end = this->unique_timestamps.data() + this->unique_timestamps.size();

    auto it = std::lower_bound(begin, end, timestamp);
    return (it - begin) - 1;
}

template class EdgeDataCPU<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class EdgeDataCPU<GPUUsageMode::ON_GPU>;
#endif