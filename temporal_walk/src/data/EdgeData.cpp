#include "EdgeData.h"
#include <algorithm>
#include <iostream>

void EdgeData::reserve(size_t size) {
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

void EdgeData::resize(size_t new_size) {
    sources.resize(new_size);
    targets.resize(new_size);
    timestamps.resize(new_size);
}

void EdgeData::push_back(int src, int tgt, int64_t ts) {
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

void EdgeData::create_alias_table(
    const std::vector<double>& weights,
    std::vector<double>& probs,
    std::vector<int>& alias) {

    const size_t n = weights.size();
    if (n == 0) {
        probs.clear();
        alias.clear();
        return;
    }

    // Resize output vectors
    probs.resize(n);
    alias.resize(n);

    // Initialize probability and alias tables
    std::vector<int> small, large;
    small.reserve(n);
    large.reserve(n);

    // Input weights should already be normalized probabilities
    // Scale them to n for alias table construction
    for (size_t i = 0; i < n; i++) {
        probs[i] = weights[i] * static_cast<double>(n);
        if (probs[i] < 1.0) {
            small.push_back(static_cast<int>(i));
        } else {
            large.push_back(static_cast<int>(i));
        }
    }

    // Create alias pairs
    while (!small.empty() && !large.empty()) {
        const int s = small.back();
        int l = large.back();
        small.pop_back();
        large.pop_back();

        alias[s] = l;
        probs[l] = (probs[l] + probs[s]) - 1.0;

        if (probs[l] < 1.0) {
            small.push_back(l);
        } else {
            large.push_back(l);
        }
    }

    // Handle remaining bins
    while (!large.empty()) {
        probs[large.back()] = 1.0;
        large.pop_back();
    }
    while (!small.empty()) {
        probs[small.back()] = 1.0;
        small.pop_back();
    }
}

void EdgeData::update_temporal_weights() {
    if (timestamps.empty()) {
        forward_ts_prob.clear();
        forward_ts_alias.clear();
        backward_ts_prob.clear();
        backward_ts_alias.clear();
        return;
    }

    const int64_t t_min = timestamps.front();
    const int64_t t_max = timestamps.back();
    const size_t num_groups = get_timestamp_group_count();

    // Forward weights computation
    {
        std::vector<double> forward_exp(num_groups);
        double forward_sum = 0.0;

        // First compute all exponential and their sum
        for (size_t i = 0; i < num_groups; i++) {
            const int64_t group_ts = timestamps[timestamp_group_offsets[i]];
            forward_exp[i] = std::exp(static_cast<double>(group_ts - t_min));
            forward_sum += forward_exp[i];
        }

        // Then normalize to get probabilities
        std::vector<double> forward_weights(num_groups);
        for (size_t i = 0; i < num_groups; i++) {
            forward_weights[i] = forward_exp[i] / forward_sum;
        }

        // Create alias table
        create_alias_table(forward_weights, forward_ts_prob, forward_ts_alias);
    }

    // Backward weights computation
    {
        std::vector<double> backward_exp(num_groups);
        double backward_sum = 0.0;

        // First compute all exponential and their sum
        for (size_t i = 0; i < num_groups; i++) {
            const int64_t group_ts = timestamps[timestamp_group_offsets[i]];
            backward_exp[i] = std::exp(static_cast<double>(t_max - group_ts));
            backward_sum += backward_exp[i];
        }

        // Then normalize to get probabilities
        std::vector<double> backward_weights(num_groups);
        for (size_t i = 0; i < num_groups; i++) {
            backward_weights[i] = backward_exp[i] / backward_sum;
        }

        // Create alias table
        create_alias_table(backward_weights, backward_ts_prob, backward_ts_alias);
    }
}

std::pair<size_t, size_t> EdgeData::get_timestamp_group_range(size_t group_idx) const {
    if (group_idx >= unique_timestamps.size()) {
        return {0, 0};
    }
    return {timestamp_group_offsets[group_idx], timestamp_group_offsets[group_idx + 1]};
}

size_t EdgeData::get_timestamp_group_count() const {
    return unique_timestamps.size();
}

size_t EdgeData::find_group_after_timestamp(int64_t timestamp) const {
    if (unique_timestamps.empty()) return 0;

    auto it = std::upper_bound(unique_timestamps.begin(), unique_timestamps.end(), timestamp);
    return it - unique_timestamps.begin();
}

size_t EdgeData::find_group_before_timestamp(int64_t timestamp) const {
    if (unique_timestamps.empty()) return 0;

    auto it = std::lower_bound(unique_timestamps.begin(), unique_timestamps.end(), timestamp);
    return (it - unique_timestamps.begin()) - 1;
}
