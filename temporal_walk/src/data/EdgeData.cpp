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

void EdgeData::update_temporal_weights() {

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
