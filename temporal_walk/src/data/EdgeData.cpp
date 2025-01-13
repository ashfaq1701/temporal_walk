#include "EdgeData.h"

void EdgeData::reserve(size_t size) {
    sources.reserve(size);
    targets.reserve(size);
    timestamps.reserve(size);
    timestamp_group_offsets.reserve(size / 4);  // Estimate: fewer groups than edges
}

void EdgeData::clear() {
    sources.clear();
    targets.clear();
    timestamps.clear();
    timestamp_group_offsets.clear();
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

void EdgeData::update_timestamp_groups() {
    timestamp_group_offsets.clear();
    if (timestamps.empty()) return;

    timestamp_group_offsets.push_back(0);  // First group starts at 0

    for (size_t i = 1; i < timestamps.size(); i++) {
        if (timestamps[i] != timestamps[i-1]) {
            timestamp_group_offsets.push_back(i);
        }
    }
    timestamp_group_offsets.push_back(timestamps.size());  // End marker
}

std::pair<size_t, size_t> EdgeData::get_timestamp_group_range(size_t group_idx) const {
    if (group_idx >= timestamp_group_offsets.size() - 1) {
        return {0, 0};  // Invalid group
    }
    return {timestamp_group_offsets[group_idx], timestamp_group_offsets[group_idx + 1]};
}

size_t EdgeData::get_timestamp_group_count() const {
    return timestamp_group_offsets.empty() ? 0 : timestamp_group_offsets.size() - 1;
}
