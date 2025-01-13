#include "TemporalGraph.h"
#include <algorithm>
#include <numeric>


// In implementation file (TemporalGraph.cpp):
TemporalGraph::TemporalGraph(bool directed, int64_t window)
    : is_directed(directed), time_window(window), latest_timestamp(0) {}

void TemporalGraph::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& new_edges) {
    if (new_edges.empty()) return;

    size_t start_idx = edges.size();
    edges.reserve(start_idx + new_edges.size());

    // Add new edges and track max timestamp
    for (const auto& [src, tgt, ts] : new_edges) {
        if (!is_directed && src > tgt) {
            edges.push_back(tgt, src, ts);
        } else {
            edges.push_back(src, tgt, ts);
        }
        latest_timestamp = std::max(latest_timestamp, ts);
    }

    // Sort and merge new edges
    sort_and_merge_edges(start_idx);

    // Update timestamp groups after sorting
    edges.update_timestamp_groups();

    // Update node mappings
    node_mapping.update(edges, start_idx, edges.size());

    // Handle time window
    if (time_window > 0) {
        delete_old_edges();
    }

    // Rebuild edge indices
    node_index.rebuild(edges, node_mapping, is_directed);
}

void TemporalGraph::sort_and_merge_edges(size_t start_idx) {
    if (start_idx >= edges.size()) return;

    // Sort new edges first
    std::vector<size_t> indices(edges.size() - start_idx);
    std::iota(indices.begin(), indices.end(), 0);

    std::sort(indices.begin(), indices.end(),
        [this, start_idx](size_t a, size_t b) {
            return edges.timestamps[start_idx + a] < edges.timestamps[start_idx + b];
        });

    // Apply permutation to new edges
    EdgeData temp;
    temp.reserve(edges.size() - start_idx);
    for (size_t idx : indices) {
        temp.push_back(
            edges.sources[start_idx + idx],
            edges.targets[start_idx + idx],
            edges.timestamps[start_idx + idx]
        );
    }

    // Copy back sorted edges
    for (size_t i = 0; i < temp.size(); i++) {
        edges.sources[start_idx + i] = temp.sources[i];
        edges.targets[start_idx + i] = temp.targets[i];
        edges.timestamps[start_idx + i] = temp.timestamps[i];
    }

    // If needed, merge with existing edges
    if (start_idx > 0) {
        std::vector<size_t> merge_indices(edges.size());
        std::iota(merge_indices.begin(), merge_indices.end(), 0);

        std::sort(merge_indices.begin(), merge_indices.end(),
            [this](size_t a, size_t b) {
                return edges.timestamps[a] < edges.timestamps[b];
            });

        EdgeData merged;
        merged.reserve(edges.size());
        for (size_t idx : merge_indices) {
            merged.push_back(
                edges.sources[idx],
                edges.targets[idx],
                edges.timestamps[idx]
            );
        }
        edges = std::move(merged);
    }
}

void TemporalGraph::delete_old_edges() {
    if (time_window <= 0 || edges.empty()) return;

    int64_t cutoff_time = latest_timestamp - time_window;
    auto it = std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), cutoff_time);
    if (it == edges.timestamps.begin()) return;

    size_t delete_count = it - edges.timestamps.begin();
    size_t remaining = edges.size() - delete_count;

    if (remaining > 0) {
        std::move(edges.sources.begin() + delete_count, edges.sources.end(), edges.sources.begin());
        std::move(edges.targets.begin() + delete_count, edges.targets.end(), edges.targets.begin());
        std::move(edges.timestamps.begin() + delete_count, edges.timestamps.end(), edges.timestamps.begin());
    }

    edges.resize(remaining);

    // Update all data structures after edge deletion
    edges.update_timestamp_groups();
    node_mapping.update(edges, 0, edges.size());  // Rebuild from scratch since indices changed
    node_index.rebuild(edges, node_mapping, is_directed);
}

size_t TemporalGraph::count_timestamps_less_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    auto it = std::lower_bound(edges.unique_timestamps.begin(), edges.unique_timestamps.end(), timestamp);
    return it - edges.unique_timestamps.begin();
}

size_t TemporalGraph::count_timestamps_greater_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    auto it = std::upper_bound(edges.unique_timestamps.begin(), edges.unique_timestamps.end(), timestamp);
    return edges.unique_timestamps.end() - it;
}

size_t TemporalGraph::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& group_offsets = is_directed ?
        node_index.inbound_group_offsets :
        node_index.outbound_group_offsets;
    const auto& group_indices = is_directed ?
        node_index.inbound_group_indices :
        node_index.outbound_group_indices;
    const auto& edge_indices = is_directed ?
        node_index.inbound_indices :
        node_index.outbound_indices;

    size_t group_start = group_offsets[dense_idx];
    size_t group_end = group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    auto it = std::lower_bound(
        group_indices.begin() + group_start,
        group_indices.begin() + group_end,
        timestamp,
        [this, &edge_indices](size_t group_pos, int64_t ts) {
            return edges.timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(group_indices.begin() + group_start, it);
}

size_t TemporalGraph::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& group_offsets = is_directed ?
        node_index.outbound_group_offsets :
        node_index.outbound_group_offsets;
    const auto& group_indices = is_directed ?
        node_index.outbound_group_indices :
        node_index.outbound_group_indices;
    const auto& edge_indices = is_directed ?
        node_index.outbound_indices :
        node_index.outbound_indices;

    size_t group_start = group_offsets[dense_idx];
    size_t group_end = group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    auto it = std::upper_bound(
        group_indices.begin() + group_start,
        group_indices.begin() + group_end,
        timestamp,
        [this, &edge_indices](int64_t ts, size_t group_pos) {
            return ts < edges.timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, group_indices.begin() + group_end);
}

std::tuple<int, int, int64_t> TemporalGraph::get_edge_at(size_t index, int64_t timestamp, bool forward) const {
    if (edges.empty()) return {-1, -1, -1};

    if (timestamp != -1) {
        // Forward walk: select from edges after timestamp
        // Backward walk: select from edges before timestamp
        size_t group_idx;
        if (forward) {
            size_t first_group = edges.find_group_after_timestamp(timestamp);
            if (first_group + index >= edges.get_timestamp_group_count()) {
                return {-1, -1, -1};
            }
            group_idx = first_group + index;
        } else {
            size_t last_group = edges.find_group_before_timestamp(timestamp);
            if (index > last_group) {
                return {-1, -1, -1};
            }
            group_idx = last_group - index;
        }

        // Get group boundaries directly
        auto [group_start, group_end] = edges.get_timestamp_group_range(group_idx);
        if (group_start == group_end) return {-1, -1, -1};

        // Random selection from group
        size_t random_idx = group_start + get_random_number(group_end - group_start);
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    } else {
        // No timestamp constraint
        size_t num_groups = edges.get_timestamp_group_count();
        if (index >= num_groups) return {-1, -1, -1};

        size_t group_idx = forward ? index : (num_groups - 1 - index);
        auto [group_start, group_end] = edges.get_timestamp_group_range(group_idx);
        if (group_start == group_end) return {-1, -1, -1};

        // Random selection from group
        size_t random_idx = group_start + get_random_number(group_end - group_start);
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    }
}

std::tuple<int, int, int64_t> TemporalGraph::get_node_edge_at(
    int node_id, size_t index, int64_t timestamp, bool forward) const {

    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return {-1, -1, -1};

    const auto& group_offsets = forward ?
        (is_directed ? node_index.outbound_group_offsets : node_index.outbound_group_offsets) :
        (is_directed ? node_index.inbound_group_offsets : node_index.outbound_group_offsets);

    const auto& group_indices = forward ?
        (is_directed ? node_index.outbound_group_indices : node_index.outbound_group_indices) :
        (is_directed ? node_index.inbound_group_indices : node_index.outbound_group_indices);

    const auto& edge_indices = forward ?
        (is_directed ? node_index.outbound_indices : node_index.outbound_indices) :
        (is_directed ? node_index.inbound_indices : node_index.outbound_indices);

    size_t group_start_offset = group_offsets[dense_idx];
    size_t group_end_offset = group_offsets[dense_idx + 1];
    if (group_start_offset == group_end_offset) return {-1, -1, -1};

    if (timestamp != -1) {
        // Count available groups and check index validity
        size_t available_groups = forward ?
            count_node_timestamps_greater_than(node_id, timestamp) :
            count_node_timestamps_less_than(node_id, timestamp);

        if (index >= available_groups) return {-1, -1, -1};

        // Find the target group
        size_t group_pos;
        if (forward) {
            auto it = std::upper_bound(
                group_indices.begin() + group_start_offset,
                group_indices.begin() + group_end_offset,
                timestamp,
                [this, &edge_indices](int64_t ts, size_t pos) {
                    return ts < edges.timestamps[edge_indices[pos]];
                });
            group_pos = (it - group_indices.begin()) + index;
        } else {
            auto it = std::lower_bound(
                group_indices.begin() + group_start_offset,
                group_indices.begin() + group_end_offset,
                timestamp,
                [this, &edge_indices](size_t pos, int64_t ts) {
                    return edges.timestamps[edge_indices[pos]] < ts;
                });
            group_pos = (it - group_indices.begin()) - index;
        }

        // Get edge range for this group
        size_t edge_start = group_indices[group_pos];
        size_t edge_end = (group_pos + 1 < group_end_offset) ?
                          group_indices[group_pos + 1] :
                          (forward ? node_index.outbound_offsets[dense_idx + 1] :
                                   node_index.inbound_offsets[dense_idx + 1]);

        // Random selection from group
        size_t random_idx = edge_indices[edge_start + get_random_number(edge_end - edge_start)];
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    } else {
        // No timestamp constraint
        size_t num_groups = group_end_offset - group_start_offset;
        if (index >= num_groups) return {-1, -1, -1};

        // Select group based on direction
        size_t group_pos = group_start_offset + (forward ? index : (num_groups - 1 - index));

        // Get edge range for this group
        size_t edge_start = group_indices[group_pos];
        size_t edge_end = (group_pos + 1 < group_end_offset) ?
                          group_indices[group_pos + 1] :
                          (forward ? node_index.outbound_offsets[dense_idx + 1] :
                                   node_index.inbound_offsets[dense_idx + 1]);

        // Random selection from group
        size_t random_idx = edge_indices[edge_start + get_random_number(edge_end - edge_start)];
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    }
}
