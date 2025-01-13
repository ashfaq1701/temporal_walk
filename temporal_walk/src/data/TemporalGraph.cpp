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

    // Apply permutation
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
}

// Counting methods
size_t TemporalGraph::count_timestamps_less_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    auto it = std::lower_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp);
    if (it == edges.timestamps.begin()) return 0;
    --it;

    size_t count = 1;  // First timestamp group
    int64_t prev_ts = edges.timestamps[0];

    for (auto curr = edges.timestamps.begin() + 1; curr <= it; ++curr) {
        if (*curr != prev_ts) {
            count++;
            prev_ts = *curr;
        }
    }
    return count;
}

size_t TemporalGraph::count_timestamps_greater_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    auto it = std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp);
    if (it == edges.timestamps.end()) return 0;

    size_t count = 1;  // First timestamp group after timestamp
    int64_t prev_ts = *it;

    for (auto curr = it + 1; curr != edges.timestamps.end(); ++curr) {
        if (*curr != prev_ts) {
            count++;
            prev_ts = *curr;
        }
    }
    return count;
}

size_t TemporalGraph::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& groups = is_directed ?
        node_index.inbound_timestamp_groups[dense_idx] :
        node_index.outbound_timestamp_groups[dense_idx];

    if (groups.empty()) return 0;

    auto it = std::lower_bound(groups.begin(), groups.end() - 1, timestamp,
        [this, dense_idx](size_t group_start, int64_t ts) {
            const auto& indices = is_directed ?
                node_index.inbound_indices :
                node_index.outbound_indices;
            return edges.timestamps[indices[group_start]] < ts;
        });

    return std::distance(groups.begin(), it);
}

size_t TemporalGraph::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& groups = is_directed ?
        node_index.outbound_timestamp_groups[dense_idx] :
        node_index.outbound_timestamp_groups[dense_idx];

    if (groups.empty()) return 0;

    auto it = std::upper_bound(groups.begin(), groups.end() - 1, timestamp,
        [this, dense_idx](int64_t ts, size_t group_start) {
            const auto& indices = is_directed ?
                node_index.outbound_indices :
                node_index.outbound_indices;
            return ts < edges.timestamps[indices[group_start]];
        });

    return std::distance(it, groups.end() - 1);
}

std::tuple<int, int, int64_t> TemporalGraph::get_edge_at(size_t index, int64_t timestamp, bool forward) const {
    if (edges.empty()) return {-1, -1, -1};

    if (timestamp != -1) {
        // Forward walk: select from edges after timestamp
        // Backward walk: select from edges before timestamp
        size_t available_groups = forward ?
            count_timestamps_greater_than(timestamp) :
            count_timestamps_less_than(timestamp);

        if (index >= available_groups) return {-1, -1, -1};

        // Find the group boundaries
        auto it = forward ?
            std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp) :
            std::lower_bound(edges.timestamps.begin(), edges.timestamps.end(), timestamp);

        size_t group_start = it - edges.timestamps.begin();
        for (size_t i = 0; i < index; i++) {
            // Skip to desired group
            int64_t curr_ts = edges.timestamps[group_start];
            while (group_start < edges.size() && edges.timestamps[group_start] == curr_ts) {
                group_start++;
            }
        }

        // Find end of selected group
        size_t group_end = group_start;
        int64_t group_ts = edges.timestamps[group_start];
        while (group_end < edges.size() && edges.timestamps[group_end] == group_ts) {
            group_end++;
        }

        // Random selection from group
        size_t random_idx = group_start + get_random_number(group_end - group_start);
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    } else {
        // No timestamp constraint
        size_t total_groups = count_timestamps_less_than(edges.timestamps.back() + 1);
        if (index >= total_groups) return {-1, -1, -1};

        size_t group_idx = forward ? index : (total_groups - 1 - index);

        // Find the group
        size_t group_start = 0;
        for (size_t i = 0; i < group_idx; i++) {
            int64_t curr_ts = edges.timestamps[group_start];
            while (group_start < edges.size() && edges.timestamps[group_start] == curr_ts) {
                group_start++;
            }
        }

        // Find group end
        size_t group_end = group_start;
        int64_t group_ts = edges.timestamps[group_start];
        while (group_end < edges.size() && edges.timestamps[group_end] == group_ts) {
            group_end++;
        }

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

    const auto& groups = forward ?
        (is_directed ? node_index.outbound_timestamp_groups[dense_idx] :
                      node_index.outbound_timestamp_groups[dense_idx]) :
        (is_directed ? node_index.inbound_timestamp_groups[dense_idx] :
                      node_index.outbound_timestamp_groups[dense_idx]);

    const auto& indices = forward ?
        (is_directed ? node_index.outbound_indices : node_index.outbound_indices) :
        (is_directed ? node_index.inbound_indices : node_index.outbound_indices);

    if (groups.empty()) return {-1, -1, -1};

    if (timestamp != -1) {
        size_t available_groups = forward ?
            count_node_timestamps_greater_than(node_id, timestamp) :
            count_node_timestamps_less_than(node_id, timestamp);

        if (index >= available_groups) return {-1, -1, -1};

        // Find group boundaries
        size_t group_idx;
        if (forward) {
            group_idx = std::upper_bound(groups.begin(), groups.end() - 1, timestamp,
                [this, &indices](int64_t ts, size_t group_start) {
                    return ts < edges.timestamps[indices[group_start]];
                }) - groups.begin() + index;
        } else {
            group_idx = std::lower_bound(groups.begin(), groups.end() - 1, timestamp,
                [this, &indices](size_t group_start, int64_t ts) {
                    return edges.timestamps[indices[group_start]] < ts;
                }) - groups.begin() - index;
        }

        // Random selection from group
        size_t group_start = groups[group_idx];
        size_t group_end = groups[group_idx + 1];
        size_t random_idx = indices[group_start + get_random_number(group_end - group_start)];

        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    } else {
        if (index >= groups.size() - 1) return {-1, -1, -1};

        size_t group_idx = forward ? index : (groups.size() - 2 - index);

        // Random selection from group
        size_t group_start = groups[group_idx];
        size_t group_end = groups[group_idx + 1];
        size_t random_idx = indices[group_start + get_random_number(group_end - group_start)];

        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    }
}
