#include "TemporalGraph.h"
#include <algorithm>
#include <iostream>


TemporalGraph::TemporalGraph(bool directed, int64_t window)
    : is_directed(directed), time_window(window), latest_timestamp(0) {}

void TemporalGraph::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& new_edges) {
    if (new_edges.empty()) return;

    const size_t start_idx = edges.size();
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
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = start_idx + i;
    }

    std::sort(indices.begin(), indices.end(),
              [this](size_t i, size_t j) {
                  return edges.timestamps[i] < edges.timestamps[j];
              });

    // Apply permutation to new edges
    EdgeData temp;
    temp.reserve(edges.size() - start_idx);
    for (size_t idx : indices) {
        temp.push_back(
            edges.sources[idx],
            edges.targets[idx],
            edges.timestamps[idx]
        );
    }

    // Copy back sorted edges
    for (size_t i = 0; i < temp.size(); i++) {
        edges.sources[start_idx + i] = temp.sources[i];
        edges.targets[start_idx + i] = temp.targets[i];
        edges.timestamps[start_idx + i] = temp.timestamps[i];
    }

    // Merge with existing edges
    if (start_idx > 0) {
        EdgeData merged;
        merged.reserve(edges.size());

        size_t i = 0; // Index for existing edges
        size_t j = start_idx; // Index for new edges

        // Merge while keeping source/target/timestamp together
        while (i < start_idx && j < edges.size()) {
            if (edges.timestamps[i] <= edges.timestamps[j]) {
                merged.push_back(edges.sources[i], edges.targets[i], edges.timestamps[i]);
                i++;
            } else {
                merged.push_back(edges.sources[j], edges.targets[j], edges.timestamps[j]);
                j++;
            }
        }

        // Add remaining existing edges
        while (i < start_idx) {
            merged.push_back(edges.sources[i], edges.targets[i], edges.timestamps[i]);
            i++;
        }

        // Add remaining new edges
        while (j < edges.size()) {
            merged.push_back(edges.sources[j], edges.targets[j], edges.timestamps[j]);
            j++;
        }

        edges = std::move(merged);
    }
}

void TemporalGraph::delete_old_edges() {
    if (time_window <= 0 || edges.empty()) return;

    const int64_t cutoff_time = latest_timestamp - time_window;
    const auto it = std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), cutoff_time);
    if (it == edges.timestamps.begin()) return;

    const int delete_count = static_cast<int>(it - edges.timestamps.begin());
    const size_t remaining = edges.size() - delete_count;

    // Track which nodes still have edges
    std::vector<bool> has_edges(node_mapping.sparse_to_dense.size(), false);

    if (remaining > 0) {
        std::move(edges.sources.begin() + delete_count, edges.sources.end(), edges.sources.begin());
        std::move(edges.targets.begin() + delete_count, edges.targets.end(), edges.targets.begin());
        std::move(edges.timestamps.begin() + delete_count, edges.timestamps.end(), edges.timestamps.begin());

        // Mark nodes that still have edges
        for (size_t i = 0; i < remaining; i++) {
            has_edges[edges.sources[i]] = true;
            has_edges[edges.targets[i]] = true;
        }
    }

    edges.resize(remaining);

    // Mark nodes with no edges as deleted
    for (size_t i = 0; i < has_edges.size(); i++) {
        if (!has_edges[i]) {
            node_mapping.mark_node_deleted(static_cast<int>(i));
        }
    }

    // Update all data structures after edge deletion
    edges.update_timestamp_groups();
    node_mapping.update(edges, 0, edges.size());
    node_index.rebuild(edges, node_mapping, is_directed);
}

size_t TemporalGraph::count_timestamps_less_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    const auto it = std::lower_bound(edges.unique_timestamps.begin(), edges.unique_timestamps.end(), timestamp);
    return it - edges.unique_timestamps.begin();
}

size_t TemporalGraph::count_timestamps_greater_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    auto it = std::upper_bound(edges.unique_timestamps.begin(), edges.unique_timestamps.end(), timestamp);
    return edges.unique_timestamps.end() - it;
}

size_t TemporalGraph::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    const int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& group_offsets = is_directed ? node_index.inbound_group_offsets : node_index.outbound_group_offsets;
    const auto& group_indices = is_directed ? node_index.inbound_group_indices : node_index.outbound_group_indices;
    const auto& edge_indices = is_directed ? node_index.inbound_indices : node_index.outbound_indices;

    size_t group_start = group_offsets[dense_idx];
    size_t group_end = group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    auto it = std::lower_bound(
        group_indices.begin() + static_cast<int>(group_start),
        group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [this, &edge_indices](size_t group_pos, int64_t ts)
        {
            return edges.timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(group_indices.begin() + static_cast<int>(group_start), it);
}

size_t TemporalGraph::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& group_offsets = node_index.outbound_group_offsets;
    const auto& group_indices = node_index.outbound_group_indices;
    const auto& edge_indices = node_index.outbound_indices;

    const size_t group_start = group_offsets[dense_idx];
    const size_t group_end = group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    const auto it = std::upper_bound(
        group_indices.begin() + static_cast<int>(group_start),
        group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [this, &edge_indices](int64_t ts, size_t group_pos)
        {
            return ts < edges.timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, group_indices.begin() + static_cast<int>(group_end));
}

std::tuple<int, int, int64_t> TemporalGraph::get_edge_at(const std::function<size_t(int, int, bool)>& index_selector,
                                                         int64_t timestamp, bool forward) const {
    if (edges.empty()) return {-1, -1, -1};

    if (timestamp != -1) {
        // Forward walk: select from edges after timestamp
        // Backward walk: select from edges before timestamp
        size_t group_idx;
        if (forward) {
            const size_t first_group = edges.find_group_after_timestamp(timestamp);
            const size_t num_groups = edges.get_timestamp_group_count() - first_group;
            if (num_groups == 0) return {-1, -1, -1};

            // Get index using lambda
            // Last param says if we should prioritize the end of the sequence, which is opposite to forward
            const size_t index = index_selector(0, static_cast<int>(num_groups), false);
            if (index >= num_groups) return {-1, -1, -1};

            group_idx = first_group + index;
        }
        else {
            const size_t last_group = edges.find_group_before_timestamp(timestamp);
            if (last_group == static_cast<size_t>(-1)) return {-1, -1, -1};

            const size_t num_groups = last_group + 1;

            const size_t index = index_selector(0, static_cast<int>(num_groups), true);
            if (index >= num_groups) return {-1, -1, -1};

            group_idx = last_group - index;
        }

        // Get group boundaries directly
        auto [group_start, group_end] = edges.get_timestamp_group_range(group_idx);
        if (group_start == group_end) return {-1, -1, -1};

        // Random selection from group
        size_t random_idx = group_start + get_random_number(static_cast<int>(group_end - group_start));
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    } else {
        // No timestamp constraint
        size_t num_groups = edges.get_timestamp_group_count();
        if (num_groups == 0) return {-1, -1, -1};

        // Get index using lambda
        // Last param says if we should prioritize the end of the sequence, which is opposite to forward
        size_t index = index_selector(0, static_cast<int>(num_groups), !forward);
        if (index >= num_groups) return {-1, -1, -1};

        size_t group_idx = forward ? index : (num_groups - 1 - index);
        auto [group_start, group_end] = edges.get_timestamp_group_range(group_idx);
        if (group_start == group_end) return {-1, -1, -1};

        // Random selection from group
        size_t random_idx = group_start + get_random_number(static_cast<int>(group_end - group_start));
        return {
            edges.sources[random_idx],
            edges.targets[random_idx],
            edges.timestamps[random_idx]
        };
    }
}

std::tuple<int, int, int64_t> TemporalGraph::get_node_edge_at(
    const int node_id, const std::function<size_t(int, int, bool)>& index_selector, const int64_t timestamp,
    const bool forward) const {

    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return {-1, -1, -1};

    // Get appropriate node indices based on direction and graph type
    const auto& group_offsets = forward
                                    ? node_index.outbound_group_offsets
                                    : (is_directed
                                           ? node_index.inbound_group_offsets
                                           : node_index.outbound_group_offsets);

    const auto& group_indices = forward
                                    ? node_index.outbound_group_indices
                                    : (is_directed
                                           ? node_index.inbound_group_indices
                                           : node_index.outbound_group_indices);

    const auto& edge_indices = forward
                                   ? node_index.outbound_indices
                                   : (is_directed ? node_index.inbound_indices : node_index.outbound_indices);

    // Get node's group range
    size_t group_start_offset = group_offsets[dense_idx];
    size_t group_end_offset = group_offsets[dense_idx + 1];
    if (group_start_offset == group_end_offset) return {-1, -1, -1};

    size_t group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp
            auto it = std::upper_bound(
                group_indices.begin() + static_cast<int>(group_start_offset),
                group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [this, &edge_indices](int64_t ts, size_t pos)
                {
                    return ts < edges.timestamps[edge_indices[pos]];
                });

            // Count available groups after timestamp
            size_t available = group_indices.begin() + static_cast<int>(group_end_offset) - it;
            if (available == 0) return {-1, -1, -1};

            // Get index using lambda
            // Last param says if we should prioritize the end of the sequence, which is opposite to forward
            size_t index = index_selector(0, static_cast<int>(available), false);
            if (index >= available) return {-1, -1, -1};

            // Select index th group after timestamp
            group_pos = (it - group_indices.begin()) + index;
        } else {
            // backward case
            // Find first group >= timestamp
            auto it = std::lower_bound(
                group_indices.begin() + static_cast<int>(group_start_offset),
                group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [this, &edge_indices](size_t pos, int64_t ts)
                {
                    return edges.timestamps[edge_indices[pos]] < ts;
                });

            // Count all available groups before timestamp
            size_t available = it - (group_indices.begin() + static_cast<int>(group_start_offset));
            if (available == 0) return {-1, -1, -1};

            // Get index using lambda
            const size_t index = index_selector(0, static_cast<int>(available), true);
            if (index >= available) return {-1, -1, -1};

            // Go backward by index from the last available position
            group_pos = (it - group_indices.begin()) - index - 1;
        }
    } else {
        // No timestamp constraint - select from all groups
        size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return {-1, -1, -1};

        // Get index using lambda
        size_t index = index_selector(0, static_cast<int>(num_groups), !forward);
        if (index >= num_groups) return {-1, -1, -1};

        // Select group based on direction
        if (forward) {
            group_pos = group_start_offset + index;
        } else {
            // For backward walks:
            // select_first() with index=0 should get latest timestamp
            // select_last() with index=num_groups-1 should get the earliest timestamp
            group_pos = group_end_offset - index - 1;
        }
    }

    // Get edge range for selected group
    const size_t edge_start = group_indices[group_pos];
    size_t edge_end;
    if (group_pos + 1 < group_end_offset) {
        edge_end = group_indices[group_pos + 1];
    } else {
        const auto& edge_offsets = forward
                                       ? node_index.outbound_offsets
                                       : (is_directed ? node_index.inbound_offsets : node_index.outbound_offsets);
        edge_end = edge_offsets[dense_idx + 1];
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + get_random_number(static_cast<int>(edge_end - edge_start))];

    return {
        edges.sources[edge_idx],
        edges.targets[edge_idx],
        edges.timestamps[edge_idx]
    };
}

std::vector<int> TemporalGraph::get_node_ids() const {
    return node_mapping.get_active_node_ids();
}

std::vector<std::tuple<int, int, int64_t>> TemporalGraph::get_edges() {
    return edges.get_edges();
}
