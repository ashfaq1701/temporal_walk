#include "TemporalGraph.cuh"
#include <algorithm>
#include <iostream>

#include "../random/IndexBasedRandomPicker.h"
#include "../random/WeightBasedRandomPicker.cuh"
#include "../random/RandomPicker.h"

TemporalGraph::TemporalGraph(
    const bool directed,
    const bool use_gpu,
    const int64_t window,
    const bool enable_weight_computation,
    const double timescale_bound)
    : is_directed(directed), use_gpu(use_gpu), time_window(window),
        enable_weight_computation(enable_weight_computation),
        timescale_bound(timescale_bound),node_index(use_gpu),
        edges(use_gpu), node_mapping(use_gpu) {}

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

    // Update node mappings
    node_mapping.update(edges, start_idx, edges.size());

    // Sort and merge new edges
    sort_and_merge_edges(start_idx);

    // Update timestamp groups after sorting
    edges.update_timestamp_groups();

    // Handle time window
    if (time_window > 0) {
        delete_old_edges();
    }

    // Rebuild edge indices
    node_index.rebuild(edges, node_mapping, is_directed);

    if (enable_weight_computation) {
        update_temporal_weights();
    }
}

void TemporalGraph::update_temporal_weights() {
    edges.update_temporal_weights(timescale_bound);
    node_index.update_temporal_weights(edges, timescale_bound);
}

void TemporalGraph::sort_and_merge_edges(const size_t start_idx) {
    if (start_idx >= edges.size()) return;

    // Sort new edges first
    std::vector<size_t> indices(edges.size() - start_idx);
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = start_idx + i;
    }

    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec)
    {
        std::sort(indices.begin(), indices.end(),
                  [&](const size_t i, const size_t j) {
                      return timestamps_vec[i] < timestamps_vec[j];
                  });

        // Apply permutation in-place using temporary vectors
        using SourcesVecType = std::decay_t<decltype(sources_vec)>;
        using TargetsVecType = std::decay_t<decltype(targets_vec)>;
        using TimestampsVecType = std::decay_t<decltype(timestamps_vec)>;

        SourcesVecType sorted_sources(edges.size() - start_idx);
        TargetsVecType sorted_targets(edges.size() - start_idx);
        TimestampsVecType sorted_timestamps(edges.size() - start_idx);

        for (size_t i = 0; i < indices.size(); i++) {
            const size_t idx = indices[i];
            sorted_sources[i] = sources_vec[idx];
            sorted_targets[i] = targets_vec[idx];
            sorted_timestamps[i] = timestamps_vec[idx];
        }

        // Copy back sorted edges
        for (size_t i = 0; i < indices.size(); i++) {
            sources_vec[start_idx + i] = sorted_sources[i];
            targets_vec[start_idx + i] = sorted_targets[i];
            timestamps_vec[start_idx + i] = sorted_timestamps[i];
        }

        // Merge with existing edges
        if (start_idx > 0) {
            // Create buffer vectors
            SourcesVecType merged_sources(edges.size());
            TargetsVecType merged_targets(edges.size());
            TimestampsVecType merged_timestamps(edges.size());

            size_t i = 0;  // Index for existing edges
            size_t j = start_idx;  // Index for new edges
            size_t k = 0;  // Index for merged result

            // Merge while keeping arrays aligned
            while (i < start_idx && j < edges.size()) {
                if (timestamps_vec[i] <= timestamps_vec[j]) {
                    merged_sources[k] = sources_vec[i];
                    merged_targets[k] = targets_vec[i];
                    merged_timestamps[k] = timestamps_vec[i];
                    i++;
                } else {
                    merged_sources[k] = sources_vec[j];
                    merged_targets[k] = targets_vec[j];
                    merged_timestamps[k] = timestamps_vec[j];
                    j++;
                }
                k++;
            }

            // Copy remaining entries
            while (i < start_idx) {
                merged_sources[k] = sources_vec[i];
                merged_targets[k] = targets_vec[i];
                merged_timestamps[k] = timestamps_vec[i];
                i++;
                k++;
            }

            while (j < edges.size()) {
                merged_sources[k] = sources_vec[j];
                merged_targets[k] = targets_vec[j];
                merged_timestamps[k] = timestamps_vec[j];
                j++;
                k++;
            }

            // Copy merged data back to edges
            edges.sources.emplace<SourcesVecType>(std::move(merged_sources));
            edges.targets.emplace<TargetsVecType>(std::move(merged_targets));
            edges.timestamps.emplace<TimestampsVecType>(std::move(merged_timestamps));
        }
    }, edges.sources, edges.targets, edges.timestamps);
}

void TemporalGraph::delete_old_edges() {
    if (time_window <= 0 || edges.empty()) return;

    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec, auto& sparse_to_dense_vec)
    {
        const int64_t cutoff_time = latest_timestamp - time_window;
        const auto it = std::upper_bound(timestamps_vec.begin(), timestamps_vec.end(), cutoff_time);
        if (it == timestamps_vec.begin()) return;

        const int delete_count = static_cast<int>(it - timestamps_vec.begin());
        const size_t remaining = edges.size() - delete_count;

        // Track which nodes still have edges
        using BoolVecType = typename RebindVectorT<size_t, std::remove_reference_t<decltype(sources_vec)>>::Type;
        BoolVecType has_edges(sparse_to_dense_vec.size(), false);

        if (remaining > 0) {
            std::move(sources_vec.begin() + delete_count, sources_vec.end(), sources_vec.begin());
            std::move(targets_vec.begin() + delete_count, targets_vec.end(), targets_vec.begin());
            std::move(timestamps_vec.begin() + delete_count, timestamps_vec.end(), timestamps_vec.begin());

            // Mark nodes that still have edges
            for (size_t i = 0; i < remaining; i++) {
                has_edges[sources_vec[i]] = true;
                has_edges[targets_vec[i]] = true;
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
    }, edges.sources, edges.targets, edges.timestamps, node_mapping.sparse_to_dense);
}

size_t TemporalGraph::count_timestamps_less_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    return std::visit([&](const auto& unique_timestamps_vec)
    {
        const auto it = std::lower_bound(unique_timestamps_vec.begin(), unique_timestamps_vec.end(), timestamp);
        return it - unique_timestamps_vec.begin();
    }, edges.unique_timestamps);
}

size_t TemporalGraph::count_timestamps_greater_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    return std::visit([&](const auto& unique_timestamps_vec)
    {
        auto it = std::upper_bound(unique_timestamps_vec.begin(), unique_timestamps_vec.end(), timestamp);
        return unique_timestamps_vec.end() - it;
    }, edges.unique_timestamps);
}

size_t TemporalGraph::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    const int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = is_directed ? node_index.inbound_timestamp_group_offsets : node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = is_directed ? node_index.inbound_timestamp_group_indices : node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = is_directed ? node_index.inbound_indices : node_index.outbound_indices;

    return std::visit([&](const auto& timestamp_group_offsets_vec, const auto& timestamp_group_indices_vec, const auto& edge_indices_vec, const auto& timestamps_vec)
    {
        const size_t group_start = timestamp_group_offsets_vec[dense_idx];
        const size_t group_end = timestamp_group_offsets_vec[dense_idx + 1];
        if (group_start == group_end) return static_cast<size_t>(0);

        // Binary search on group indices
        auto it = std::lower_bound(
            timestamp_group_indices_vec.begin() + static_cast<int>(group_start),
            timestamp_group_indices_vec.begin() + static_cast<int>(group_end),
            timestamp,
            [this, &edge_indices_vec, &timestamps_vec](size_t group_pos, int64_t ts)
            {
                return timestamps_vec[edge_indices_vec[group_pos]] < ts;
            });

        return static_cast<size_t>(std::distance(timestamp_group_indices_vec.begin() + static_cast<int>(group_start), it));
    }, timestamp_group_offsets, timestamp_group_indices, edge_indices, edges.timestamps);
}

size_t TemporalGraph::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = node_index.outbound_indices;

    return std::visit([&](const auto& timestamp_group_offsets_vec, const auto& timestamp_group_indices_vec, const auto& edge_indices_vec, const auto& timestamps_vec)
    {
        const size_t group_start = timestamp_group_offsets_vec[dense_idx];
        const size_t group_end = timestamp_group_offsets_vec[dense_idx + 1];
        if (group_start == group_end) return static_cast<size_t>(0);

        // Binary search on group indices
        const auto it = std::upper_bound(
            timestamp_group_indices_vec.begin() + static_cast<int>(group_start),
            timestamp_group_indices_vec.begin() + static_cast<int>(group_end),
            timestamp,
            [this, &edge_indices_vec, &timestamps_vec](int64_t ts, size_t group_pos)
            {
                return ts < timestamps_vec[edge_indices_vec[group_pos]];
            });

        return static_cast<size_t>(std::distance(it, timestamp_group_indices_vec.begin() + static_cast<int>(group_end)));
    }, timestamp_group_offsets, timestamp_group_indices, edge_indices, edges.timestamps);
}

std::tuple<int, int, int64_t> TemporalGraph::get_edge_at(
    RandomPicker& picker,
    int64_t timestamp,
    bool forward) const {

    if (edges.empty()) return {-1, -1, -1};

    const size_t num_groups = edges.get_timestamp_group_count();
    if (num_groups == 0) return {-1, -1, -1};

    return std::visit([&](const auto& sources_vec, const auto& targets_vec, const auto& timestamps_vec)
    {
        size_t group_idx;
        if (timestamp != -1) {
            if (forward) {
                const size_t first_group = edges.find_group_after_timestamp(timestamp);
                const size_t available_groups = num_groups - first_group;
                if (available_groups == 0) return std::tuple<int, int, int64_t>{-1, -1, -1};

                if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                    const size_t index = index_picker->pick_random(0, static_cast<int>(available_groups), false);
                    if (index >= available_groups) return std::tuple<int, int, int64_t>{-1, -1, -1};
                    group_idx = first_group + index;
                }
                else {
                    auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                    group_idx = weight_picker->pick_random(
                        edges.forward_cumulative_weights_exponential,
                        static_cast<int>(first_group),
                        static_cast<int>(num_groups));
                }
            } else {
                const size_t last_group = edges.find_group_before_timestamp(timestamp);
                if (last_group == static_cast<size_t>(-1)) return std::tuple<int, int, int64_t>{-1, -1, -1};

                const size_t available_groups = last_group + 1;
                if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                    const size_t index = index_picker->pick_random(0, static_cast<int>(available_groups), true);
                    if (index >= available_groups) return std::tuple<int, int, int64_t>{-1, -1, -1};
                    group_idx = last_group - (available_groups - index - 1);
                }
                else {
                    auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                    group_idx = weight_picker->pick_random(
                        edges.backward_cumulative_weights_exponential,
                        0,
                        static_cast<int>(last_group + 1));
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(num_groups), !forward);
                if (index >= num_groups) return std::tuple<int, int, int64_t>{-1, -1, -1};
                group_idx = index;
            } else {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                if (forward) {
                    group_idx = weight_picker->pick_random(
                        edges.forward_cumulative_weights_exponential,
                        0,
                        static_cast<int>(num_groups));
                }
                else {
                    group_idx = weight_picker->pick_random(
                        edges.backward_cumulative_weights_exponential,
                        0,
                        static_cast<int>(num_groups));
                }
            }
        }

        // Get selected group's boundaries
        auto [group_start, group_end] = edges.get_timestamp_group_range(group_idx);
        if (group_start == group_end) {
            return  std::tuple<int, int, int64_t>{-1, -1, -1};
        }

        // Random selection from the chosen group
        const size_t random_idx = group_start + generate_random_number_bounded_by(static_cast<int>(group_end - group_start));
        return std::tuple<int, int, int64_t>{sources_vec[random_idx], targets_vec[random_idx], timestamps_vec[random_idx]};
    }, edges.sources, edges.targets, edges.timestamps);
}

std::tuple<int, int, int64_t> TemporalGraph::get_node_edge_at(
    const int node_id,
    RandomPicker& picker,
    const int64_t timestamp,
    const bool forward) const {

    const int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return {-1, -1, -1};

    // Get appropriate node indices based on direction and graph type
    const auto& timestamp_group_offsets = forward
        ? node_index.outbound_timestamp_group_offsets
        : (is_directed ? node_index.inbound_timestamp_group_offsets : node_index.outbound_timestamp_group_offsets);

    const auto& timestamp_group_indices = forward
        ? node_index.outbound_timestamp_group_indices
        : (is_directed ? node_index.inbound_timestamp_group_indices : node_index.outbound_timestamp_group_indices);

    const auto& edge_indices = forward
        ? node_index.outbound_indices
        : (is_directed ? node_index.inbound_indices : node_index.outbound_indices);

    return std::visit([&](const auto& sources_vec, const auto& targets_vec, const auto& timestamps_vec,
        const auto& outbound_offsets_vec, const auto& inbound_offsets_vec, const auto& timestamp_group_offsets_vec,
        const auto& timestamp_group_indices_vec, const auto& edge_indices_vec)
    {
        // Get node's group range
        const size_t group_start_offset = timestamp_group_offsets_vec[dense_idx];
        const size_t group_end_offset = timestamp_group_offsets_vec[dense_idx + 1];
        if (group_start_offset == group_end_offset) return std::tuple<int, int, int64_t>{-1, -1, -1};

        size_t group_pos;
        if (timestamp != -1) {
            if (forward) {
                // Find first group after timestamp
                auto it = std::upper_bound(
                    timestamp_group_indices_vec.begin() + static_cast<int>(group_start_offset),
                    timestamp_group_indices_vec.begin() + static_cast<int>(group_end_offset),
                    timestamp,
                    [this, &edge_indices_vec, &timestamps_vec](int64_t ts, size_t pos) {
                        return ts < timestamps_vec[edge_indices_vec[pos]];
                    });

                // Count available groups after timestamp
                const size_t available = timestamp_group_indices_vec.begin() +
                    static_cast<int>(group_end_offset) - it;
                if (available == 0) return std::tuple<int, int, int64_t>{-1, -1, -1};

                const size_t start_pos = it - timestamp_group_indices_vec.begin();
                if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                    const size_t index = index_picker->pick_random(0, static_cast<int>(available), false);
                    if (index >= available) return std::tuple<int, int, int64_t>{-1, -1, -1};
                    group_pos = start_pos + index;
                }
                else
                {
                    auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                    group_pos = weight_picker->pick_random(
                        node_index.outbound_forward_cumulative_weights_exponential,
                        static_cast<int>(start_pos),
                        static_cast<int>(group_end_offset));
                }
            } else {
                // Find first group >= timestamp
                auto it = std::lower_bound(
                    timestamp_group_indices_vec.begin() + static_cast<int>(group_start_offset),
                    timestamp_group_indices_vec.begin() + static_cast<int>(group_end_offset),
                    timestamp,
                    [this, &edge_indices_vec, &timestamps_vec](size_t pos, int64_t ts) {
                        return timestamps_vec[edge_indices_vec[pos]] < ts;
                    });

                const size_t available = it - (timestamp_group_indices_vec.begin() +
                    static_cast<int>(group_start_offset));
                if (available == 0) return std::tuple<int, int, int64_t>{-1, -1, -1};

                if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                    const size_t index = index_picker->pick_random(0, static_cast<int>(available), true);
                    if (index >= available) return std::tuple<int, int, int64_t>{-1, -1, -1};
                    group_pos = (it - timestamp_group_indices_vec.begin()) - 1 - (available - index - 1);
                }
                else
                {
                    auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                    group_pos = weight_picker->pick_random(
                        is_directed
                            ? node_index.inbound_backward_cumulative_weights_exponential
                            : node_index.outbound_backward_cumulative_weights_exponential,
                        static_cast<int>(group_start_offset), // start from node's first group
                        static_cast<int>(it - timestamp_group_indices_vec.begin()) // up to and excluding first group >= timestamp
                    );
                }
            }
        } else {
            // No timestamp constraint - select from all groups
            const size_t num_groups = group_end_offset - group_start_offset;
            if (num_groups == 0) return std::tuple<int, int, int64_t>{-1, -1, -1};

            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(num_groups), !forward);
                if (index >= num_groups) return std::tuple<int, int, int64_t>{-1, -1, -1};
                group_pos = forward
                    ? group_start_offset + index
                    : group_end_offset - 1 - (num_groups - index - 1);
            }
            else
            {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                if (forward)
                {
                    group_pos = weight_picker->pick_random(
                        node_index.outbound_forward_cumulative_weights_exponential,
                        static_cast<int>(group_start_offset),
                        static_cast<int>(group_end_offset));
                }
                else
                {
                    group_pos = weight_picker->pick_random(
                        is_directed
                            ? node_index.inbound_backward_cumulative_weights_exponential
                            : node_index.outbound_backward_cumulative_weights_exponential,
                        static_cast<int>(group_start_offset),
                        static_cast<int>(group_end_offset));
                }
            }
        }

        // Get edge range for selected group
        const size_t edge_start = timestamp_group_indices_vec[group_pos];
        const size_t edge_end = (group_pos + 1 < group_end_offset)
            ? timestamp_group_indices_vec[group_pos + 1]
            : (forward ? outbound_offsets_vec[dense_idx + 1]
                      : (is_directed ? inbound_offsets_vec[dense_idx + 1]
                                    : outbound_offsets_vec[dense_idx + 1]));

        // Validate range before random selection
        if (edge_start >= edge_end || edge_start >= edge_indices_vec.size() || edge_end > edge_indices_vec.size()) {
            return std::tuple<int, int, int64_t>{-1, -1, -1};
        }

        // Random selection from group
        const size_t edge_idx = edge_indices_vec[edge_start + generate_random_number_bounded_by(static_cast<int>(edge_end - edge_start))];

        return std::tuple<int, int, int64_t>{
            sources_vec[edge_idx],
            targets_vec[edge_idx],
            timestamps_vec[edge_idx]
        };
    }, edges.sources, edges.targets, edges.timestamps, node_index.outbound_offsets, node_index.inbound_offsets,
    timestamp_group_offsets, timestamp_group_indices, edge_indices);
}

std::vector<int> TemporalGraph::get_node_ids() const {
    return node_mapping.get_active_node_ids();
}

std::vector<std::tuple<int, int, int64_t>> TemporalGraph::get_edges() {
    return edges.get_edges();
}
