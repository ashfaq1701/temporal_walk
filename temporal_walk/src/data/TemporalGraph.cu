#include "TemporalGraph.cuh"
#include <algorithm>
#include <iostream>

#include "../cuda/cuda_functions.cuh"
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
        timescale_bound(timescale_bound), latest_timestamp(0),
        node_index(use_gpu), edges(use_gpu), node_mapping(use_gpu) {}

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

    // Create indices for sorting
    std::vector<size_t> indices(edges.size() - start_idx);

    // Sort indices based on timestamps
    cuda_functions::sort_edges_by_timestamp(
        indices,
        edges.timestamps,
        start_idx,
        edges.should_use_gpu()
    );

    // Gather sorted data using the indices
    cuda_functions::gather_sorted_edges(
        indices,
        edges.sources,
        edges.sources,
        start_idx,
        edges.should_use_gpu()
    );
    cuda_functions::gather_sorted_edges(
        indices,
        edges.targets,
        edges.targets,
        start_idx,
        edges.should_use_gpu()
    );
    cuda_functions::gather_sorted_edges(
        indices,
        edges.timestamps,
        edges.timestamps,
        start_idx,
        edges.should_use_gpu()
    );

    // Merge with existing edges if needed
    if (start_idx > 0) {
        cuda_functions::merge_sorted_edges(
            edges.timestamps,
            edges.sources,
            edges.targets,
            start_idx,
            edges.should_use_gpu()
        );
    }
}

void TemporalGraph::delete_old_edges() {
    if (time_window <= 0 || edges.empty()) return;

    const int64_t cutoff_time = latest_timestamp - time_window;

    // Find edges to delete and count remaining
    auto [delete_count, remaining] = cuda_functions::find_cutoff_edges(
        edges.timestamps, cutoff_time, edges.should_use_gpu());

    if (delete_count == 0) return;

    // Track which nodes still have edges
    DualVector<short> has_edges(edges.should_use_gpu());
    has_edges.assign(node_mapping.sparse_to_dense.size(), 0);

    if (remaining > 0) {
        // Compact the edges after deletion
        cuda_functions::compact_edges_after_deletion(
            edges.sources, delete_count, edges.should_use_gpu());
        cuda_functions::compact_edges_after_deletion(
            edges.targets, delete_count, edges.should_use_gpu());
        cuda_functions::compact_edges_after_deletion(
            edges.timestamps, delete_count, edges.should_use_gpu());

        // Mark nodes that still have edges
        cuda_functions::mark_remaining_edges(
            edges.sources,
            edges.targets,
            has_edges,
            remaining,
            edges.should_use_gpu());
    }

    edges.resize(remaining);

    // Mark nodes with no edges as deleted
    if (edges.should_use_gpu()) {
        #ifdef HAS_CUDA
        auto has_edges_ptr = has_edges.device_data().get();
        const size_t is_deleted_size = node_mapping.is_deleted.size();
        thrust::for_each(thrust::device,
            thrust::counting_iterator<size_t>(0),
            thrust::counting_iterator<size_t>(has_edges.size()),
            [has_edges = has_edges_ptr,
             deleted_ptr = node_mapping.is_deleted.device_data().get(),
             size = is_deleted_size] __device__ (const size_t i) {
                if (!has_edges[i]) {
                    NodeMapping::device_mark_node_deleted(static_cast<int>(i), deleted_ptr, size);
                }
            });
        #endif
    } else {
        for (size_t i = 0; i < has_edges.size(); i++) {
            if (!has_edges[i]) {
                node_mapping.host_mark_node_deleted(static_cast<int>(i));
            }
        }
    }

    // Update all data structures after edge deletion
    edges.update_timestamp_groups();
    node_mapping.update(edges, 0, edges.size());
    node_index.rebuild(edges, node_mapping, is_directed);
}

size_t TemporalGraph::count_timestamps_less_than(const int64_t timestamp) const {
    return cuda_functions::count_less_than(
        edges.unique_timestamps,
        timestamp,
        edges.should_use_gpu());
}

size_t TemporalGraph::count_timestamps_greater_than(const int64_t timestamp) const {
    return cuda_functions::count_greater_than(
        edges.unique_timestamps,
        timestamp,
        edges.should_use_gpu());
}

size_t TemporalGraph::count_node_timestamps_less_than(const int node_id, const int64_t timestamp) const {
    const int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = is_directed ?
        node_index.inbound_timestamp_group_offsets : node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = is_directed ?
        node_index.inbound_timestamp_group_indices : node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = is_directed ?
        node_index.inbound_indices : node_index.outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    return cuda_functions::count_node_timestamps_less_than(
        timestamp_group_indices,
        group_start,
        group_end,
        edge_indices,
        edges.timestamps,
        timestamp,
        edges.should_use_gpu());
}

size_t TemporalGraph::count_node_timestamps_greater_than(const int node_id, const int64_t timestamp) const {
    const int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = node_index.outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    return cuda_functions::count_node_timestamps_greater_than(
        timestamp_group_indices,
        group_start,
        group_end,
        edge_indices,
        edges.timestamps,
        timestamp,
        edges.should_use_gpu());
}

// Helper for picking random group based on timestamp
size_t TemporalGraph::get_timestamped_group_idx(
    const EdgeData &edges,
    RandomPicker &picker,
    const size_t num_groups,
    const int64_t timestamp,
    const bool forward,
    const DualVector<double> &forward_weights,
    const DualVector<double> &backward_weights) const {
    if (forward) {
        const size_t first_group = edges.find_group_after_timestamp(timestamp);
        const size_t available_groups = num_groups - first_group;
        if (available_groups == 0) return static_cast<size_t>(-1);

        if (auto *index_picker = dynamic_cast<IndexBasedRandomPicker *>(&picker)) {
            const size_t index = index_picker->pick_random(0, static_cast<int>(available_groups), false, use_gpu);
            if (index >= available_groups) return static_cast<size_t>(-1);
            return first_group + index;
        } else {
            auto *weight_picker = dynamic_cast<WeightBasedRandomPicker *>(&picker);
            return weight_picker->pick_random(
                forward_weights,
                static_cast<int>(first_group),
                static_cast<int>(num_groups));
        }
    } else {
        const size_t last_group = edges.find_group_before_timestamp(timestamp);
        if (last_group == static_cast<size_t>(-1)) return static_cast<size_t>(-1);

        const size_t available_groups = last_group + 1;
        if (auto *index_picker = dynamic_cast<IndexBasedRandomPicker *>(&picker)) {
            const size_t index = index_picker->pick_random(0, static_cast<int>(available_groups), true, use_gpu);
            if (index >= available_groups) return static_cast<size_t>(-1);
            return last_group - (available_groups - index - 1);
        } else {
            auto *weight_picker = dynamic_cast<WeightBasedRandomPicker *>(&picker);
            return weight_picker->pick_random(
                backward_weights,
                0,
                static_cast<int>(last_group + 1));
        }
    }
}

size_t TemporalGraph::get_untimed_group_idx(
   RandomPicker& picker,
   const size_t num_groups,
   const bool forward,
   const DualVector<double>& forward_weights,
   const DualVector<double>& backward_weights) const {

    if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
        const size_t index = index_picker->pick_random(0, static_cast<int>(num_groups), !forward, use_gpu);
        if (index >= num_groups) return static_cast<size_t>(-1);
        return index;
    } else {
        auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
        if (forward) {
            return weight_picker->pick_random(
                forward_weights,
                0,
                static_cast<int>(num_groups));
        } else {
            return weight_picker->pick_random(
                backward_weights,
                0,
                static_cast<int>(num_groups));
        }
    }
}

std::tuple<int, int, int64_t> TemporalGraph::get_edge_at(
   RandomPicker& picker,
   const int64_t timestamp,
   const bool forward) const {

    if (edges.empty()) return {-1, -1, -1};

    const size_t num_groups = edges.get_timestamp_group_count();
    if (num_groups == 0) return {-1, -1, -1};

    // Get group index based on constraints
    size_t group_idx;
    if (timestamp != -1) {
        group_idx = get_timestamped_group_idx(
            edges,
            picker,
            num_groups,
            timestamp,
            forward,
            edges.forward_cumulative_weights_exponential,
            edges.backward_cumulative_weights_exponential);
    } else {
        group_idx = get_untimed_group_idx(
            picker,
            num_groups,
            forward,
            edges.forward_cumulative_weights_exponential,
            edges.backward_cumulative_weights_exponential);
    }

    if (group_idx == static_cast<size_t>(-1)) return {-1, -1, -1};

    // Get selected group's boundaries
    auto [group_start, group_end] = edges.get_timestamp_group_range(group_idx);
    if (group_start == group_end) return {-1, -1, -1};

    // Use is_valid_edge_range for validation
    if (!is_valid_edge_range(group_start, group_end, edges.size())) {
        return {-1, -1, -1};
    }

    // Random selection from the chosen group
    const size_t random_idx = group_start + get_random_number(static_cast<int>(group_end - group_start));

    return cuda_functions::get_edge_at_index(
        edges.sources,
        edges.targets,
        edges.timestamps,
        random_idx,
        edges.should_use_gpu());
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

    // Get node's group range
    const size_t group_start_offset = timestamp_group_offsets[dense_idx];
    const size_t group_end_offset = timestamp_group_offsets[dense_idx + 1];
    if (group_start_offset == group_end_offset) return {-1, -1, -1};

    size_t group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp using platform-specific search
            auto [start_pos, available] = cuda_functions::timestamped_node_group_search_forward(
                timestamp_group_indices,
                group_start_offset,
                group_end_offset,
                edge_indices,
                edges.timestamps,
                timestamp,
                edges.should_use_gpu());

            if (available == 0) return {-1, -1, -1};

            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(available), false, use_gpu);
                if (index >= available) return {-1, -1, -1};
                group_pos = start_pos + index;
            } else {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                group_pos = weight_picker->pick_random(
                    node_index.outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(start_pos),
                    static_cast<int>(group_end_offset));
            }
        } else {
            // Find first group before timestamp using platform-specific search
            auto [end_pos, available] = cuda_functions::timestamped_node_group_search_backward(
                timestamp_group_indices,
                group_start_offset,
                group_end_offset,
                edge_indices,
                edges.timestamps,
                timestamp,
                edges.should_use_gpu());

            if (available == 0) return {-1, -1, -1};

            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(available), true, use_gpu);
                if (index >= available) return {-1, -1, -1};
                group_pos = end_pos - 1 - (available - index - 1);
            } else {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
                group_pos = weight_picker->pick_random(
                    is_directed
                        ? node_index.inbound_backward_cumulative_weights_exponential
                        : node_index.outbound_backward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(end_pos));
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        const size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return {-1, -1, -1};

        if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
            const size_t index = index_picker->pick_random(0, static_cast<int>(num_groups), !forward, use_gpu);
            if (index >= num_groups) return {-1, -1, -1};
            group_pos = forward
                ? group_start_offset + index
                : group_end_offset - 1 - (num_groups - index - 1);
        } else {
            auto* weight_picker = dynamic_cast<WeightBasedRandomPicker*>(&picker);
            if (forward) {
                group_pos = weight_picker->pick_random(
                    node_index.outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset));
            } else {
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
    const size_t edge_start = timestamp_group_indices[group_pos];
    const size_t edge_end = (group_pos + 1 < group_end_offset)
        ? timestamp_group_indices[group_pos + 1]
        : (forward ? node_index.outbound_offsets[dense_idx + 1]
                  : (is_directed ? node_index.inbound_offsets[dense_idx + 1]
                                : node_index.outbound_offsets[dense_idx + 1]));

    // Use is_valid_edge_range for validation
    if (!is_valid_edge_range(edge_start, edge_end, edge_indices.size())) {
        return {-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + get_random_number(static_cast<int>(edge_end - edge_start))];

    return cuda_functions::get_edge_at_index(
        edges.sources,
        edges.targets,
        edges.timestamps,
        edge_idx,
        edges.should_use_gpu());
}

std::vector<int> TemporalGraph::get_node_ids() const {
    return node_mapping.get_active_node_ids();
}

std::vector<std::tuple<int, int, int64_t>> TemporalGraph::get_edges() {
    return edges.get_edges();
}

bool TemporalGraph::is_valid_edge_range(
        const size_t edge_start,
        const size_t edge_end,
        const size_t max_edge_index) {

    return edge_start < edge_end &&
           edge_start < max_edge_index &&
           edge_end <= max_edge_index;
}
