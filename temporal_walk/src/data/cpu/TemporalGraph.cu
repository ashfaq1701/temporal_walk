#include "TemporalGraph.cuh"
#include <algorithm>

#include "../../random/IndexBasedRandomPicker.h"
#include "../../random/WeightBasedRandomPicker.cuh"
#include "../../random/RandomPicker.h"

#include "../../utils/utils.h"

#ifdef HAS_CUDA
#include <thrust/sort.h>
#include <thrust/execution_policy.h>
#endif


#ifdef HAS_CUDA
__device__ bool compare_timestamps(const int64_t* ts, const size_t i, const size_t j) {
    return ts[i] < ts[j];
}
#endif

template<GPUUsageMode GPUUsage>
TemporalGraph<GPUUsage>::TemporalGraph(
    const bool directed,
    const int64_t window,
    const bool enable_weight_computation,
    const double timescale_bound)
    : is_directed(directed)
    , time_window(window)
    , enable_weight_computation(enable_weight_computation)
    , timescale_bound(timescale_bound)
    , latest_timestamp(0)
    , node_index()
    , edges()
    , node_mapping()
    {}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& new_edges) {
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

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::update_temporal_weights() {
    edges.update_temporal_weights(timescale_bound);
    node_index.update_temporal_weights(edges, timescale_bound);
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::sort_and_merge_edges(const size_t start_idx) {
    if (start_idx >= edges.size()) return;

    // Sort new edges first
    SizeVector indices(edges.size() - start_idx);
    for (size_t i = 0; i < indices.size(); i++) {
        indices[i] = start_idx + i;
    }

    #ifdef HAS_CUDA
    if constexpr (GPUUsage == GPUUsageMode::DATA_ON_GPU) {
        thrust::sort(thrust::device, indices.begin(), indices.end(),
            [ts = edges.timestamps.data().get()] __device__ (const size_t i, const size_t j) {
                return compare_timestamps(ts, i, j);
            });
    } else
    #endif
    {
        std::sort(indices.begin(), indices.end(),
            [this](const size_t i, const size_t j) {
                return edges.timestamps[i] < edges.timestamps[j];
            });
    }

    // Apply permutation in-place using temporary vectors
    IntVector sorted_sources(edges.size() - start_idx);
    IntVector sorted_targets(edges.size() - start_idx);
    Int64TVector sorted_timestamps(edges.size() - start_idx);

    for (size_t i = 0; i < indices.size(); i++) {
        const size_t idx = indices[i];
        sorted_sources[i] = edges.sources[idx];
        sorted_targets[i] = edges.targets[idx];
        sorted_timestamps[i] = edges.timestamps[idx];
    }

    // Copy back sorted edges
    for (size_t i = 0; i < indices.size(); i++) {
        edges.sources[start_idx + i] = sorted_sources[i];
        edges.targets[start_idx + i] = sorted_targets[i];
        edges.timestamps[start_idx + i] = sorted_timestamps[i];
    }

    // Merge with existing edges
    if (start_idx > 0) {
        // Create buffer vectors
        IntVector merged_sources(edges.size());
        IntVector merged_targets(edges.size());
        Int64TVector merged_timestamps(edges.size());

        size_t i = 0;  // Index for existing edges
        size_t j = start_idx;  // Index for new edges
        size_t k = 0;  // Index for merged result

        // Merge while keeping arrays aligned
        while (i < start_idx && j < edges.size()) {
            if (edges.timestamps[i] <= edges.timestamps[j]) {
                merged_sources[k] = edges.sources[i];
                merged_targets[k] = edges.targets[i];
                merged_timestamps[k] = edges.timestamps[i];
                i++;
            } else {
                merged_sources[k] = edges.sources[j];
                merged_targets[k] = edges.targets[j];
                merged_timestamps[k] = edges.timestamps[j];
                j++;
            }
            k++;
        }

        // Copy remaining entries
        while (i < start_idx) {
            merged_sources[k] = edges.sources[i];
            merged_targets[k] = edges.targets[i];
            merged_timestamps[k] = edges.timestamps[i];
            i++;
            k++;
        }

        while (j < edges.size()) {
            merged_sources[k] = edges.sources[j];
            merged_targets[k] = edges.targets[j];
            merged_timestamps[k] = edges.timestamps[j];
            j++;
            k++;
        }

        // Copy merged data back to edges
        edges.sources = std::move(merged_sources);
        edges.targets = std::move(merged_targets);
        edges.timestamps = std::move(merged_timestamps);
    }
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::delete_old_edges() {
    if (time_window <= 0 || edges.empty()) return;

    const int64_t cutoff_time = latest_timestamp - time_window;
    const auto it = std::upper_bound(edges.timestamps.begin(), edges.timestamps.end(), cutoff_time);
    if (it == edges.timestamps.begin()) return;

    const int delete_count = static_cast<int>(it - edges.timestamps.begin());
    const size_t remaining = edges.size() - delete_count;

    // Track which nodes still have edges
    BoolVector has_edges(node_mapping.sparse_to_dense.size(), false);

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

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    const auto it = std::lower_bound(edges.unique_timestamps.begin(), edges.unique_timestamps.end(), timestamp);
    return it - edges.unique_timestamps.begin();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const {
    if (edges.empty()) return 0;

    auto it = std::upper_bound(edges.unique_timestamps.begin(), edges.unique_timestamps.end(), timestamp);
    return edges.unique_timestamps.end() - it;
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    const int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = is_directed ? node_index.inbound_timestamp_group_offsets : node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = is_directed ? node_index.inbound_timestamp_group_indices : node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = is_directed ? node_index.inbound_indices : node_index.outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    auto it = std::lower_bound(
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [this, &edge_indices](size_t group_pos, int64_t ts)
        {
            return edges.timestamps[edge_indices[group_pos]] < ts;
        });

    return std::distance(timestamp_group_indices.begin() + static_cast<int>(group_start), it);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = node_index.outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    // Binary search on group indices
    const auto it = std::upper_bound(
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [this, &edge_indices](int64_t ts, size_t group_pos)
        {
            return ts < edges.timestamps[edge_indices[group_pos]];
        });

    return std::distance(it, timestamp_group_indices.begin() + static_cast<int>(group_end));
}

template<GPUUsageMode GPUUsage>
std::tuple<int, int, int64_t> TemporalGraph<GPUUsage>::get_edge_at(
    RandomPicker& picker,
    int64_t timestamp,
    const bool forward) const {

    if (edges.empty()) return {-1, -1, -1};

    const size_t num_groups = edges.get_timestamp_group_count();
    if (num_groups == 0) return {-1, -1, -1};

    size_t group_idx;
    if (timestamp != -1) {
        if (forward) {
            const size_t first_group = edges.find_group_after_timestamp(timestamp);
            const size_t available_groups = num_groups - first_group;
            if (available_groups == 0) return {-1, -1, -1};

            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(available_groups), false);
                if (index >= available_groups) return {-1, -1, -1};
                group_idx = first_group + index;
            }
            else {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker<GPUUsage>*>(&picker);
                group_idx = weight_picker->pick_random(
                    edges.forward_cumulative_weights_exponential,
                    static_cast<int>(first_group),
                    static_cast<int>(num_groups));
            }
        } else {
            const size_t last_group = edges.find_group_before_timestamp(timestamp);
            if (last_group == static_cast<size_t>(-1)) return {-1, -1, -1};

            const size_t available_groups = last_group + 1;
            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(available_groups), true);
                if (index >= available_groups) return {-1, -1, -1};
                group_idx = last_group - (available_groups - index - 1);
            }
            else {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker<GPUUsage>*>(&picker);
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
            if (index >= num_groups) return {-1, -1, -1};
            group_idx = index;
        } else {
            auto* weight_picker = dynamic_cast<WeightBasedRandomPicker<GPUUsage>*>(&picker);
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
        return {-1, -1, -1};
    }

    // Random selection from the chosen group
    const size_t random_idx = group_start + generate_random_number_bounded_by(static_cast<int>(group_end - group_start));
    return {
        edges.sources[random_idx],
        edges.targets[random_idx],
        edges.timestamps[random_idx]
    };
}

template<GPUUsageMode GPUUsage>
std::tuple<int, int, int64_t> TemporalGraph<GPUUsage>::get_node_edge_at(
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
            // Find first group after timestamp
            auto it = std::upper_bound(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [this, &edge_indices](int64_t ts, size_t pos) {
                    return ts < edges.timestamps[edge_indices[pos]];
                });

            // Count available groups after timestamp
            const size_t available = timestamp_group_indices.begin() +
                static_cast<int>(group_end_offset) - it;
            if (available == 0) return {-1, -1, -1};

            const size_t start_pos = it - timestamp_group_indices.begin();
            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(available), false);
                if (index >= available) return {-1, -1, -1};
                group_pos = start_pos + index;
            }
            else
            {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker<GPUUsage>*>(&picker);
                group_pos = weight_picker->pick_random(
                    node_index.outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(start_pos),
                    static_cast<int>(group_end_offset));
            }
        } else {
            // Find first group >= timestamp
            auto it = std::lower_bound(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [this, &edge_indices](size_t pos, int64_t ts) {
                    return edges.timestamps[edge_indices[pos]] < ts;
                });

            const size_t available = it - (timestamp_group_indices.begin() +
                static_cast<int>(group_start_offset));
            if (available == 0) return {-1, -1, -1};

            if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
                const size_t index = index_picker->pick_random(0, static_cast<int>(available), true);
                if (index >= available) return {-1, -1, -1};
                group_pos = (it - timestamp_group_indices.begin()) - 1 - (available - index - 1);
            }
            else
            {
                auto* weight_picker = dynamic_cast<WeightBasedRandomPicker<GPUUsage>*>(&picker);
                group_pos = weight_picker->pick_random(
                    is_directed
                        ? node_index.inbound_backward_cumulative_weights_exponential
                        : node_index.outbound_backward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset), // start from node's first group
                    static_cast<int>(it - timestamp_group_indices.begin()) // up to and excluding first group >= timestamp
                );
            }
        }
    } else {
        // No timestamp constraint - select from all groups
        const size_t num_groups = group_end_offset - group_start_offset;
        if (num_groups == 0) return {-1, -1, -1};

        if (auto* index_picker = dynamic_cast<IndexBasedRandomPicker*>(&picker)) {
            const size_t index = index_picker->pick_random(0, static_cast<int>(num_groups), !forward);
            if (index >= num_groups) return {-1, -1, -1};
            group_pos = forward
                ? group_start_offset + index
                : group_end_offset - 1 - (num_groups - index - 1);
        }
        else
        {
            auto* weight_picker = dynamic_cast<WeightBasedRandomPicker<GPUUsage>*>(&picker);
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
    const size_t edge_start = timestamp_group_indices[group_pos];
    const size_t edge_end = (group_pos + 1 < group_end_offset)
        ? timestamp_group_indices[group_pos + 1]
        : (forward ? node_index.outbound_offsets[dense_idx + 1]
                  : (is_directed ? node_index.inbound_offsets[dense_idx + 1]
                                : node_index.outbound_offsets[dense_idx + 1]));

    // Validate range before random selection
    if (edge_start >= edge_end || edge_start >= edge_indices.size() || edge_end > edge_indices.size()) {
        return {-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + generate_random_number_bounded_by(static_cast<int>(edge_end - edge_start))];

    return {
        edges.sources[edge_idx],
        edges.targets[edge_idx],
        edges.timestamps[edge_idx]
    };
}

template<GPUUsageMode GPUUsage>
std::vector<int> TemporalGraph<GPUUsage>::get_node_ids() const {
    return node_mapping.get_active_node_ids();
}

template<GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> TemporalGraph<GPUUsage>::get_edges() {
    return edges.get_edges();
}

template class TemporalGraph<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalGraph<GPUUsageMode::DATA_ON_GPU>;
template class TemporalGraph<GPUUsageMode::DATA_ON_HOST>;
#endif
