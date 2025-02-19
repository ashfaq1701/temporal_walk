#include "TemporalGraphCUDA.cuh"

#ifdef HAS_CUDA

#include <thrust/binary_search.h>
#include <thrust/gather.h>
#include <thrust/detail/sequence.inl>
#include <thrust/detail/sort.inl>

#include "../../random/IndexBasedRandomPicker.h"
#include "../../random/WeightBasedRandomPicker.cuh"
#include "NodeMappingCUDA.cuh"

template<GPUUsageMode GPUUsage>
void TemporalGraphCUDA<GPUUsage>::delete_old_edges() {
    if (this->time_window <= 0 || this->edges.empty()) return;

    const int64_t cutoff_time = this->latest_timestamp - this->time_window;

    // Use thrust::upper_bound instead of std::upper_bound
    auto it = thrust::upper_bound(
        this->get_policy(),
        this->edges.timestamps.begin(),
        this->edges.timestamps.end(),
        cutoff_time
    );
    if (it == this->edges.timestamps.begin()) return;

    const int delete_count = static_cast<int>(it - this->edges.timestamps.begin());
    const size_t remaining = this->edges.size() - delete_count;

    // Create bool vector for tracking nodes with edges
    typename SelectVectorType<bool, GPUUsage>::type has_edges(this->node_mapping.sparse_to_dense.size(), false);
    bool* has_edges_ptr = thrust::raw_pointer_cast(has_edges.data());

    if (remaining > 0) {
        // Move edges using thrust::copy
        thrust::copy(
            this->get_policy(),
            this->edges.sources.begin() + delete_count,
            this->edges.sources.end(),
            this->edges.sources.begin()
        );
        thrust::copy(
            this->get_policy(),
            this->edges.targets.begin() + delete_count,
            this->edges.targets.end(),
            this->edges.targets.begin()
        );
        thrust::copy(
            this->get_policy(),
            this->edges.timestamps.begin() + delete_count,
            this->edges.timestamps.end(),
            this->edges.timestamps.begin()
        );

        // Mark nodes with edges in parallel
        const int* sources_ptr = thrust::raw_pointer_cast(this->edges.sources.data());
        const int* targets_ptr = thrust::raw_pointer_cast(this->edges.targets.data());

        thrust::for_each(
            this->get_policy(),
            thrust::make_counting_iterator<size_t>(0),
            thrust::make_counting_iterator<size_t>(remaining),
            [sources_ptr, targets_ptr, has_edges_ptr] __host__ __device__ (const size_t i) {
                has_edges_ptr[sources_ptr[i]] = true;
                has_edges_ptr[targets_ptr[i]] = true;
            }
        );
    }

    this->edges.resize(remaining);

    bool* d_is_deleted = thrust::raw_pointer_cast(this->node_mapping.is_deleted.data());
    const auto is_deleted_size = this->node_mapping.is_deleted.size();

    // Mark deleted nodes in parallel
    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(has_edges.size()),
        [has_edges_ptr, d_is_deleted, is_deleted_size] __host__ __device__ (const size_t i) {
            if (!has_edges_ptr[i]) {
                mark_node_deleted(d_is_deleted, static_cast<int>(i), is_deleted_size);
            }
        }
    );

    // Update data structures
    this->edges.update_timestamp_groups();
    this->node_mapping.update(this->edges, 0, this->edges.size());
    this->node_index.rebuild(this->edges, this->node_mapping, this->is_directed);
}

template<GPUUsageMode GPUUsage>
void TemporalGraphCUDA<GPUUsage>::sort_and_merge_edges(const size_t start_idx) {
    if (start_idx >= this->edges.size()) return;

    // Create index array
    typename TemporalGraph<GPUUsage>::SizeVector indices(this->edges.size() - start_idx);
    thrust::sequence(
        this->get_policy(),
        indices.begin(),
        indices.end(),
        start_idx
    );

    // Sort indices based on timestamps
    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges.timestamps.data());
    thrust::sort(
        this->get_policy(),
        indices.begin(),
        indices.end(),
        [timestamps_ptr] __host__ __device__ (const size_t i, const size_t j) {
            return timestamps_ptr[i] < timestamps_ptr[j];
        }
    );

    // Create temporary vectors for sorted data
    typename TemporalGraph<GPUUsage>::IntVector sorted_sources(this->edges.size() - start_idx);
    typename TemporalGraph<GPUUsage>::IntVector sorted_targets(this->edges.size() - start_idx);
    typename TemporalGraph<GPUUsage>::Int64TVector sorted_timestamps(this->edges.size() - start_idx);

    // Apply permutation using gather
    thrust::gather(
        this->get_policy(),
        indices.begin(),
        indices.end(),
        this->edges.sources.begin(),
        sorted_sources.begin()
    );
    thrust::gather(
        this->get_policy(),
        indices.begin(),
        indices.end(),
        this->edges.targets.begin(),
        sorted_targets.begin()
    );
    thrust::gather(
        this->get_policy(),
        indices.begin(),
        indices.end(),
        this->edges.timestamps.begin(),
        sorted_timestamps.begin()
    );

    // Copy sorted data back
    thrust::copy(
        this->get_policy(),
        sorted_sources.begin(),
        sorted_sources.end(),
        this->edges.sources.begin() + start_idx
    );
    thrust::copy(
        this->get_policy(),
        sorted_targets.begin(),
        sorted_targets.end(),
        this->edges.targets.begin() + start_idx
    );
    thrust::copy(
        this->get_policy(),
        sorted_timestamps.begin(),
        sorted_timestamps.end(),
        this->edges.timestamps.begin() + start_idx
    );

    // Handle merging if we have existing edges
    if (start_idx > 0) {
        typename TemporalGraph<GPUUsage>::IntVector merged_sources(this->edges.size());
        typename TemporalGraph<GPUUsage>::IntVector merged_targets(this->edges.size());
        typename TemporalGraph<GPUUsage>::Int64TVector merged_timestamps(this->edges.size());

        // Create iterators for merge operation
        auto first1 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges.sources.begin(),
            this->edges.targets.begin(),
            this->edges.timestamps.begin()
        ));
        auto last1 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges.sources.begin() + start_idx,
            this->edges.targets.begin() + start_idx,
            this->edges.timestamps.begin() + start_idx
        ));
        auto first2 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges.sources.begin() + start_idx,
            this->edges.targets.begin() + start_idx,
            this->edges.timestamps.begin() + start_idx
        ));
        auto last2 = thrust::make_zip_iterator(thrust::make_tuple(
            this->edges.sources.end(),
            this->edges.targets.end(),
            this->edges.timestamps.end()
        ));
        auto result = thrust::make_zip_iterator(thrust::make_tuple(
            merged_sources.begin(),
            merged_targets.begin(),
            merged_timestamps.begin()
        ));

        // Merge based on timestamps
        thrust::merge(
            this->get_policy(),
            first1, last1,
            first2, last2,
            result,
            [] __host__ __device__ (const thrust::tuple<int, int, int64_t>& a,
                                   const thrust::tuple<int, int, int64_t>& b) {
                return thrust::get<2>(a) <= thrust::get<2>(b);
            }
        );

        // Move merged results back
        this->edges.sources = std::move(merged_sources);
        this->edges.targets = std::move(merged_targets);
        this->edges.timestamps = std::move(merged_timestamps);
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const {
    if (this->edges.empty()) return 0;

    const auto it = thrust::lower_bound(
        this->get_policy(),
        this->edges.unique_timestamps.begin(),
        this->edges.unique_timestamps.end(),
        timestamp);
    return it - this->edges.unique_timestamps.begin();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const {
    if (this->edges.empty()) return 0;

    auto it = thrust::upper_bound(
        this->get_policy(),
        this->edges.unique_timestamps.begin(),
        this->edges.unique_timestamps.end(),
        timestamp);
    return this->edges.unique_timestamps.end() - it;
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const {
    // Used for backward walks
    const int dense_idx = this->node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = this->is_directed ?
        this->node_index.inbound_timestamp_group_offsets : this->node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = this->is_directed ?
        this->node_index.inbound_timestamp_group_indices : this->node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = this->is_directed ?
        this->node_index.inbound_indices : this->node_index.outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges.timestamps.data());
    const unsigned long* edge_indices_ptr = thrust::raw_pointer_cast(edge_indices.data());

    // Binary search on group indices
    auto it = thrust::lower_bound(
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices_ptr] __host__ __device__ (const size_t group_pos, const int64_t ts)
        {
            return timestamps_ptr[edge_indices_ptr[group_pos]] < ts;
        });

    return std::distance(timestamp_group_indices.begin() + static_cast<int>(group_start), it);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraphCUDA<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const {
    // Used for forward walks
    int dense_idx = this->node_mapping.to_dense(node_id);
    if (dense_idx < 0) return 0;

    const auto& timestamp_group_offsets = this->node_index.outbound_timestamp_group_offsets;
    const auto& timestamp_group_indices = this->node_index.outbound_timestamp_group_indices;
    const auto& edge_indices = this->node_index.outbound_indices;

    const size_t group_start = timestamp_group_offsets[dense_idx];
    const size_t group_end = timestamp_group_offsets[dense_idx + 1];
    if (group_start == group_end) return 0;

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges.timestamps.data());
    const unsigned long* edge_indices_ptr = thrust::raw_pointer_cast(edge_indices.data());

    // Binary search on group indices
    const auto it = std::upper_bound(
        timestamp_group_indices.begin() + static_cast<int>(group_start),
        timestamp_group_indices.begin() + static_cast<int>(group_end),
        timestamp,
        [timestamps_ptr, edge_indices_ptr] __host__ __device__ (const int64_t ts, const size_t group_pos)
        {
            return ts < timestamps_ptr[edge_indices_ptr[group_pos]];
        });

    return std::distance(it, timestamp_group_indices.begin() + static_cast<int>(group_end));
}

template<GPUUsageMode GPUUsage>
std::tuple<int, int, int64_t> TemporalGraphCUDA<GPUUsage>::get_node_edge_at(
    const int node_id,
    RandomPicker& picker,
    const int64_t timestamp,
    const bool forward) const {

    const int dense_idx = this->node_mapping.to_dense(node_id);
    if (dense_idx < 0) return {-1, -1, -1};

    // Get appropriate node indices based on direction and graph type
    const auto& timestamp_group_offsets = forward
        ? this->node_index.outbound_timestamp_group_offsets
        : (this->is_directed ? this->node_index.inbound_timestamp_group_offsets : this->node_index.outbound_timestamp_group_offsets);

    const auto& timestamp_group_indices = forward
        ? this->node_index.outbound_timestamp_group_indices
        : (this->is_directed ? this->node_index.inbound_timestamp_group_indices : this->node_index.outbound_timestamp_group_indices);

    const auto& edge_indices = forward
        ? this->node_index.outbound_indices
        : (this->is_directed ? this->node_index.inbound_indices : this->node_index.outbound_indices);

    // Get node's group range
    const size_t group_start_offset = timestamp_group_offsets[dense_idx];
    const size_t group_end_offset = timestamp_group_offsets[dense_idx + 1];
    if (group_start_offset == group_end_offset) return {-1, -1, -1};

    const int64_t* timestamps_ptr = thrust::raw_pointer_cast(this->edges.timestamps.data());
    const unsigned long* edge_indices_ptr = thrust::raw_pointer_cast(edge_indices.data());

    size_t group_pos;
    if (timestamp != -1) {
        if (forward) {
            // Find first group after timestamp
            auto it = thrust::upper_bound(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [timestamps_ptr, edge_indices_ptr] __host__ __device__ (const int64_t ts, const size_t pos) {
                    return ts < timestamps_ptr[edge_indices_ptr[pos]];
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
                    this->node_index.outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(start_pos),
                    static_cast<int>(group_end_offset));
            }
        } else {
            // Find first group >= timestamp
            auto it = thrust::lower_bound(
                timestamp_group_indices.begin() + static_cast<int>(group_start_offset),
                timestamp_group_indices.begin() + static_cast<int>(group_end_offset),
                timestamp,
                [timestamps_ptr, edge_indices_ptr] __host__ __device__ (const size_t pos, const int64_t ts) {
                    return timestamps_ptr[edge_indices_ptr[pos]] < ts;
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
                    this->is_directed
                        ? this->node_index.inbound_backward_cumulative_weights_exponential
                        : this->node_index.outbound_backward_cumulative_weights_exponential,
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
                    this->node_index.outbound_forward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset));
            }
            else
            {
                group_pos = weight_picker->pick_random(
                    this->is_directed
                        ? this->node_index.inbound_backward_cumulative_weights_exponential
                        : this->node_index.outbound_backward_cumulative_weights_exponential,
                    static_cast<int>(group_start_offset),
                    static_cast<int>(group_end_offset));
            }
        }
    }

    // Get edge range for selected group
    const size_t edge_start = timestamp_group_indices[group_pos];
    const size_t edge_end = (group_pos + 1 < group_end_offset)
        ? timestamp_group_indices[group_pos + 1]
        : (forward ? this->node_index.outbound_offsets[dense_idx + 1]
                  : (this->is_directed ? this->node_index.inbound_offsets[dense_idx + 1]
                                : this->node_index.outbound_offsets[dense_idx + 1]));

    // Validate range before random selection
    if (edge_start >= edge_end || edge_start >= edge_indices.size() || edge_end > edge_indices.size()) {
        return {-1, -1, -1};
    }

    // Random selection from group
    const size_t edge_idx = edge_indices[edge_start + generate_random_number_bounded_by(static_cast<int>(edge_end - edge_start))];

    return {
        this->edges.sources[edge_idx],
        this->edges.targets[edge_idx],
        this->edges.timestamps[edge_idx]
    };
}

template class TemporalGraphCUDA<GPUUsageMode::DATA_ON_GPU>;
template class TemporalGraphCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
