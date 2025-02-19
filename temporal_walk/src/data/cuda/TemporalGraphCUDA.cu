#include "TemporalGraphCUDA.cuh"

#ifdef HAS_CUDA

#include <thrust/binary_search.h>

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

template class TemporalGraphCUDA<GPUUsageMode::DATA_ON_GPU>;
template class TemporalGraphCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
