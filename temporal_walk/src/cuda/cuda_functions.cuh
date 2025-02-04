#ifndef CUDA_FUNCTIONS_CUH
#define CUDA_FUNCTIONS_CUH

#include <algorithm>
#include <numeric>
#include <random/IndexBasedRandomPicker.h>

#include <random/WeightBasedRandomPicker.cuh>

#include "DualVector.cuh"

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/count.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/sort.h>
#include <thrust/detail/gather.inl>
#endif

class RandomPicker;

namespace cuda_functions {

    template<typename T>
    double get_value_at(
        const DualVector<T>& vec, int index, bool use_gpu) {
        #ifdef HAS_CUDA
        return use_gpu ? vec.device_at(index) : vec.host_at(index);
        #else
        return vec.host_at(index);
        #endif
    }

    // In cuda_functions.cuh
    template<typename T>
    int find_max_node_id(
        const DualVector<T> &sources,
        const DualVector<T> &targets,
        const size_t start_idx,
        const size_t end_idx,
        const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            // Find max in sources
            const int max_sources = thrust::reduce(
                thrust::device,
                sources.device_begin() + start_idx,
                sources.device_begin() + end_idx,
                0,
                thrust::maximum<int>());

            // Find max in targets
            const int max_targets = thrust::reduce(
                thrust::device,
                targets.device_begin() + start_idx,
                targets.device_begin() + end_idx,
                0,
                thrust::maximum<int>());

            return std::max(max_sources, max_targets);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        // CPU version
        int max_id = 0;
        for (size_t i = start_idx; i < end_idx; i++) {
            max_id = std::max({max_id, sources[i], targets[i]});
        }
        return max_id;
    }

    template<typename T>
   void mark_nodes_not_deleted(
       const DualVector<T>& sources,
       const DualVector<T>& targets,
       DualVector<short>& is_deleted,
       const size_t start_idx,
       const size_t end_idx,
       const bool use_gpu,
       short not_deleted_value) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto sources_ptr = sources.device_data().get();
            auto targets_ptr = targets.device_data().get();
            auto is_deleted_ptr = is_deleted.device_data().get();

            thrust::for_each(thrust::device,
                thrust::counting_iterator<size_t>(start_idx),
                thrust::counting_iterator<size_t>(end_idx),
                [sources_ptr, targets_ptr, is_deleted_ptr, not_deleted_value] __device__ (size_t i) {
                    is_deleted_ptr[sources_ptr[i]] = not_deleted_value;
                    is_deleted_ptr[targets_ptr[i]] = not_deleted_value;
                });
            #endif
        } else {
            for (size_t i = start_idx; i < end_idx; i++) {
                is_deleted[sources[i]] = not_deleted_value;
                is_deleted[targets[i]] = not_deleted_value;
            }
        }
    }

    // Platform-specific upper_bound and distance calculation
    inline size_t find_upper_bound_position(const DualVector<int64_t>& vec, const int64_t value, const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            const auto it = thrust::upper_bound(thrust::device,
                vec.device_begin(),
                vec.device_end(),
                value);
            return thrust::distance(vec.device_begin(), it);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        const auto it = std::upper_bound(
            vec.host_begin(),
            vec.host_end(),
            value);
        return std::distance(vec.host_begin(), it);
    }

    // Platform-specific lower_bound and distance calculation
    inline size_t find_lower_bound_position(const DualVector<int64_t>& vec, const int64_t value, const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            const auto it = thrust::lower_bound(thrust::device,
                vec.device_begin(),
                vec.device_end(),
                value);
            return thrust::distance(vec.device_begin(), it);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        const auto it = std::lower_bound(
            vec.host_begin(),
            vec.host_end(),
            value);
        return std::distance(vec.host_begin(), it);
    }

    // For computing prefix sums/cumulative sums
    template<typename T>
    void compute_prefix_sum(DualVector<T>& values, const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::inclusive_scan(thrust::device,
                values.device_begin(),
                values.device_end(),
                values.device_begin());
            return;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        for (size_t i = 1; i < values.size(); i++) {
            values[i] += values[i-1];
        }
    }

    template<typename T>
    void compute_cumulative_weights(DualVector<T>& weights, double total_sum, size_t start, size_t end, bool use_gpu) {
        if (start >= end || end > weights.size()) {
            return;
        }

        if (use_gpu) {
            #ifdef HAS_CUDA
            // First normalize weights
            auto weight_data = weights.device_data().get();
            const T inv_sum = static_cast<T>(1.0 / total_sum);
            thrust::for_each(thrust::device,
                thrust::counting_iterator<size_t>(start),
                thrust::counting_iterator<size_t>(end),
                [=] __device__ (size_t i) {
                    weight_data[i] *= inv_sum;
                }
            );

            // Then compute prefix sum
            thrust::inclusive_scan(thrust::device,
                weights.device_begin() + start,
                weights.device_begin() + end,
                weights.device_begin() + start
            );
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        } else {
            // First normalize
            const T inv_sum = static_cast<T>(1.0 / total_sum);
            for (size_t i = start; i < end; i++) {
                weights[i] *= inv_sum;
            }

            // Then compute prefix sum
            for (size_t i = start + 1; i < end; i++) {
                weights[i] += weights[i-1];
            }
        }
    }

    template<typename T>
    size_t count_matching(const DualVector<T>& vec, const T target_value, const bool use_gpu) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return thrust::count(
                thrust::device,
                vec.device_begin(),
                vec.device_end(),
                target_value
            );
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return std::count(vec.host_begin(), vec.host_end(), target_value);
    }

    template<typename T>
    std::vector<T> filter_active_nodes(
        const DualVector<T>& nodes,
        const DualVector<short>& is_deleted,
        short target_value,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            // Create device vector for results
            thrust::device_vector<T> d_result(nodes.size());

            // Use copy_if directly with device vectors
            const auto end = thrust::copy_if(
                thrust::device,
                nodes.device_begin(),
                nodes.device_end(),
                d_result.begin(),
                [deleted_ptr = is_deleted.device_data().get(), target = target_value]
                __device__ (T sparse_id) {
                    return deleted_ptr[sparse_id] == target;
                }
            );

            // Copy results back to host
            std::vector<T> result(thrust::distance(d_result.begin(), end));
            thrust::copy(d_result.begin(), end, result.begin());
            return result;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        // CPU version
        std::vector<T> result;
        result.reserve(nodes.size());
        for (auto it = nodes.host_begin(); it != nodes.host_end(); ++it) {
            if (int sparse_id = *it; is_deleted[sparse_id] == target_value) {
                result.push_back(sparse_id);
            }
        }
        return result;
    }

    // For sorting edges by timestamp
    template<typename TimestampType>
    void sort_edges_by_timestamp(
        std::vector<size_t>& indices,
        const DualVector<TimestampType>& timestamps,
        const size_t start_idx,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::device_vector<size_t> d_indices(indices.size());
            thrust::sequence(thrust::device, d_indices.begin(), d_indices.end(), start_idx);

            thrust::sort(thrust::device,
                d_indices.begin(),
                d_indices.end(),
                [ts = timestamps.device_data().get()] __device__ (const size_t i, const size_t j) {
                    return ts[static_cast<int>(i)] < ts[static_cast<int>(j)];
                });

            thrust::copy(d_indices.begin(), d_indices.end(), indices.begin());
            return;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        std::iota(indices.begin(), indices.end(), start_idx);
        std::sort(indices.begin(), indices.end(),
            [&timestamps](const size_t i, const size_t j) {
                return timestamps[i] < timestamps[j];
            });
    }

    // For gathering sorted edges
    template<typename T>
    void gather_sorted_edges(
        const std::vector<size_t>& indices,
        const DualVector<T>& source,
        DualVector<T>& dest,
        const size_t start_idx,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::device_vector<size_t> d_indices = indices;
            thrust::device_vector<T> d_result(indices.size());

            thrust::gather(thrust::device,
                d_indices.begin(), d_indices.end(),
                source.device_begin(),
                d_result.begin());

            thrust::copy(thrust::device,
                d_result.begin(), d_result.end(),
                dest.device_begin() + static_cast<int>(start_idx));
            return;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        std::vector<T> result(indices.size());
        for (size_t i = 0; i < indices.size(); i++) {
            result[i] = source[indices[i]];
        }
        for (size_t i = 0; i < indices.size(); i++) {
            dest[start_idx + i] = result[i];
        }
    }

    // For merging sorted edges
    template<typename T, typename V>
    void merge_sorted_edges(
        DualVector<T>& timestamps,
        DualVector<V>& sources,
        DualVector<V>& targets,
        const size_t start_idx,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::device_vector<T> merged_timestamps(timestamps.size());
            thrust::device_vector<T> merged_sources(sources.size());
            thrust::device_vector<T> merged_targets(targets.size());

            thrust::merge_by_key(
                thrust::device,
                timestamps.device_begin(),
                timestamps.device_begin() + static_cast<int>(start_idx),
                timestamps.device_begin() + static_cast<int>(start_idx),
                timestamps.device_end(),
                thrust::make_zip_iterator(thrust::make_tuple(
                    sources.device_begin(),
                    targets.device_begin()
                )),
                thrust::make_zip_iterator(thrust::make_tuple(
                    sources.device_begin() + static_cast<int>(start_idx),
                    targets.device_begin() + static_cast<int>(sources.size())
                )),
                merged_timestamps.begin(),
                thrust::make_zip_iterator(thrust::make_tuple(
                    merged_sources.begin(),
                    merged_targets.begin()
                ))
            );

            timestamps.set_device_vector(std::move(merged_timestamps));
            sources.set_device_vector(std::move(merged_sources));
            targets.set_device_vector(std::move(merged_targets));
            return;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        // Original CPU merge implementation remains the same
        std::vector<T> merged_timestamps(timestamps.size());
        std::vector<V> merged_sources(sources.size());
        std::vector<V> merged_targets(targets.size());

        size_t i = 0;
        size_t j = start_idx;
        size_t k = 0;

        while (i < start_idx && j < timestamps.size()) {
            if (timestamps[i] <= timestamps[j]) {
                merged_timestamps[k] = timestamps[i];
                merged_sources[k] = sources[i];
                merged_targets[k] = targets[i];
                i++;
            } else {
                merged_timestamps[k] = timestamps[j];
                merged_sources[k] = sources[j];
                merged_targets[k] = targets[j];
                j++;
            }
            k++;
        }

        while (i < start_idx) {
            merged_timestamps[k] = timestamps[i];
            merged_sources[k] = sources[i];
            merged_targets[k] = targets[i];
            i++;
            k++;
        }

        while (j < timestamps.size()) {
            merged_timestamps[k] = timestamps[j];
            merged_sources[k] = sources[j];
            merged_targets[k] = targets[j];
            j++;
            k++;
        }

        timestamps.set_host_vector(std::move(merged_timestamps));
        sources.set_host_vector(std::move(merged_sources));
        targets.set_host_vector(std::move(merged_targets));
    }

    // Find cutoff point for deletion
    template<typename T>
    std::pair<size_t, size_t> find_cutoff_edges(
        const DualVector<T>& timestamps,
        T cutoff_time,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto it = thrust::upper_bound(thrust::device,
                timestamps.device_begin(),
                timestamps.device_end(),
                cutoff_time);

            size_t delete_count = thrust::distance(timestamps.device_begin(), it);
            return {delete_count, timestamps.size() - delete_count};
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::upper_bound(
            timestamps.host_begin(),
            timestamps.host_end(),
            cutoff_time);

        size_t delete_count = std::distance(timestamps.host_begin(), it);
        return {delete_count, timestamps.size() - delete_count};
    }

    // Move remaining edges after deletion
    template<typename T>
    void compact_edges_after_deletion(
        DualVector<T>& vec,
        size_t delete_count,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::copy(thrust::device,
                vec.device_begin() + delete_count,
                vec.device_end(),
                vec.device_begin());
            return;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        std::move(vec.host_begin() + delete_count,
                 vec.host_end(),
                 vec.host_begin());
    }

    // Mark nodes that still have edges
    template<typename T, typename V>
    void mark_remaining_edges(
        const DualVector<T>& sources,
        const DualVector<T>& targets,
        DualVector<V>& has_edges,
        const size_t remaining,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto src_data = sources.device_data().get();
            auto tgt_data = targets.device_data().get();
            auto has_edges_data = has_edges.device_data().get();

            thrust::for_each(thrust::device,
                thrust::counting_iterator<size_t>(0),
                thrust::counting_iterator<size_t>(remaining),
                [=] __device__ (const size_t i) {
                    has_edges_data[src_data[static_cast<int>(i)]] = 1;
                    has_edges_data[tgt_data[static_cast<int>(i)]] = 1;
                });
            return;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        for (size_t i = 0; i < remaining; i++) {
            has_edges[sources[i]] = 1;
            has_edges[targets[i]] = 1;
        }
    }

    template<typename T>
    size_t count_less_than(
        const DualVector<T>& vec,
        const T value,
        const bool use_gpu) {

        if (vec.empty()) return 0;

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto it = thrust::lower_bound(thrust::device,
                vec.device_begin(),
                vec.device_end(),
                value);
            return thrust::distance(vec.device_begin(), it);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::lower_bound(
            vec.host_begin(),
            vec.host_end(),
            value);
        return std::distance(vec.host_begin(), it);
    }

    template<typename T>
    size_t count_greater_than(
        const DualVector<T>& vec,
        const T value,
        const bool use_gpu) {

        if (vec.empty()) return 0;

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto it = thrust::upper_bound(thrust::device,
                vec.device_begin(),
                vec.device_end(),
                value);
            return thrust::distance(it, vec.device_end());
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::upper_bound(
            vec.host_begin(),
            vec.host_end(),
            value);
        return std::distance(it, vec.host_end());
    }

    // Helper function for timestamp comparisons
    template<typename T, typename V>
    size_t count_node_timestamps_less_than(
        const DualVector<T>& timestamp_group_indices,
        const size_t group_start,
        const size_t group_end,
        const DualVector<T>& edge_indices,
        const DualVector<V>& timestamps,
        V timestamp,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto timestamps_data = timestamps.device_data().get();
            auto edge_indices_data = edge_indices.device_data().get();

            auto it = thrust::lower_bound(
                thrust::device,
                timestamp_group_indices.device_begin() + static_cast<int>(group_start),
                timestamp_group_indices.device_begin() + static_cast<int>(group_end),
                timestamp,
                [timestamps_data, edge_indices_data] __device__ (const size_t group_pos, const V ts) {
                    return timestamps_data[static_cast<int>(edge_indices_data[static_cast<int>(group_pos)])] < ts;
                });

            return thrust::distance(
                timestamp_group_indices.device_begin() + static_cast<int>(group_start),
                it);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::lower_bound(
            timestamp_group_indices.host_begin() + static_cast<int>(group_start),
            timestamp_group_indices.host_begin() + static_cast<int>(group_end),
            timestamp,
            [&timestamps, &edge_indices](const size_t group_pos, const V ts) {
                return timestamps[edge_indices[group_pos]] < ts;
            });

        return std::distance(
            timestamp_group_indices.host_begin() + static_cast<int>(group_start),
            it);
    }

    template<typename T, typename V>
    size_t count_node_timestamps_greater_than(
        const DualVector<T>& timestamp_group_indices,
        const size_t group_start,
        const size_t group_end,
        const DualVector<T>& edge_indices,
        const DualVector<V>& timestamps,
        V timestamp,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto timestamps_data = timestamps.device_data().get();
            auto edge_indices_data = edge_indices.device_data().get();

            auto it = thrust::upper_bound(
                thrust::device,
                timestamp_group_indices.device_begin() + static_cast<int>(group_start),
                timestamp_group_indices.device_begin() + static_cast<int>(group_end),
                timestamp,
                [timestamps_data, edge_indices_data] __device__ (const V ts, const size_t group_pos) {
                    return ts < timestamps_data[static_cast<int>(edge_indices_data[static_cast<int>(group_pos)])];
                });

            return thrust::distance(it,
                timestamp_group_indices.device_begin() + static_cast<int>(group_end));
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::upper_bound(
            timestamp_group_indices.host_begin() + static_cast<int>(group_start),
            timestamp_group_indices.host_begin() + static_cast<int>(group_end),
            timestamp,
            [&timestamps, &edge_indices](const V ts, const size_t group_pos) {
                return ts < timestamps[edge_indices[group_pos]];
            });

        return std::distance(it,
            timestamp_group_indices.host_begin() + static_cast<int>(group_end));
    }

    // Helper for getting edge data at index
    template<typename T, typename V>
    std::tuple<T, T, V> get_edge_at_index(
        const DualVector<T>& sources,
        const DualVector<T>& targets,
        const DualVector<V>& timestamps,
        size_t idx,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            return {
                sources.device_at(idx),
                targets.device_at(idx),
                timestamps.device_at(idx)
            };
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        return {
            sources.host_at(idx),
            targets.host_at(idx),
            timestamps.host_at(idx)
        };
    }

    template<typename T, typename V>
    std::pair<size_t, size_t> timestamped_node_group_search_forward(
        const DualVector<T>& timestamp_group_indices,
        const size_t group_start,
        const size_t group_end,
        const DualVector<T>& edge_indices,
        const DualVector<V>& timestamps,
        const V timestamp,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto ts_data = timestamps.device_data().get();
            auto edge_idx_data = edge_indices.device_data().get();

            auto it = thrust::upper_bound(
                thrust::device,
                timestamp_group_indices.device_begin() + static_cast<int>(group_start),
                timestamp_group_indices.device_begin() + static_cast<int>(group_end),
                timestamp,
                [ts_data, edge_idx_data] __device__ (const V ts_val, const size_t pos) {
                    return ts_val < ts_data[static_cast<int>(edge_idx_data[static_cast<int>(pos)])];
                });

            const size_t available = timestamp_group_indices.device_begin() +
                static_cast<int>(group_end) - it;
            const size_t start_pos = it - timestamp_group_indices.device_begin();
            return {start_pos, available};
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::upper_bound(
            timestamp_group_indices.host_begin() + static_cast<int>(group_start),
            timestamp_group_indices.host_begin() + static_cast<int>(group_end),
            timestamp,
            [&timestamps, &edge_indices](const V ts, const size_t pos) {
                return ts < timestamps[edge_indices[pos]];
            });

        const size_t available = timestamp_group_indices.host_begin() +
            static_cast<int>(group_end) - it;
        const size_t start_pos = it - timestamp_group_indices.host_begin();
        return {start_pos, available};
    }

    template<typename T, typename V>
    std::pair<size_t, size_t> timestamped_node_group_search_backward(
        const DualVector<T>& timestamp_group_indices,
        const size_t group_start,
        const size_t group_end,
        const DualVector<T>& edge_indices,
        const DualVector<V>& timestamps,
        const V timestamp,
        const bool use_gpu) {

        if (use_gpu) {
            #ifdef HAS_CUDA
            auto ts_data = timestamps.device_data().get();
            auto edge_idx_data = edge_indices.device_data().get();

            auto it = thrust::lower_bound(
                thrust::device,
                timestamp_group_indices.device_begin() + static_cast<int>(group_start),
                timestamp_group_indices.device_begin() + static_cast<int>(group_end),
                timestamp,
                [ts_data, edge_idx_data] __device__ (const size_t pos, const V ts_val) {
                    return ts_data[static_cast<int>(edge_idx_data[static_cast<int>(pos)])] < ts_val;
                });

            const size_t available = it - (timestamp_group_indices.device_begin() +
                static_cast<int>(group_start));
            const size_t end_pos = it - timestamp_group_indices.device_begin();
            return {end_pos, available};
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }

        auto it = std::lower_bound(
            timestamp_group_indices.host_begin() + static_cast<int>(group_start),
            timestamp_group_indices.host_begin() + static_cast<int>(group_end),
            timestamp,
            [&timestamps, &edge_indices](const size_t pos, const V ts) {
                return timestamps[edge_indices[pos]] < ts;
            });

        const size_t available = it - (timestamp_group_indices.host_begin() +
            static_cast<int>(group_start));
        const size_t end_pos = it - timestamp_group_indices.host_begin();
        return {end_pos, available};
    }
}

#endif //CUDA_FUNCTIONS_CUH
