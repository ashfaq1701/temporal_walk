#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>
#include <cmath>
#include "../cuda/dual_vector.cuh"

#ifdef HAS_CUDA
#include <thrust/binary_search.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/iterator_traits.h>
#endif

struct EdgeData {
    bool use_gpu;

    // Core edge data
    DualVector<int> sources;
    DualVector<int> targets;
    DualVector<int64_t> timestamps;

    // Timestamp grouping
    DualVector<size_t> timestamp_group_offsets;     // Start of each timestamp group
    DualVector<int64_t> unique_timestamps;          // Corresponding unique timestamps

    DualVector<double> forward_cumulative_weights_exponential;  // For forward temporal sampling
    DualVector<double> backward_cumulative_weights_exponential; // For backward temporal sampling

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);

    std::vector<std::tuple<int, int, int64_t>> get_edges();

    explicit EdgeData(bool use_gpu);

    // Group management
    void update_timestamp_groups();  // Call after sorting
    void update_temporal_weights(double timescale_bound);

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const; // For backward walks

    bool should_use_gpu() const;
};

#endif //EDGEDATA_H
