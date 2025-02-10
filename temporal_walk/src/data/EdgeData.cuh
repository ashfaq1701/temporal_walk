#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "../cuda_common/types.cuh"

class EdgeData {

public:
    bool use_gpu;

    // Core edge data
    VectorTypes<int>::Vector sources{};
    VectorTypes<int>::Vector targets{};
    VectorTypes<int64_t>::Vector timestamps{};

    // Timestamp grouping
    VectorTypes<size_t>::Vector timestamp_group_offsets{};     // Start of each timestamp group
    VectorTypes<int64_t>::Vector unique_timestamps{};          // Corresponding unique timestamps

    VectorTypes<double>::Vector forward_cumulative_weights_exponential{};  // For forward temporal sampling
    VectorTypes<double>::Vector backward_cumulative_weights_exponential{}; // For backward temporal sampling

    explicit EdgeData(bool use_gpu);

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);

    std::vector<std::tuple<int, int, int64_t>> get_edges();

    // Group management
    void update_timestamp_groups();  // Call after sorting
    void update_temporal_weights(double timescale_bound);

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const; // For backward walks
};

#endif //EDGEDATA_H
