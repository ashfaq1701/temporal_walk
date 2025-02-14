#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>
#include <cmath>

#include "../../core/structs.h"
#include "../../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class EdgeData {
protected:
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64Vector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using DoubleVector = typename SelectVectorType<double, GPUUsage>::type;

public:
    virtual ~EdgeData() = default;

    // Core edge data
    IntVector sources{};
    IntVector targets{};
    Int64Vector timestamps{};

    // Timestamp grouping
    SizeVector timestamp_group_offsets{};     // Start of each timestamp group
    Int64Vector unique_timestamps{};          // Corresponding unique timestamps

    DoubleVector forward_cumulative_weights_exponential{};  // For forward temporal sampling
    DoubleVector backward_cumulative_weights_exponential{}; // For backward temporal sampling

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);

    virtual std::vector<std::tuple<int, int, int64_t>> get_edges();

    // Group management
    virtual void update_timestamp_groups();  // Call after sorting
    virtual void update_temporal_weights(double timescale_bound);

    [[nodiscard]] virtual std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] virtual size_t find_group_after_timestamp(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] virtual size_t find_group_before_timestamp(int64_t timestamp) const; // For backward walks
};

#endif //EDGEDATA_H
