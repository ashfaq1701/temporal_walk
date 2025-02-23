#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>
#include <cmath>

#include "../../core/structs.h"
#include "../../common/types.cuh"

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

    /**
    * HOST METHODS
    */
    HOST void virtual reserve_host(size_t size);
    HOST void virtual clear_host();
    [[nodiscard]] virtual HOST size_t size_host() const;
    [[nodiscard]] virtual HOST bool empty_host() const;
    virtual HOST void resize_host(size_t new_size);

    virtual HOST void add_edges_host(int* src, int* tgt, int64_t* ts, size_t size);
    virtual HOST void push_back_host(int src, int tgt, int64_t ts);

    virtual std::vector<std::tuple<int, int, int64_t>> get_edges_host();

    // Group management
    virtual HOST void update_timestamp_groups_host();  // Call after sorting
    virtual HOST void update_temporal_weights_host(double timescale_bound);

    [[nodiscard]] virtual HOST std::pair<size_t, size_t> get_timestamp_group_range_host(size_t group_idx) const;
    [[nodiscard]] virtual HOST size_t get_timestamp_group_count_host() const;

    // Group lookup
    [[nodiscard]] virtual HOST size_t find_group_after_timestamp_host(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] virtual HOST size_t find_group_before_timestamp_host(int64_t timestamp) const; // For backward walks

    /**
    * DEVICE METHODS
    */
    virtual DEVICE void reserve_device(size_t size);
    virtual DEVICE void clear_device();
    [[nodiscard]] virtual DEVICE size_t size_device() const;
    [[nodiscard]] virtual DEVICE bool empty_device() const;
    virtual HOST void resize_device(size_t new_size);

    virtual DEVICE void add_edges_device(int* src, int* tgt, int64_t* ts, size_t size);
    virtual DEVICE void push_back_device(int src, int tgt, int64_t ts);

    virtual DEVICE std::vector<std::tuple<int, int, int64_t>> get_edges_device();

    // Group management
    virtual DEVICE void update_timestamp_groups_device();  // Call after sorting
    virtual DEVICE void update_temporal_weights_device(double timescale_bound);

    [[nodiscard]] virtual DEVICE std::pair<size_t, size_t> get_timestamp_group_range_device(size_t group_idx) const;
    [[nodiscard]] virtual DEVICE size_t get_timestamp_group_count_device() const;

    // Group lookup
    [[nodiscard]] virtual DEVICE size_t find_group_after_timestamp_device(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] virtual DEVICE size_t find_group_before_timestamp_device(int64_t timestamp) const; // For backward walks
};

#endif //EDGEDATA_H
