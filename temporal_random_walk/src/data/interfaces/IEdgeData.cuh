#ifndef I_EDGEDATA_H
#define I_EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>
#include <cmath>

#include "../../structs/structs.cuh"
#include "../../structs/enums.h"
#include "../../common/types.cuh"

template<GPUUsageMode GPUUsage>
class IEdgeData {
protected:
    using IntVector = typename SelectVectorType<int, GPUUsage>::type;
    using Int64Vector = typename SelectVectorType<int64_t, GPUUsage>::type;
    using SizeVector = typename SelectVectorType<size_t, GPUUsage>::type;
    using DoubleVector = typename SelectVectorType<double, GPUUsage>::type;
    using EdgeVector = typename SelectVectorType<Edge, GPUUsage>::type;

public:
    virtual ~IEdgeData() = default;

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
    virtual HOST void reserve_host(size_t size) {}
    virtual HOST void clear_host() {}
    [[nodiscard]] virtual HOST size_t size_host() const { return 0; }
    [[nodiscard]] virtual HOST bool empty_host() const { return true; }
    virtual HOST void resize_host(size_t new_size) {}

    virtual HOST void add_edges_host(int* src, int* tgt, int64_t* ts, size_t size) {}
    virtual HOST void push_back_host(int src, int tgt, int64_t ts) {}

    virtual EdgeVector get_edges_host() { return EdgeVector(); }

    // Group management
    virtual HOST void update_timestamp_groups_host() {}  // Call after sorting
    virtual HOST void update_temporal_weights_host(double timescale_bound) {};

    [[nodiscard]] virtual HOST SizeRange get_timestamp_group_range_host(size_t group_idx) const { return {}; };
    [[nodiscard]] virtual HOST size_t get_timestamp_group_count_host() const { return 0; }

    // Group lookup
    [[nodiscard]] virtual HOST size_t find_group_after_timestamp_host(int64_t timestamp) const { return 0; }  // For forward walks
    [[nodiscard]] virtual HOST size_t find_group_before_timestamp_host(int64_t timestamp) const { return 0; } // For backward walks

    /**
    * DEVICE METHODS
    */
    virtual DEVICE void reserve_device(size_t size) {}
    virtual DEVICE void clear_device() {}
    [[nodiscard]] virtual DEVICE size_t size_device() const { return 0; }
    [[nodiscard]] virtual DEVICE bool empty_device() const { return true; }
    virtual HOST void resize_device(size_t new_size) {}

    virtual DEVICE void add_edges_device(int* src, int* tgt, int64_t* ts, size_t size) {}
    virtual DEVICE void push_back_device(int src, int tgt, int64_t ts) {}

    virtual DEVICE EdgeVector get_edges_device() { return EdgeVector(); }

    // Group management
    virtual DEVICE void update_timestamp_groups_device() {}  // Call after sorting
    virtual DEVICE void update_temporal_weights_device(double timescale_bound) {}

    [[nodiscard]] virtual DEVICE SizeRange get_timestamp_group_range_device(size_t group_idx) const { return {}; }
    [[nodiscard]] virtual DEVICE size_t get_timestamp_group_count_device() const { return 0; }

    // Group lookup
    [[nodiscard]] virtual DEVICE size_t find_group_after_timestamp_device(int64_t timestamp) const { return 0; }  // For forward walks
    [[nodiscard]] virtual DEVICE size_t find_group_before_timestamp_device(int64_t timestamp) const { return 0; } // For backward walks
};

#endif //I_EDGEDATA_H
