#ifndef EDGEDATA_CPU_H
#define EDGEDATA_CPU_H

#include <vector>
#include <cstdint>

#include "../interfaces/IEdgeData.cuh"
#include "../../data/enums.h"
#include "../../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCPU : public IEdgeData<GPUUsage> {

public:
    HOST void reserve_host(size_t size) override;
    HOST void clear_host() override;
    [[nodiscard]] HOST size_t size_host() const override;
    [[nodiscard]] HOST bool empty_host() const override;
    HOST void resize_host(size_t new_size) override;

    HOST void add_edges_host(int* src, int* tgt, int64_t* ts, size_t size) override;
    HOST void push_back_host(int src, int tgt, int64_t ts) override;

    HOST typename IEdgeData<GPUUsage>::EdgeVector get_edges_host() override;

    // Group management
    HOST void update_timestamp_groups_host() override;  // Call after sorting
    HOST void update_temporal_weights_host(double timescale_bound) override;

    [[nodiscard]] HOST SizeRange get_timestamp_group_range_host(size_t group_idx) const override;
    [[nodiscard]] HOST size_t get_timestamp_group_count_host() const override;

    // Group lookup
    [[nodiscard]] HOST size_t find_group_after_timestamp_host(int64_t timestamp) const override;  // For forward walks
    [[nodiscard]] HOST size_t find_group_before_timestamp_host(int64_t timestamp) const override; // For backward walks
};

#endif //EDGEDATA_CPU_H
