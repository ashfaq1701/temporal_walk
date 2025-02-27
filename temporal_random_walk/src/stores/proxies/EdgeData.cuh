#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <functional>
#include "../cpu/EdgeDataCPU.cuh"
#include "../cuda/EdgeDataCUDA.cuh"

#include "../../data/enums.h"

template<GPUUsageMode GPUUsage>
class EdgeData : protected std::conditional_t<
    (GPUUsage == GPUUsageMode::ON_CPU), EdgeDataCPU<GPUUsage>, EdgeDataCUDA<GPUUsage>> {

protected:
    using BaseType = std::conditional_t<
        (GPUUsage == ON_CPU), EdgeDataCPU<GPUUsage>, EdgeDataCUDA<GPUUsage>>;

    BaseType edge_data;

public:
    using BaseType::sources;
    using BaseType::targets;
    using BaseType::timestamps;

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    virtual void resize(size_t new_size);

    void add_edges(int* src, int* tgt, int64_t* ts, size_t size);
    void push_back(int src, int tgt, int64_t ts);

    std::vector<Edge> get_edges();

    // Group management
    void update_timestamp_groups();
    void update_temporal_weights(double timescale_bound);

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx);
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;
    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const;
};

#endif //EDGEDATA_H
