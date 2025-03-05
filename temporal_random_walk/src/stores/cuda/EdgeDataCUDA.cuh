#ifndef EDGEDATACUDA_H
#define EDGEDATACUDA_H

#include "../interfaces/IEdgeData.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA : public IEdgeData<GPUUsage> {
public:
    #ifdef HAS_CUDA
    // Group management
    HOST void update_timestamp_groups() override;  // Call after sorting

    HOST void compute_temporal_weights(double timescale_bound) override;

    // Group lookup
    [[nodiscard]] HOST size_t find_group_after_timestamp(int64_t timestamp) const override;  // For forward walks
    [[nodiscard]] HOST size_t find_group_before_timestamp(int64_t timestamp) const override; // For backward walks
    #endif
};



#endif //EDGEDATACUDA_H
