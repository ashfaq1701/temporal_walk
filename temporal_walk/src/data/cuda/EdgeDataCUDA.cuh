#ifndef EDGEDATA_CUDA_H
#define EDGEDATA_CUDA_H

#include "../cpu/EdgeData.cuh"
#include "PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA final : public EdgeData<GPUUsage>, public PolicyProvider<GPUUsage> {
public:
#ifdef HAS_CUDA
    std::vector<std::tuple<int, int, int64_t>> get_edges() override;

    void update_timestamp_groups() override;

#endif
};

#endif //EDGEDATA_CUDA_H
