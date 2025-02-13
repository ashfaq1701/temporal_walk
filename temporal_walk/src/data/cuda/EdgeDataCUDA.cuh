#ifndef EDGEDATA_CUDA_H
#define EDGEDATA_CUDA_H

#include "../cpu/EdgeData.cuh"
#include "../../cuda_common/config.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA final : public EdgeData<GPUUsage> {
public:
#ifdef HAS_CUDA

    std::vector<std::tuple<int, int, int64_t>> get_edges() override;

#endif
};

#endif //EDGEDATA_CUDA_H
