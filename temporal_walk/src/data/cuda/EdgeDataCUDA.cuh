#ifndef EDGEDATA_CUDA_H
#define EDGEDATA_CUDA_H

#include "../cpu/EdgeData.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA final : public EdgeData<GPUUsage> {
public:
#ifdef HAS_CUDA

#endif
};

#endif //EDGEDATA_CUDA_H
