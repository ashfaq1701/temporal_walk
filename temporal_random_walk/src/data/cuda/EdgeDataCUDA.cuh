#ifndef EDGEDATACUDA_H
#define EDGEDATACUDA_H

#include "../interfaces/EdgeData.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA : public EdgeData<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //EDGEDATACUDA_H
