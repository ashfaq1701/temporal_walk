#ifndef EDGEDATACUDA_H
#define EDGEDATACUDA_H

#include "../interfaces/IEdgeData.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA : public IEdgeData<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //EDGEDATACUDA_H
