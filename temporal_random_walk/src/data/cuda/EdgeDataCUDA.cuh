#ifndef EDGEDATACUDA_H
#define EDGEDATACUDA_H

#include "../thrust/EdgeDataThrust.cuh"

template<GPUUsageMode GPUUsage>
class EdgeDataCUDA : public EdgeDataThrust<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //EDGEDATACUDA_H
