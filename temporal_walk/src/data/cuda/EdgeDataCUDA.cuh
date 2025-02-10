#ifndef EDGEDATA_GPU_H
#define EDGEDATA_GPU_H

#include "../cpu/EdgeData.cuh"

template<bool UseGPU>
class EdgeDataCUDA : public EdgeData<UseGPU> {
#ifdef USE_CUDA

#endif
};

#endif //EDGEDATA_GPU_H
