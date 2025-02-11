#ifndef EDGEDATA_CUDA_H
#define EDGEDATA_CUDA_H

#include "../cpu/EdgeData.cuh"

template<bool UseGPU>
class EdgeDataCUDA : public EdgeData<UseGPU> {
#ifdef USE_CUDA

#endif
};

#endif //EDGEDATA_CUDA_H
