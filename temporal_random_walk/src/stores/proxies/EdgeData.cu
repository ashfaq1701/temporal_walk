#include "EdgeData.cuh"


template class EdgeData<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class EdgeData<GPUUsageMode::ON_GPU>;
#endif

