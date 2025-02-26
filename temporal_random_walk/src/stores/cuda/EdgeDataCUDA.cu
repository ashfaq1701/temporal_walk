#include "EdgeDataCUDA.cuh"

#ifdef HAS_CUDA

template class EdgeDataCUDA<GPUUsageMode::ON_GPU>;
#endif
