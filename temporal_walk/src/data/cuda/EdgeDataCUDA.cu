#include "EdgeDataCUDA.cuh"

#ifdef HAS_CUDA


template class EdgeDataCUDA<GPUUsageMode::ON_CPU>;
template class EdgeDataCUDA<GPUUsageMode::DATA_ON_GPU>;
template class EdgeDataCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
