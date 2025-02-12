#include "TemporalGraphCUDA.cuh"

#ifdef HAS_CUDA


template class TemporalGraphCUDA<GPUUsageMode::ON_CPU>;
template class TemporalGraphCUDA<GPUUsageMode::DATA_ON_GPU>;
template class TemporalGraphCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
