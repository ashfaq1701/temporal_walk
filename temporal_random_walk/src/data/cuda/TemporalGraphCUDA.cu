#include "TemporalGraphCUDA.cuh"

#ifdef HAS_CUDA

template class TemporalGraphCUDA<GPUUsageMode::ON_GPU>;
#endif
