#include "NodeMappingCUDA.cuh"

#ifdef HAS_CUDA

template class NodeMappingCUDA<GPUUsageMode::ON_GPU>;
#endif
