#include "NodeMappingCUDA.cuh"

#ifdef HAS_CUDA


template class NodeMappingCUDA<GPUUsageMode::ON_CPU>;
template class NodeMappingCUDA<GPUUsageMode::DATA_ON_GPU>;
template class NodeMappingCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
