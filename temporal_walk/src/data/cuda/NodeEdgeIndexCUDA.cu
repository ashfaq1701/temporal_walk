#include "NodeEdgeIndexCUDA.cuh"

#ifdef HAS_CUDA


template class NodeEdgeIndexCUDA<GPUUsageMode::DATA_ON_GPU>;
template class NodeEdgeIndexCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
