#include "NodeEdgeIndexCUDA.cuh"

#ifdef HAS_CUDA

template class NodeEdgeIndexCUDA<GPUUsageMode::ON_GPU>;
#endif
