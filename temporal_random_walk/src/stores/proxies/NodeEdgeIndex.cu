#include "NodeEdgeIndex.cuh"


template class NodeEdgeIndex<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeEdgeIndex<GPUUsageMode::ON_GPU>;
#endif
