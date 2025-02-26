#include "NodeMapping.cuh"

template class NodeMapping<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeMapping<GPUUsageMode::ON_GPU>;
#endif
