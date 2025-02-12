#ifndef NODEEDGEINDEX_CUDA_H
#define NODEEDGEINDEX_CUDA_H

#include "../cpu/NodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public NodeEdgeIndex<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEEDGEINDEX_CUDA_H
