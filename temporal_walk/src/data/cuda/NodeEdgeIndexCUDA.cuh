#ifndef NODEEDGEINDEX_CUDA_H
#define NODEEDGEINDEX_CUDA_H

#include "../cpu/NodeEdgeIndex.cuh"

template<bool UseGPU>
class NodeEdgeIndexCUDA : public NodeEdgeIndex<UseGPU> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEEDGEINDEX_CUDA_H
