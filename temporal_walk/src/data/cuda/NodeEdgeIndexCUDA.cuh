#ifndef NODEEDGEINDEX_GPU_H
#define NODEEDGEINDEX_GPU_H

#include "../cpu/NodeEdgeIndex.cuh"

template<bool UseGPU>
class NodeEdgeIndexCUDA : public NodeEdgeIndex<UseGPU> {
#ifdef USE_CUDA

#endif
};

#endif //NODEEDGEINDEX_GPU_H
