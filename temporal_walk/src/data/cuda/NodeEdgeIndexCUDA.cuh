#ifndef NODEEDGEINDEX_CUDA_H
#define NODEEDGEINDEX_CUDA_H

#include "../cpu/NodeEdgeIndex.cuh"
#include "../../cuda_common/config.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA final : public NodeEdgeIndex<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEEDGEINDEX_CUDA_H
