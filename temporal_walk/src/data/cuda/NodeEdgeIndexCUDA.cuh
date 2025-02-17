#ifndef NODEEDGEINDEX_CUDA_H
#define NODEEDGEINDEX_CUDA_H

#include "../cpu/NodeEdgeIndex.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA final : public NodeEdgeIndex<GPUUsage>, public PolicyProvider<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEEDGEINDEX_CUDA_H
