#ifndef NODEEDGEINDEX_CUDA_H
#define NODEEDGEINDEX_CUDA_H

#include "../cpu/NodeEdgeIndex.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA final : public NodeEdgeIndex<GPUUsage>, public PolicyProvider<GPUUsage> {
public:
#ifdef HAS_CUDA
    void rebuild(const EdgeData<GPUUsage>& edges, const NodeMapping<GPUUsage>& mapping, bool is_directed) override;

    void update_temporal_weights(const EdgeData<GPUUsage>& edges, double timescale_bound) override;

#endif
};

#endif //NODEEDGEINDEX_CUDA_H
