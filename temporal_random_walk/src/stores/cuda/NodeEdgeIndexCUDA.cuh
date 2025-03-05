#ifndef NODEEDGEINDEXCUDA_H
#define NODEEDGEINDEXCUDA_H

#include "../../data/enums.h"
#include "../interfaces/INodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public INodeEdgeIndex<GPUUsage> {
public:
    #ifdef HAS_CUDA

    HOST void compute_temporal_weights(
        const IEdgeData<GPUUsage>* edges,
        double timescale_bound,
        size_t num_nodes) override;

    #endif
};

#endif //NODEEDGEINDEXCUDA_H
