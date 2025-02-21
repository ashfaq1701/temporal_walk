#ifndef NODEEDGEINDEXCUDA_H
#define NODEEDGEINDEXCUDA_H

#include "../thrust/NodeEdgeIndexThrust.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public NodeEdgeIndexThrust<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //NODEEDGEINDEXCUDA_H
