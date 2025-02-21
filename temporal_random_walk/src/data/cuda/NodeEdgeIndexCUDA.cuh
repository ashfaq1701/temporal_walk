#ifndef NODEEDGEINDEXCUDA_H
#define NODEEDGEINDEXCUDA_H

#include "../../core/structs.h"
#include "../cpu/NodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public NodeEdgeIndex<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //NODEEDGEINDEXCUDA_H
