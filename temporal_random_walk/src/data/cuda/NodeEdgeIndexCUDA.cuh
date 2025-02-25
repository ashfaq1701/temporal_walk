#ifndef NODEEDGEINDEXCUDA_H
#define NODEEDGEINDEXCUDA_H

#include "../../structs/enums.h"
#include "../interfaces/INodeEdgeIndex.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCUDA : public INodeEdgeIndex<GPUUsage> {
#ifdef HAS_CUDA

#endif
};

#endif //NODEEDGEINDEXCUDA_H
