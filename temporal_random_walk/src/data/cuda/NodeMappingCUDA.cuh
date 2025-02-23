#ifndef NODEMAPPINGCUDA_H
#define NODEMAPPINGCUDA_H

#include "../interfaces/NodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA : public NodeMapping<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //NODEMAPPINGCUDA_H
