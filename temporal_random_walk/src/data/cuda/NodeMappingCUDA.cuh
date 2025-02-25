#ifndef NODEMAPPINGCUDA_H
#define NODEMAPPINGCUDA_H

#include "../../structs/enums.h"
#include "../interfaces/INodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA : public INodeMapping<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //NODEMAPPINGCUDA_H
