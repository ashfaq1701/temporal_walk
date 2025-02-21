#ifndef NODEMAPPINGCUDA_H
#define NODEMAPPINGCUDA_H

#include "../thrust/NodeMappingThrust.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA : public NodeMappingThrust<GPUUsage> {
#ifdef HAS_CUDA

#endif
};



#endif //NODEMAPPINGCUDA_H
