#ifndef NODEMAPPING_GPU_H
#define NODEMAPPING_GPU_H

#include "../cpu/NodeMapping.cuh"

template<bool UseGPU>
class NodeMappingCUDA : public NodeMapping<UseGPU> {
#ifdef USE_CUDA

#endif
};

#endif //NODEMAPPING_GPU_H
