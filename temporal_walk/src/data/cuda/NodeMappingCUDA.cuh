#ifndef NODEMAPPING_CUDA_H
#define NODEMAPPING_CUDA_H

#include "../cpu/NodeMapping.cuh"

template<bool UseGPU>
class NodeMappingCUDA : public NodeMapping<UseGPU> {
#ifdef USE_CUDA

#endif
};

#endif //NODEMAPPING_CUDA_H
