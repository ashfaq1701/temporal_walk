#ifndef TEMPORALGRAPH_GPU_H
#define TEMPORALGRAPH_GPU_H

#include "../cpu/TemporalGraph.cuh"

template<bool UseGPU>
class TemporalGraphCUDA : public TemporalGraph<UseGPU> {
public:
    // Inherit constructors from base class
    using TemporalGraph<UseGPU>::TemporalGraph;

#ifdef USE_CUDA

#endif
};

#endif //TEMPORALGRAPH_GPU_H
