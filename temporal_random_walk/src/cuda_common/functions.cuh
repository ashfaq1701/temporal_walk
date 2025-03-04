#ifndef CUDA_FUNCTIONS_H
#define CUDA_FUNCTIONS_H

#ifdef HAS_CUDA

#include <curand_kernel.h>

__global__ inline void setup_curand_states(curandState* rand_states, const unsigned long seed) {
    const int tid = threadIdx.x + blockIdx.x * blockDim.x;
    curand_init(seed, tid, 0, &rand_states[tid]);
}

#endif

#endif // CUDA_FUNCTIONS_H