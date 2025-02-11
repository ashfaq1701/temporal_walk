#ifndef TEMPORAL_WALK_CUDA_MACROS_H
#define TEMPORAL_WALK_CUDA_MACROS_H

// CUDA function attributes
#ifdef HAS_CUDA
    #define DEVICE_FUNC __device__
    #define HOST_FUNC __host__
    #define HOST_DEVICE_FUNC __host__ __device__
#else
    #define DEVICE_FUNC
    #define HOST_FUNC
    #define HOST_DEVICE_FUNC
#endif

#endif // TEMPORAL_WALK_CUDA_MACROS_H