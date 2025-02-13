#ifndef TEMPORAL_WALK_CUDA_MACROS_H
#define TEMPORAL_WALK_CUDA_MACROS_H

#include <thrust/execution_policy.h>

// CUDA function attributes
#ifdef HAS_CUDA

constexpr auto DEVICE_POLICY = thrust::device;
constexpr auto HOST_POLICY = thrust::host;

#endif

#endif // TEMPORAL_WALK_CUDA_MACROS_H