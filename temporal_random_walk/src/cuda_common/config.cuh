#ifndef TEMPORAL_RANDOM_WALK_CUDA_MACROS_H
#define TEMPORAL_RANDOM_WALK_CUDA_MACROS_H

// CUDA function attributes
#ifdef HAS_CUDA

#include <thrust/execution_policy.h>

constexpr auto DEVICE_POLICY = thrust::device;
constexpr auto HOST_POLICY = thrust::host;

#endif

#endif // TEMPORAL_RANDOM_WALK_CUDA_MACROS_H