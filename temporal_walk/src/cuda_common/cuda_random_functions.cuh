#ifndef CUDA_RANDOM_FUNCTIONS_CUH
#define CUDA_RANDOM_FUNCTIONS_CUH

#ifdef HAS_CUDA

#include <thrust/execution_policy.h>
#include <thrust/random.h>
#include <thrust/random/linear_congruential_engine.h>
#include <thrust/device_vector.h>
#include <thrust/shuffle.h>
#include <type_traits>

template <typename T>
__device__ inline T generate_random_value_cuda(T start, T end) {
    thrust::random::default_random_engine rng;
    rng.seed(threadIdx.x + blockDim.x * blockIdx.x + clock64());

    thrust::random::uniform_real_distribution<T> dist(start, end);
    return dist(rng);
}

__device__ inline int generate_random_int_cuda(const int start, const int end) {
    thrust::random::default_random_engine rng;
    rng.seed(threadIdx.x + blockDim.x * blockIdx.x + clock64());

    thrust::random::uniform_int_distribution<int> dist(start, end);
    return dist(rng);
}

__device__ inline int generate_random_number_bounded_by_cuda(const int max_bound) {
    return generate_random_int_cuda(0, max_bound);
}

__device__ inline bool generate_random_boolean_cuda() {
    return generate_random_int_cuda(0, 1) == 1;
}

__device__ inline int pick_random_number_cuda(const int a, const int b) {
    return generate_random_boolean_cuda() ? a : b;
}

__device__ inline int pick_other_number_cuda(const int first, const int second, const int picked_number) {
    return (picked_number == first) ? second : first;
}

// Note: shuffle is host-only as it uses thrust::device_vector
template <typename T>
__device__ inline void shuffle_vector_cuda(thrust::device_vector<T>& vec) {
    thrust::random::default_random_engine rng;
    rng.seed(threadIdx.x + blockDim.x * blockIdx.x + clock64());

    thrust::shuffle(thrust::device, vec.begin(), vec.end(), rng);
}

#endif // HAS_CUDA
#endif // CUDA_RANDOM_FUNCTIONS_CUH