#ifndef RAND_UTILS_H
#define RAND_UTILS_H

#include <random>
#include "../data/common_vector.cuh"
#include "../data/enums.h"
#ifdef HAS_CUDA
#include <curand_kernel.h>
#include "../cuda_common/CudaRandomStates.cuh"
#endif

thread_local static std::mt19937 thread_local_gen{std::random_device{}()};

template <typename T>
T generate_random_value_host(T start, T end) {
    std::uniform_real_distribution<T> dist(start, end);
    return dist(thread_local_gen);
}

inline int generate_random_int_host(const int start, const int end) {
    std::uniform_int_distribution<> dist(start, end);
    return dist(thread_local_gen);
}

inline int generate_random_number_bounded_by_host(const int max_bound) {
    return generate_random_int_host(0, max_bound - 1);
}

inline bool generate_random_boolean_host() {
    return generate_random_int_host(0, 1) == 1;
}

inline int pick_random_number_host(const int a, const int b) {
    return generate_random_boolean_host() ? a : b;
}

template <typename T>
void shuffle_vector_host(std::vector<T> vec) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec.begin(), vec.end(), rng);
}

#ifdef HAS_CUDA

template <typename T>
DEVICE T generate_random_value_device(T start, T end, curandState* state) {
    return start + (end - start) * curand_uniform(state);
}

__device__ int generate_random_int_device(int start, int end, curandState* state) {
    return start + (curand(state) % (end - start + 1));
}

__device__ int generate_random_number_bounded_by_device(int max_bound, curandState* state) {
    return curand(state) % max_bound;
}

__device__ bool generate_random_boolean_device(curandState* state) {
    return curand_uniform(state) > 0.5f;
}

template <typename T>
__global__ void shuffle_kernel(T* vec, int size, curandState* states) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx >= size) return;

    // Fisher-Yates shuffle (Parallelized)
    for (int i = size - 1; i > 0; --i) {
        int j = curand(&states[idx]) % (i + 1);
        if (idx == i || idx == j) {
            T temp = vec[i];
            vec[i] = vec[j];
            vec[j] = temp;
        }
    }
}

template <typename T>
void shuffle_vector(CommonVector<T, ON_GPU>& vec) {
    int size = vec.size();
    shuffle_kernel<<<CudaRandomStates::num_blocks, CudaRandomStates::num_threads>>>(vec.data, size, CudaRandomStates::get_states());
}

#endif

#endif // RAND_UTILS_H
