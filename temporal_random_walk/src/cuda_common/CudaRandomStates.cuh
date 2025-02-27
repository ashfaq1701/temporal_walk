#ifndef CUDA_RANDOM_STATES_CUH
#define CUDA_RANDOM_STATES_CUH

#ifdef HAS_CUDA

#include <curand_kernel.h>
#include <cuda_runtime.h>
#include <ctime>

class CudaRandomStates {
public:
    // Initialize random states (should be called at startup)
    static void initialize();

    // Access the device random states
    static curandState* get_states();

    // Get device properties
    static unsigned int get_thread_count();
    static dim3 get_grid_dim();
    static dim3 get_block_dim();

    // Cleanup memory (called at program exit)
    static void cleanup();

private:
    static void init_device_properties();
    static void allocate_states();

    static curandState* d_states;
    static unsigned int num_blocks;
    static unsigned int num_threads;
    static unsigned int total_threads;
    static bool initialized;
};

#endif // HAS_CUDA
#endif // CUDA_RANDOM_STATES_CUH
