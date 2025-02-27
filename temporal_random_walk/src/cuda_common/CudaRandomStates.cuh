// CUDARandomStates.cuh
#ifndef CUDARANDOMSTATES_CUH
#define CUDARANDOMSTATES_CUH

#ifdef HAS_CUDA

#include <curand_kernel.h>

class CUDARandomStates {
private:
    // Dynamic state
    static curandState* d_states;
    static bool initialized;

    // Static hardware properties (initialized once)
    static unsigned int num_blocks;
    static unsigned int num_threads;
    static unsigned int total_threads;
    static bool hardware_initialized;

    static CUDARandomStates& instance() {
        static CUDARandomStates instance;
        return instance;
    }

    CUDARandomStates();
    ~CUDARandomStates();

    static void init_hardware_properties();

public:
    CUDARandomStates(const CUDARandomStates&) = delete;
    CUDARandomStates& operator=(const CUDARandomStates&) = delete;

    static void initialize();
    static curandState* get_states();
    static unsigned int get_thread_count();
    static dim3 get_grid_dim();
    static dim3 get_block_dim();
};

#endif

#endif //CUDARANDOMSTATES_CUH
