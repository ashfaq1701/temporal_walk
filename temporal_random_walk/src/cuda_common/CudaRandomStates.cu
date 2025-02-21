#include "CudaRandomStates.cuh"

#ifdef HAS_CUDA

// Static member initialization
unsigned int CUDARandomStates::num_blocks = 0;
unsigned int CUDARandomStates::num_threads = 0;
unsigned int CUDARandomStates::total_threads = 0;
bool CUDARandomStates::hardware_initialized = false;

// Separate kernel function
__global__ void init_random_states_kernel(curandState* states, const unsigned long seed) {
    const auto idx = threadIdx.x + blockDim.x * blockIdx.x;
    curand_init(seed, idx, 0, &states[idx]);
}

void CUDARandomStates::init_hardware_properties() {
    if (!hardware_initialized) {
        cudaDeviceProp prop{};
        cudaGetDeviceProperties(&prop, 0);

        num_threads = (prop.maxThreadsPerBlock / prop.warpSize) * prop.warpSize / 2;
        num_blocks = (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) / num_threads;
        total_threads = num_threads * num_blocks;

        hardware_initialized = true;
    }
}

CUDARandomStates::CUDARandomStates() : d_states(nullptr), initialized(false) {
    init_hardware_properties();
}

CUDARandomStates::~CUDARandomStates() {
    if (d_states) {
        cudaFree(d_states);
    }
}

void CUDARandomStates::initialize() {
    auto& inst = instance();
    if (!inst.initialized) {
        cudaMalloc(&inst.d_states, total_threads * sizeof(curandState));

        // Call the kernel function instead of using a lambda
        init_random_states_kernel<<<num_blocks, num_threads>>>(
            inst.d_states, static_cast<unsigned long>(time(nullptr))
        );
        cudaDeviceSynchronize();
        inst.initialized = true;
    }
}

curandState* CUDARandomStates::get_states() { return instance().d_states; }
unsigned int CUDARandomStates::get_thread_count() { return total_threads; }
dim3 CUDARandomStates::get_grid_dim() { return {num_blocks}; }
dim3 CUDARandomStates::get_block_dim() { return {num_threads}; }

#endif