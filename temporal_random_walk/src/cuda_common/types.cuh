#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <type_traits>
#include "../core/structs.h"

#ifdef HAS_CUDA
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
#endif

enum class VectorStorageType {
    STD_VECTOR,
    THRUST_DEVICE_VECTOR,
    THRUST_HOST_VECTOR
};

// Base template (default case: CPU only)
template <typename T, GPUUsageMode GPUUsage, typename Enable = void>
struct SelectVectorType;

// Specialization for CPU mode
template <typename T, GPUUsageMode GPUUsage>
struct SelectVectorType<T, GPUUsage, std::enable_if_t<GPUUsage == GPUUsageMode::ON_CPU>> {
    using type = std::vector<T>;

    static constexpr VectorStorageType get_vector_storage_type() {
        return VectorStorageType::STD_VECTOR;
    }
};

#ifdef HAS_CUDA
// Specialization for GPU mode (Device memory)
template <typename T, GPUUsageMode GPUUsage>
struct SelectVectorType<T, GPUUsage, std::enable_if_t<GPUUsage == GPUUsageMode::ON_GPU_USING_CUDA>> {
    using type = thrust::device_vector<T>;

    static constexpr VectorStorageType get_vector_storage_type() {
        return VectorStorageType::THRUST_DEVICE_VECTOR;
    }
};

// Specialization for GPU mode (Host memory)
template <typename T, GPUUsageMode GPUUsage>
struct SelectVectorType<T, GPUUsage, std::enable_if_t<GPUUsage == GPUUsageMode::ON_HOST_USING_THRUST>> {
    using type = thrust::host_vector<T>;

    static constexpr VectorStorageType get_vector_storage_type() {
        return VectorStorageType::THRUST_HOST_VECTOR;
    }
};
#endif

#endif // TYPES_H
