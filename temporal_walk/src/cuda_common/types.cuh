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

template <typename T, GPUUsageMode GPUUsage>
struct SelectVectorType {
#ifdef HAS_CUDA
    using type = typename std::conditional<
        GPUUsage != GPUUsageMode::ON_CPU,
        typename std::conditional<GPUUsage == GPUUsageMode::DATA_ON_GPU,
            thrust::device_vector<T>,
            thrust::host_vector<T>
        >::type,
        std::vector<T>
    >::type;
#else
    using type = std::vector<T>;
#endif

    static constexpr VectorStorageType get_vector_storage_type() {
#ifdef HAS_CUDA
        if (GPUUsage == GPUUsageMode::ON_CPU) {
            return VectorStorageType::STD_VECTOR;
        }
        return GPUUsage == GPUUsageMode::DATA_ON_GPU ?
            VectorStorageType::THRUST_DEVICE_VECTOR :
            VectorStorageType::THRUST_HOST_VECTOR;
#else
        return VectorStorageType::STD_VECTOR;
#endif
    }
};

#endif // TYPES_H
