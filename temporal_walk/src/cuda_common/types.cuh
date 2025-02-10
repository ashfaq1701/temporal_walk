#ifndef TYPES_H
#define TYPES_H

#include <vector>
#include <variant>
#include <type_traits>
#ifdef USE_CUDA
    #include <thrust/device_vector.h>
    #include <thrust/host_vector.h>
#endif

#include <iostream>
#include <vector>
#include <stdexcept>
#include <cmath>
#include <random>

template<typename T>
struct VectorTypes {
#ifdef USE_CUDA
    using Device = thrust::device_vector<T>;
    using Host = thrust::host_vector<T>;
    using Vector = std::variant<Host, Device>;

    static Vector select(bool use_gpu) {
        return use_gpu ? Vector(Device()) : Vector(Host());
    }

    static Vector select(bool use_gpu, size_t size) {
        return use_gpu ? Vector(Device(size)) : Vector(Host(size));
    }
#else
    using Host = std::vector<T>;
    using Vector = std::variant<Host>;

    static Vector select(bool use_gpu) {
        return Vector(Host());
    }

    static Vector select(bool use_gpu, size_t size) {
        return Vector(Host(size));
    }
#endif
};

template <typename NewType, typename VecType>
struct RebindVectorT {
    using CleanVecType = std::remove_reference_t<VecType>;
    using ValueType = typename CleanVecType::value_type;

    #ifdef USE_CUDA
    using Type = std::conditional_t<
        std::is_same_v<CleanVecType, std::vector<ValueType>>,
        std::vector<NewType>,
        std::conditional_t<
            std::is_same_v<CleanVecType, thrust::host_vector<ValueType>>,
            thrust::host_vector<NewType>,
            std::conditional_t<
                std::is_same_v<CleanVecType, thrust::device_vector<ValueType>>,
                thrust::device_vector<NewType>,
                void  // Unsupported type
            >
        >
    >;
    #else
    using Type = std::conditional_t<
        std::is_same_v<CleanVecType, std::vector<ValueType>>,
        std::vector<NewType>,
        void  // Unsupported type when CUDA is disabled
    >;
    #endif
};

#endif // TYPES_H
