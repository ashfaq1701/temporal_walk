#ifndef TEMPORAL_WALK_VECTOR_H
#define TEMPORAL_WALK_VECTOR_H

#include <vector>
#ifdef HAS_CUDA
#include <thrust/device_vector.h>
#endif

template<typename T>
class DualVector {
private:
    bool use_gpu;
    std::vector<T> h_vec;
#ifdef HAS_CUDA
    thrust::device_vector<T> d_vec;
#endif

public:
    explicit DualVector(bool use_gpu_param) : use_gpu(use_gpu_param) {
        #ifndef HAS_CUDA
        if (use_gpu) throw std::runtime_error("GPU support not compiled in");
        #endif
    }

    // Native accessors when specific behavior is needed
    std::vector<T>& host() {
        if (use_gpu) throw std::runtime_error("Using host vector in device mode");
        return h_vec;
    }

    const std::vector<T>& host() const {
        if (use_gpu) throw std::runtime_error("Using host vector in device mode");
        return h_vec;
    }

    #ifdef HAS_CUDA
    thrust::device_vector<T>& device() {
        if (!use_gpu) throw std::runtime_error("Using device vector in host mode");
        return d_vec;
    }

    const thrust::device_vector<T>& device() const {
        if (!use_gpu) throw std::runtime_error("Using device vector in host mode");
        return d_vec;
    }
    #endif

    // Common vector operations
    void push_back(const T& value) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            d_vec.push_back(value);
            #endif
        } else {
            h_vec.push_back(value);
        }
    }

    [[nodiscard]] size_t size() const {
        return use_gpu ?
            #ifdef HAS_CUDA
            d_vec.size()
            #else
            0
            #endif
            : h_vec.size();
    }

    [[nodiscard]] bool empty() const {
        return size() == 0;
    }

    void clear() {
        if (use_gpu) {
            #ifdef HAS_CUDA
            d_vec.clear();
            #endif
        } else {
            h_vec.clear();
        }
    }

    void reserve(size_t n) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            d_vec.reserve(n);
            #endif
        } else {
            h_vec.reserve(n);
        }
    }

    [[nodiscard]] bool is_gpu() const { return use_gpu; }

    // Element access operations
    T& operator[](size_t index) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return d_vec[index];
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return h_vec[index];
    }

    const T& operator[](size_t index) const {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return d_vec[index];
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return h_vec[index];
    }

    // Iterator support for algorithms
    auto begin() {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return device().begin();
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return host().begin();
    }

    auto end() {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return device().end();
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return host().end();
    }

    const auto begin() const {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return device().begin();
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return host().begin();
    }

    const auto end() const {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return device().end();
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return host().end();
    }

    // Moving data
    void move(size_t dest_pos, size_t src_pos, size_t count) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            thrust::copy(d_vec.begin() + src_pos, d_vec.begin() + src_pos + count,
                        d_vec.begin() + dest_pos);
            #endif
        } else {
            std::move(h_vec.begin() + src_pos, h_vec.begin() + src_pos + count,
                     h_vec.begin() + dest_pos);
        }
    }

    // Support for std::move
    DualVector(DualVector&& other) noexcept = default;
    DualVector& operator=(DualVector&& other) noexcept = default;

    // Back operations (used in push_back)
    T& back() {
        if (use_gpu) {
            #ifdef HAS_CUDA
            return d_vec.back();
            #endif
        }
        return h_vec.back();
    }

    // Resize support
    void resize(size_t new_size) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            d_vec.resize(new_size);
            #endif
        } else {
            h_vec.resize(new_size);
        }
    }

    void resize(size_t new_size, const T& default_value) {
        if (use_gpu) {
            #ifdef HAS_CUDA
            d_vec.resize(new_size, default_value);
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        } else {
            h_vec.resize(new_size, default_value);
        }
    }

    std::vector<T> to_vector() const {
        if (use_gpu) {
            #ifdef HAS_CUDA
            std::vector<T> result(d_vec.size());
            thrust::copy(d_vec.begin(), d_vec.end(), result.begin());
            return result;
            #else
            throw std::runtime_error("GPU support not compiled in");
            #endif
        }
        return h_vec;  // Already a vector, just return a copy
    }
};

#endif // TEMPORAL_WALK_VECTOR_H