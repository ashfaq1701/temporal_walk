#include "EdgeDataCUDA.cuh"

#ifdef HAS_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>
#include <thrust/execution_policy.h>

template<GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> EdgeDataCUDA<GPUUsage>::get_edges() {
    const size_t n = this->sources.size();
    std::vector<std::tuple<int, int, int64_t>> result(n);

    if constexpr (GPUUsage == GPUUsageMode::DATA_ON_GPU) {
        thrust::device_vector<thrust::tuple<int, int, int64_t>> d_tuples(n);

        thrust::copy(
            this->get_policy(),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    this->sources.begin(),
                    this->targets.begin(),
                    this->timestamps.begin()
                )),
            thrust::make_zip_iterator(
                thrust::make_tuple(
                    this->sources.end(),
                    this->targets.end(),
                    this->timestamps.end()
                )),
                d_tuples.begin()
        );

        thrust::copy(
            DEVICE_POLICY,
            d_tuples.begin(),
            d_tuples.end(),
            reinterpret_cast<thrust::tuple<int, int, int64_t>*>(result.data())
        );
    }
    else if constexpr (GPUUsage == GPUUsageMode::DATA_ON_HOST) {
        thrust::copy(
            this->get_policy(),
            thrust::make_zip_iterator(thrust::make_tuple(
                this->sources.begin(),
                this->targets.begin(),
                this->timestamps.begin()
            )),
            thrust::make_zip_iterator(thrust::make_tuple(
                this->sources.end(),
                this->targets.end(),
                this->timestamps.end()
            )),
            reinterpret_cast<thrust::tuple<int, int, int64_t>*>(result.data())
        );
    }

    return result;
}

template <GPUUsageMode GPUUsage>
void EdgeDataCUDA<GPUUsage>::update_timestamp_groups() {
    if (this->timestamps.empty()) {
        this->timestamp_group_offsets.clear();
        this->unique_timestamps.clear();
        return;
    }

    const size_t n = this->timestamps.size();

    // Create a temporary vector for flags where timestamps change
    typename SelectVectorType<int, GPUUsage>::type flags(n);

    thrust::transform(
        this->get_policy(),
        this->timestamps.begin() + 1,
        this->timestamps.end(),
        this->timestamps.begin(),
        flags.begin() + 1,
        [] __host__ __device__ (const int64_t curr, const int64_t prev) { return curr != prev ? 1 : 0; });

    // First element is always a group start
    thrust::fill_n(flags.begin(), 1, 1);

    // Count total groups (sum of flags)
    size_t num_groups = thrust::reduce(flags.begin(), flags.end());

    // Resize output vectors
    this->timestamp_group_offsets.resize(num_groups + 1);  // +1 for end offset
    this->unique_timestamps.resize(num_groups);

    // Find positions of group boundaries
    thrust::copy_if(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(n),
        flags.begin(),
        this->timestamp_group_offsets.begin(),
        [] __host__ __device__ (const int flag) { return flag == 1; });

    // Add final offset
    thrust::fill_n(this->timestamp_group_offsets.begin() + num_groups, 1, n);

    // Get unique timestamps at group boundaries
    thrust::copy_if(
        this->get_policy(),
        this->timestamps.begin(),
        this->timestamps.end(),
        flags.begin(),
        this->unique_timestamps.begin(),
        [] __host__ __device__ (const int flag) { return flag == 1; });
}

template class EdgeDataCUDA<GPUUsageMode::DATA_ON_GPU>;
template class EdgeDataCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
