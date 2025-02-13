#include "EdgeDataCUDA.cuh"

#ifdef HAS_CUDA

#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

template<GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> EdgeDataCUDA<GPUUsage>::get_edges() {
    const size_t n = this->sources.size();
    std::vector<std::tuple<int, int, int64_t>> result(n);

    if constexpr (GPUUsage == GPUUsageMode::DATA_ON_GPU) {
        thrust::device_vector<thrust::tuple<int, int, int64_t>> d_tuples(n);

        thrust::copy(
            DEVICE_POLICY,
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
            HOST_POLICY,
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

template class EdgeDataCUDA<GPUUsageMode::DATA_ON_GPU>;
template class EdgeDataCUDA<GPUUsageMode::DATA_ON_HOST>;
#endif
