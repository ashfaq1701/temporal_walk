#include "NodeMappingThrust.cuh"

#ifdef HAS_CUDA

#include <thrust/extrema.h>
#include <thrust/count.h>

__host__ __device__ int to_dense(const int* sparse_to_dense, const int sparse_id, const int size) {
    return (sparse_id < size) ? sparse_to_dense[sparse_id] : -1;
}

__host__ __device__ void mark_node_deleted(bool* is_deleted, const int sparse_id, const int size) {
    if (sparse_id < size) {
        is_deleted[sparse_id] = true;
    }
}

template<GPUUsageMode GPUUsage>
void NodeMappingThrust<GPUUsage>::update(const EdgeData<GPUUsage>& edges, const size_t start_idx, const size_t end_idx) {
    auto max_source = thrust::max_element(
        PolicyProvider<GPUUsage>::get_policy(),
        edges.sources.begin() + start_idx,
        edges.sources.begin() + end_idx
    );

    auto max_target = thrust::max_element(
        PolicyProvider<GPUUsage>::get_policy(),
        edges.targets.begin() + start_idx,
        edges.targets.begin() + end_idx
    );

    auto max_source_element = *max_source;
    auto max_target_element = *max_target;

    int max_node_id = std::max(
        max_source != edges.sources.end() ? *max_source : 0,
        max_target != edges.targets.end() ? *max_target : 0
    );

    if (max_node_id < 0) {
        return;
    }

    // Extend vectors if needed
    if (max_node_id >= this->sparse_to_dense.size()) {
        this->sparse_to_dense.resize(max_node_id + 1, -1);
        this->is_deleted.resize(max_node_id + 1, true);
    }

    typename SelectVectorType<int, GPUUsage>::type new_node_flags(max_node_id + 1, 0);
    int* new_node_flags_ptr = thrust::raw_pointer_cast(new_node_flags.data());
    int* sparse_to_dense_ptr = thrust::raw_pointer_cast(this->sparse_to_dense.data());
    bool* is_deleted_ptr = thrust::raw_pointer_cast(this->is_deleted.data());
    const int* sources_ptr = thrust::raw_pointer_cast(edges.sources.data());
    const int* targets_ptr = thrust::raw_pointer_cast(edges.targets.data());


    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(start_idx),
        thrust::make_counting_iterator<size_t>(end_idx),
        [new_node_flags_ptr, sparse_to_dense_ptr, is_deleted_ptr, sources_ptr, targets_ptr]
        __host__ __device__ (const size_t idx) {
            const int source = sources_ptr[idx];
            const int target = targets_ptr[idx];

            is_deleted_ptr[source] = false;
            is_deleted_ptr[target] = false;

            if (sparse_to_dense_ptr[source] == -1) {
                new_node_flags_ptr[source] = 1;
            }
            if (sparse_to_dense_ptr[target] == -1) {
                new_node_flags_ptr[target] = 1;
            }
        }
    );

    // Calculate positions for new nodes
    typename SelectVectorType<int, GPUUsage>::type new_node_positions(new_node_flags.size());
    thrust::exclusive_scan(
        this->get_policy(),
        new_node_flags.begin(),
        new_node_flags.end(),
        new_node_positions.begin()
    );

    // Get total count and resize dense_to_sparse
    size_t old_size = this->dense_to_sparse.size();
    const size_t new_nodes = thrust::reduce(new_node_flags.begin(), new_node_flags.end());
    this->dense_to_sparse.resize(old_size + new_nodes);

    // Get raw pointers for final phase
    int* dense_to_sparse_ptr = thrust::raw_pointer_cast(this->dense_to_sparse.data());
    const int* new_node_positions_ptr = thrust::raw_pointer_cast(new_node_positions.data());

    // Assign dense indices in parallel
    thrust::for_each(
        this->get_policy(),
        thrust::make_counting_iterator<size_t>(0),
        thrust::make_counting_iterator<size_t>(new_node_flags.size()),
        [sparse_to_dense_ptr, dense_to_sparse_ptr, new_node_flags_ptr, new_node_positions_ptr, old_size]
        __host__ __device__ (const size_t idx) {
            if(new_node_flags_ptr[idx]) {
                const int new_dense_idx = static_cast<int>(old_size) + new_node_positions_ptr[idx];
                sparse_to_dense_ptr[idx] = new_dense_idx;
                dense_to_sparse_ptr[new_dense_idx] = static_cast<int>(idx);
            }
        }
    );
}

template<GPUUsageMode GPUUsage>
size_t NodeMappingThrust<GPUUsage>::active_size() const {
    return thrust::count(
        this->get_policy(),
        this->is_deleted.begin(),
        this->is_deleted.end(),
        false
    );
}

template<GPUUsageMode GPUUsage>
std::vector<int> NodeMappingThrust<GPUUsage>::get_active_node_ids() const {
    // Create temporary device vector for output
    typename SelectVectorType<int, GPUUsage>::type temp_output(this->dense_to_sparse.size());

    // Copy selected elements to temp device vector
    auto end = thrust::copy_if(
        this->get_policy(),
        this->dense_to_sparse.begin(),
        this->dense_to_sparse.end(),
        temp_output.begin(),
        [this] __host__ __device__ (const int sparse_id) {
            return !this->is_deleted[sparse_id];
        }
    );

    temp_output.resize(thrust::distance(temp_output.begin(), end));

    // Copy result back to host
    std::vector<int> result(temp_output.size());
    thrust::copy(temp_output.begin(), temp_output.end(), result.begin());

    return result;
}

template class NodeMappingThrust<GPUUsageMode::ON_GPU_USING_CUDA>;
template class NodeMappingThrust<GPUUsageMode::ON_HOST_USING_THRUST>;
#endif
