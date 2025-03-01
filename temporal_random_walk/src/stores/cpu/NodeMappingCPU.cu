#include "NodeMappingCPU.cuh"
#include <algorithm>

template<GPUUsageMode GPUUsage>
HOST void NodeMappingCPU<GPUUsage>::clear_host() {
    this->sparse_to_dense.clear();
    this->dense_to_sparse.clear();
    this->is_deleted.clear();
}

template<GPUUsageMode GPUUsage>
HOST void NodeMappingCPU<GPUUsage>::reserve_host(const size_t size) {
    this->sparse_to_dense.reserve(size);
    this->dense_to_sparse.reserve(size);
    this->is_deleted.reserve(size);
}

template<GPUUsageMode GPUUsage>
HOST void NodeMappingCPU<GPUUsage>::mark_node_deleted_host(const int sparse_id) {
    if (sparse_id < this->is_deleted.size()) {
        this->is_deleted[sparse_id] = true;
    }
}

template<GPUUsageMode GPUUsage>
HOST void NodeMappingCPU<GPUUsage>::update_host(const IEdgeData<GPUUsage>* edges, const size_t start_idx, const size_t end_idx) {
    // First pass: find max node ID
    int max_node_id = 0;
    for (size_t i = start_idx; i < end_idx; i++) {
        max_node_id = std::max({
            max_node_id,
            static_cast<int>(edges->sources[i]),
            static_cast<int>(edges->targets[i])
        });
    }

    // Extend sparse_to_dense if needed
    if (max_node_id >= this->sparse_to_dense.size()) {
        this->sparse_to_dense.resize(max_node_id + 1, -1);
        this->is_deleted.resize(max_node_id + 1, true);
    }

    typename INodeMapping<GPUUsage>::IntVector new_nodes;
    new_nodes.reserve((end_idx - start_idx) * 2);

    for (size_t i = start_idx; i < end_idx; i++) {
        new_nodes.push_back(edges->sources[i]);
        new_nodes.push_back(edges->targets[i]);
    }

    std::sort(new_nodes.begin(), new_nodes.end());

    // Map unmapped nodes
    for (int node : new_nodes) {
        if (node < 0) continue;

        this->is_deleted[node] = false;

        if (this->sparse_to_dense[node] == -1) {
            this->sparse_to_dense[node] = static_cast<int>(this->dense_to_sparse.size());
            this->dense_to_sparse.push_back(node);
        }
    }
}

template<GPUUsageMode GPUUsage>
HOST int NodeMappingCPU<GPUUsage>::to_dense_host(const int sparse_id) const {
    return sparse_id < this->sparse_to_dense.size() ? this->sparse_to_dense[sparse_id] : -1;
}

template<GPUUsageMode GPUUsage>
HOST int NodeMappingCPU<GPUUsage>::to_sparse_host(const int dense_idx) const {
    return dense_idx < this->dense_to_sparse.size() ? this->dense_to_sparse[dense_idx] : -1;
}

template<GPUUsageMode GPUUsage>
HOST size_t NodeMappingCPU<GPUUsage>::size_host() const {
    return this->dense_to_sparse.size();
}

template<GPUUsageMode GPUUsage>
HOST size_t NodeMappingCPU<GPUUsage>::active_size_host() const {
    return std::count(this->is_deleted.begin(), this->is_deleted.end(), false);
}

template<GPUUsageMode GPUUsage>
HOST typename INodeMapping<GPUUsage>::IntVector NodeMappingCPU<GPUUsage>::get_active_node_ids_host() const {
    typename INodeMapping<GPUUsage>::IntVector active_ids;
    active_ids.reserve(this->dense_to_sparse.size());
    for (int sparse_id : this->dense_to_sparse) {
        if (!this->is_deleted[sparse_id]) {
            active_ids.push_back(sparse_id);
        }
    }

    return active_ids;
}

template<GPUUsageMode GPUUsage>
HOST bool NodeMappingCPU<GPUUsage>::has_node_host(int sparse_id) const {
    return sparse_id < this->sparse_to_dense.size() && this->sparse_to_dense[sparse_id] != -1;
}

template<GPUUsageMode GPUUsage>
HOST typename INodeMapping<GPUUsage>::IntVector NodeMappingCPU<GPUUsage>::get_all_sparse_ids_host() const {
    return this->dense_to_sparse;
}

template class NodeMappingCPU<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class NodeMappingCPU<GPUUsageMode::ON_GPU>;
#endif
