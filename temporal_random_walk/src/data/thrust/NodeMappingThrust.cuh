#ifndef NODEMAPPING_THRUST_H
#define NODEMAPPING_THRUST_H

#include "../cpu/NodeMapping.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

#ifdef HAS_CUDA

__host__ __device__ int to_dense(const int* sparse_to_dense, int sparse_id, int size);
__host__ __device__ void mark_node_deleted(bool* is_deleted, int sparse_id, int size);

#endif

template<GPUUsageMode GPUUsage>
class NodeMappingThrust : public NodeMapping<GPUUsage>, public PolicyProvider<GPUUsage> {

public:
#ifdef HAS_CUDA

    void update(const EdgeData<GPUUsage>& edges, size_t start_idx, size_t end_idx) override;

    [[nodiscard]] size_t active_size() const override;
    [[nodiscard]] std::vector<int> get_active_node_ids() const override;

#endif
};

#endif //NODEMAPPING_THRUST_H
