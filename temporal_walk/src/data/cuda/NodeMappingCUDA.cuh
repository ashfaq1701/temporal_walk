#ifndef NODEMAPPING_CUDA_H
#define NODEMAPPING_CUDA_H

#include "../cpu/NodeMapping.cuh"
#include "../../cuda_common/PolicyProvider.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCUDA final : public NodeMapping<GPUUsage>, public PolicyProvider<GPUUsage> {

public:
#ifdef HAS_CUDA

    void update(const EdgeData<GPUUsage>& edges, size_t start_idx, size_t end_idx) override;

    [[nodiscard]] size_t active_size() const override;
    [[nodiscard]] std::vector<int> get_active_node_ids() const override;

#endif
};

#endif //NODEMAPPING_CUDA_H
