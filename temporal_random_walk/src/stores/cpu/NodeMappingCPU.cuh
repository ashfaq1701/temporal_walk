#ifndef NODEMAPPING_CPU_H
#define NODEMAPPING_CPU_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "../../data/enums.h"
#include "../interfaces/IEdgeData.cuh"
#include "../interfaces/INodeMapping.cuh"
#include "../../cuda_common/types.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCPU : public INodeMapping<GPUUsage> {
public:
   ~NodeMappingCPU() override = default;

   HOST void update_host(const IEdgeData<GPUUsage>& edges, size_t start_idx, size_t end_idx) override;
   [[nodiscard]] HOST int to_dense_host(int sparse_id) const override;
   [[nodiscard]] HOST int to_sparse_host(int dense_idx) const override;
   [[nodiscard]] HOST size_t size_host() const override;
   [[nodiscard]] HOST size_t active_size_host() const override;

   // Helper methods
   [[nodiscard]] HOST typename INodeMapping<GPUUsage>::IntVector get_active_node_ids_host() const override;
   HOST void clear_host() override;
   HOST void reserve_host(size_t size) override;
   HOST void mark_node_deleted_host(int sparse_id) override;
   [[nodiscard]] HOST bool has_node_host(int sparse_id) const override;
   [[nodiscard]] HOST typename INodeMapping<GPUUsage>::IntVector get_all_sparse_ids_host() const override;
};

#endif //NODEMAPPING_CPU_H
