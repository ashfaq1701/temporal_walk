#ifndef NODEMAPPING_CPU_H
#define NODEMAPPING_CPU_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "../../structs/enums.h"
#include "../interfaces/EdgeData.cuh"
#include "../interfaces/NodeMapping.cuh"
#include "../../common/types.cuh"

template<GPUUsageMode GPUUsage>
class NodeMappingCPU {

protected:
   using IntVector = typename SelectVectorType<int, GPUUsage>::type;
   using BoolVector = typename SelectVectorType<bool, GPUUsage>::type;

public:
   virtual ~NodeMappingCPU() = default;

   IntVector sparse_to_dense{};    // Maps sparse ID to dense index
   IntVector dense_to_sparse{};    // Maps dense index back to sparse ID

   BoolVector is_deleted{};        // Tracks deleted status of nodes

   virtual void update(const EdgeData<GPUUsage>& edges, size_t start_idx, size_t end_idx);
   [[nodiscard]] int to_dense(int sparse_id) const;
   [[nodiscard]] int to_sparse(int dense_idx) const;
   [[nodiscard]] size_t size() const;
   [[nodiscard]] virtual size_t active_size() const;

   // Helper methods
   [[nodiscard]] virtual std::vector<int> get_active_node_ids() const;
   void clear();
   void reserve(size_t size);
   void mark_node_deleted(int sparse_id);
   [[nodiscard]] bool has_node(int sparse_id) const;
   [[nodiscard]] IntVector get_all_sparse_ids() const;
};

#endif //NODEMAPPING_CPU_H
