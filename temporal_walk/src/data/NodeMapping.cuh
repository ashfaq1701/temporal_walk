#ifndef NODEMAPPING_H
#define NODEMAPPING_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "EdgeData.cuh"

template<bool UseGPU>
struct NodeMapping {

   using IntVector = typename SelectVectorType<int, UseGPU>::type;
   using BoolVector = typename SelectVectorType<bool, UseGPU>::type;

   IntVector sparse_to_dense;    // Maps sparse ID to dense index
   IntVector dense_to_sparse;    // Maps dense index back to sparse ID
   BoolVector is_deleted;        // Tracks deleted status of nodes

   void update(const EdgeData<UseGPU>& edges, size_t start_idx, size_t end_idx);
   [[nodiscard]] int to_dense(int sparse_id) const;
   [[nodiscard]] int to_sparse(int dense_idx) const;
   [[nodiscard]] size_t size() const;
   [[nodiscard]] size_t active_size() const;

   // Helper methods
   [[nodiscard]] std::vector<int> get_active_node_ids() const;
   void clear();
   void reserve(size_t size);
   void mark_node_deleted(int sparse_id);
   [[nodiscard]] bool has_node(int sparse_id) const;
   [[nodiscard]] std::vector<int> get_all_sparse_ids() const;
};

#endif //NODEMAPPING_H
