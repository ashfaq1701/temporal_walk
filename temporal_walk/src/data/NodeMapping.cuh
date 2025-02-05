#ifndef NODEMAPPING_H
#define NODEMAPPING_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "EdgeData.cuh"
#include "../cuda/dual_vector.cuh"

#ifdef HAS_CUDA
#include <thrust/copy.h>
#include <thrust/count.h>
#include <thrust/device_vector.h>
#include <thrust/execution_policy.h>
#define HOST_DEVICE __host__ __device__
#else
#define HOST_DEVICE
#endif

constexpr short ITEM_DELETED = 1;
constexpr short ITEM_NOT_DELETED = 0;

struct NodeMapping {
   bool use_gpu;

   DualVector<int> sparse_to_dense;    // Maps sparse ID to dense index
   DualVector<int> dense_to_sparse;    // Maps dense index back to sparse ID
   DualVector<short> is_deleted;        // Tracks deleted status of nodes

   explicit NodeMapping(bool use_gpu);

   void update(const EdgeData& edges, size_t start_idx, size_t end_idx);
   [[nodiscard]] int to_dense(int sparse_id) const;
   [[nodiscard]] int to_sparse(int dense_idx) const;
   [[nodiscard]] size_t size() const;
   [[nodiscard]] size_t active_size() const;

   // Helper methods
   [[nodiscard]] std::vector<int> get_active_node_ids() const;
   void clear();
   void reserve(size_t size);

   #ifdef HAS_CUDA
   static __device__ void device_mark_node_deleted(
       const int sparse_id,
       short* deleted_ptr,
       const size_t vector_size)
   {
      if (sparse_id < vector_size) {
         deleted_ptr[sparse_id] = ITEM_DELETED;
      }
   }
   #endif

   void host_mark_node_deleted(int sparse_id);               // Host-specific version

   [[nodiscard]] bool has_node(int sparse_id) const;
   [[nodiscard]] std::vector<int> get_all_sparse_ids() const;
};

#undef HOST_DEVICE

#endif //NODEMAPPING_H
