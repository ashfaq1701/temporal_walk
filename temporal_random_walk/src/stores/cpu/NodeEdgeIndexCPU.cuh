#ifndef NODEEDGEINDEX_CPU_H
#define NODEEDGEINDEX_CPU_H

#include <cstdint>
#include "../../data/enums.h"
#include "../interfaces/INodeEdgeIndex.cuh"
#include "../interfaces/INodeMapping.cuh"

template<GPUUsageMode GPUUsage>
class NodeEdgeIndexCPU : public INodeEdgeIndex<GPUUsage>
{
public:
    ~NodeEdgeIndexCPU() override = default;

    HOST void resize_node_offset_and_indices_vectors(size_t num_nodes, size_t num_edges, bool is_directed) override;

    HOST void compute_dense_vectors_host(
        const IEdgeData<GPUUsage>* edges,
        const INodeMapping<GPUUsage>* mapping,
        typename INodeEdgeIndex<GPUUsage>::IntVector& source_dense_ids,
        typename INodeEdgeIndex<GPUUsage>::IntVector& target_dense_ids) override;

    HOST void compute_node_offsets_and_indices_host(
        const IEdgeData<GPUUsage>* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& source_dense_ids,
        typename INodeEdgeIndex<GPUUsage>::IntVector& target_dense_ids,
        bool is_directed,
        size_t num_nodes,
        size_t num_edges) override;

    HOST void compute_node_edge_timestamp_group_offsets_host(
        const IEdgeData<GPUUsage>* edges,
        bool is_directed,
        size_t num_nodes,
        typename INodeEdgeIndex<GPUUsage>::SizeVector& outbound_node_timestamp_group_count,
        typename INodeEdgeIndex<GPUUsage>::SizeVector& inbound_node_timestamp_group_count) override;

    void resize_node_timestamp_group_indices(bool is_directed);

    HOST void compute_node_edge_timestamp_group_indices_host(
        const IEdgeData<GPUUsage>* edges,
        bool is_directed,
        size_t num_nodes) override;

    HOST void rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed) override;

    // Core access methods
    [[nodiscard]] HOST SizeRange get_edge_range_host(int dense_node_id, bool forward, bool is_directed) const override;
    [[nodiscard]] HOST SizeRange get_timestamp_group_range_host(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const override;
    [[nodiscard]] HOST size_t get_timestamp_group_count_host(int dense_node_id, bool forward, bool directed) const override;

    HOST void resize_weight_vectors() override;
    HOST void compute_temporal_weights_host(const IEdgeData<GPUUsage>* edges, double timescale_bound) override;
    HOST void update_temporal_weights(const IEdgeData<GPUUsage>* edges, double timescale_bound) override;

protected:
    HOST typename INodeEdgeIndex<GPUUsage>::SizeVector get_timestamp_offset_vector_host(bool forward, bool directed) const override;
};

#endif //NODEEDGEINDEX_CPU_H
