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

    HOST void clear_host() override;

    /**
     * START METHODS FOR REBUILD
     */
    HOST void populate_dense_ids_host(
        const IEdgeData<GPUUsage>* edges,
        const INodeMapping<GPUUsage>* mapping,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets);

    HOST void allocate_node_edge_offsets(size_t num_nodes, bool is_directed);

    HOST void compute_node_edge_offsets_host(
        const IEdgeData<GPUUsage>* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
        size_t num_nodes,
        bool is_directed);

    HOST void allocate_node_edge_indices(bool is_directed);

    HOST void compute_node_edge_indices_host(
        const IEdgeData<GPUUsage>* edges,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_sources,
        typename INodeEdgeIndex<GPUUsage>::IntVector& dense_targets,
        typename INodeEdgeIndex<GPUUsage>::SizeVector& outbound_running_index,
        typename INodeEdgeIndex<GPUUsage>::SizeVector& inbound_running_index,
        bool is_directed);

    HOST void compute_node_timestamp_offsets_host(
        const IEdgeData<GPUUsage>* edges,
        size_t num_nodes,
        bool is_directed);

    HOST void allocate_node_timestamp_indices(bool is_directed);

    HOST void compute_node_timestamp_indices_host(
        const IEdgeData<GPUUsage>* edges,
        size_t num_nodes,
        bool is_directed);
    /**
     * END METHODS FOR REBUILD
     */

    HOST void rebuild(const IEdgeData<GPUUsage>* edges, const INodeMapping<GPUUsage>* mapping, bool is_directed) override;

    // Core access methods
    [[nodiscard]] HOST SizeRange get_edge_range_host(int dense_node_id, bool forward, bool is_directed) const override;
    [[nodiscard]] HOST SizeRange get_timestamp_group_range_host(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const override;
    [[nodiscard]] HOST size_t get_timestamp_group_count_host(int dense_node_id, bool forward, bool directed) const override;

    HOST void compute_temporal_weights_host(
        const IEdgeData<GPUUsage>* edges,
        double timescale_bound,
        size_t num_nodes) override;

    HOST void update_temporal_weights(const IEdgeData<GPUUsage>* edges, double timescale_bound) override;

protected:
    HOST typename INodeEdgeIndex<GPUUsage>::SizeVector get_timestamp_offset_vector_host(bool forward, bool directed) const override;
};

#endif //NODEEDGEINDEX_CPU_H
