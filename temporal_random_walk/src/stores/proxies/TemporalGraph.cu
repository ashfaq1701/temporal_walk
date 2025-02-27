#include "TemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
TemporalGraph<GPUUsage>::TemporalGraph(
    bool directed,
    int64_t window,
    bool enable_weight_computation,
    double timescale_bound)
    : temporal_graph(BaseType(directed, window, enable_weight_computation, timescale_bound)) {}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::sort_and_merge_edges(size_t start_idx)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        temporal_graph.sort_and_merge_edges_host(start_idx);
    }
    else
    {
        temporal_graph.sort_and_merge_edges_device(start_idx);
    }
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::add_multiple_edges(const std::vector<Edge>& new_edges)
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        typename ITemporalGraph<GPUUsage>::EdgeVector edge_vector;
        edge_vector.allocate(new_edges.size());
        for (const auto& edge : new_edges)
        {
            edge_vector.push_back(edge);
        }
        temporal_graph.add_multiple_edges_host(edge_vector);
    }
    else
    {
        // Similar conversion for GPU mode
        typename ITemporalGraph<GPUUsage>::EdgeVector edge_vector;
        edge_vector.allocate(new_edges.size());
        for (const auto& edge : new_edges)
        {
            edge_vector.push_back(edge);
        }
        temporal_graph.add_multiple_edges_device(edge_vector);
    }
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::update_temporal_weights()
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        temporal_graph.update_temporal_weights_host();
    }
    else
    {
        temporal_graph.update_temporal_weights_device();
    }
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::delete_old_edges()
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        temporal_graph.delete_old_edges_host();
    }
    else
    {
        temporal_graph.delete_old_edges_device();
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.count_timestamps_less_than_host(timestamp);
    }
    else
    {
        return temporal_graph.count_timestamps_less_than_device(timestamp);
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.count_timestamps_greater_than_host(timestamp);
    }
    else
    {
        return temporal_graph.count_timestamps_greater_than_device(timestamp);
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.count_node_timestamps_less_than_host(node_id, timestamp);
    }
    else
    {
        return temporal_graph.count_node_timestamps_less_than_device(node_id, timestamp);
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.count_node_timestamps_greater_than_host(node_id, timestamp);
    }
    else
    {
        return temporal_graph.count_node_timestamps_greater_than_device(node_id, timestamp);
    }
}

template<GPUUsageMode GPUUsage>
Edge TemporalGraph<GPUUsage>::get_edge_at(RandomPicker* picker, int64_t timestamp, bool forward) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.get_edge_at_host(picker, timestamp, forward);
    }
    else
    {
        return temporal_graph.get_edge_at_device(picker, timestamp, forward);
    }
}

template<GPUUsageMode GPUUsage>
Edge TemporalGraph<GPUUsage>::get_node_edge_at(int node_id, RandomPicker* picker, int64_t timestamp, bool forward) const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.get_node_edge_at_host(node_id, picker, timestamp, forward);
    }
    else
    {
        return temporal_graph.get_node_edge_at_device(node_id, picker, timestamp, forward);
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::get_total_edges() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.get_total_edges_host();
    }
    else
    {
        return temporal_graph.get_total_edges_device();
    }
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::get_node_count() const
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.get_node_count_host();
    }
    else
    {
        return temporal_graph.get_node_count_device();
    }
}

template<GPUUsageMode GPUUsage>
int64_t TemporalGraph<GPUUsage>::get_latest_timestamp()
{
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        return temporal_graph.get_latest_timestamp_host();
    }
    else
    {
        return temporal_graph.get_latest_timestamp_device();
    }
}

template<GPUUsageMode GPUUsage>
std::vector<int> TemporalGraph<GPUUsage>::get_node_ids() const
{
    std::vector<int> result;
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto node_ids = temporal_graph.get_node_ids_host();
        for (int i = 0; i < node_ids.size(); i++)
        {
            result.push_back(node_ids[i]);
        }
    }
    else
    {
        auto node_ids = temporal_graph.get_node_ids_device();
        for (int i = 0; i < node_ids.size(); i++)
        {
            result.push_back(node_ids[i]);
        }
    }
    return result;
}

template<GPUUsageMode GPUUsage>
std::vector<Edge> TemporalGraph<GPUUsage>::get_edges()
{
    std::vector<Edge> result;
    if (GPUUsage == GPUUsageMode::ON_CPU)
    {
        auto edges = temporal_graph.get_edges_host();
        for (int i = 0; i < edges.size(); i++)
        {
            result.push_back(edges[i]);
        }
    }
    else
    {
        auto edges = temporal_graph.get_edges_device();
        for (int i = 0; i < edges.size(); i++)
        {
            result.push_back(edges[i]);
        }
    }
    return result;
}

template class TemporalGraph<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalGraph<GPUUsageMode::ON_GPU>;
#endif
