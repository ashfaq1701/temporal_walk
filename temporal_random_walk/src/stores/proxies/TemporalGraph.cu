#include "TemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
TemporalGraph<GPUUsage>::TemporalGraph(
    bool directed,
    int64_t window,
    bool enable_weight_computation,
    double timescale_bound)
    : temporal_graph(new BaseType(directed, window, enable_weight_computation, timescale_bound)) {}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::sort_and_merge_edges(size_t start_idx)
{
    temporal_graph->sort_and_merge_edges(start_idx);
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::add_multiple_edges(const std::vector<Edge>& new_edges)
{
    typename ITemporalGraph<GPUUsage>::EdgeVector edge_vector;
    edge_vector.reserve(new_edges.size());
    for (const auto& edge : new_edges)
    {
        edge_vector.push_back(edge);
    }
    temporal_graph->add_multiple_edges(edge_vector);
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::update_temporal_weights()
{
    temporal_graph->update_temporal_weights();
}

template<GPUUsageMode GPUUsage>
void TemporalGraph<GPUUsage>::delete_old_edges()
{
    temporal_graph->delete_old_edges();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_less_than(int64_t timestamp) const
{
    return temporal_graph->count_timestamps_less_than(timestamp);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_timestamps_greater_than(int64_t timestamp) const
{
    return temporal_graph->count_timestamps_greater_than(timestamp);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_less_than(int node_id, int64_t timestamp) const
{
    return temporal_graph->count_node_timestamps_less_than(node_id, timestamp);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::count_node_timestamps_greater_than(int node_id, int64_t timestamp) const
{
    return temporal_graph->count_node_timestamps_greater_than(node_id, timestamp);
}

template<GPUUsageMode GPUUsage>
Edge TemporalGraph<GPUUsage>::get_edge_at(RandomPicker<GPUUsage>* picker, int64_t timestamp, bool forward) const
{
    return temporal_graph->get_edge_at_host(picker, timestamp, forward);
}

template<GPUUsageMode GPUUsage>
Edge TemporalGraph<GPUUsage>::get_node_edge_at(int node_id, RandomPicker<GPUUsage>* picker, int64_t timestamp, bool forward) const
{
    return temporal_graph->get_node_edge_at_host(node_id, picker, timestamp, forward);
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::get_total_edges() const
{
    return temporal_graph->get_total_edges();
}

template<GPUUsageMode GPUUsage>
size_t TemporalGraph<GPUUsage>::get_node_count() const
{
    return temporal_graph->get_node_count();
}

template<GPUUsageMode GPUUsage>
int64_t TemporalGraph<GPUUsage>::get_latest_timestamp()
{
    return temporal_graph->get_latest_timestamp();
}

template<GPUUsageMode GPUUsage>
std::vector<int> TemporalGraph<GPUUsage>::get_node_ids() const
{
    std::vector<int> result;
    auto node_ids = temporal_graph->get_node_ids();
    for (int i = 0; i < node_ids.size(); i++)
    {
        result.push_back(node_ids[i]);
    }
    return result;
}

template<GPUUsageMode GPUUsage>
std::vector<Edge> TemporalGraph<GPUUsage>::get_edges()
{
    std::vector<Edge> result;
    auto edges = temporal_graph->get_edges();
    for (int i = 0; i < edges.size(); i++)
    {
        result.push_back(edges[i]);
    }
    return result;
}

template class TemporalGraph<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalGraph<GPUUsageMode::ON_GPU>;
#endif
