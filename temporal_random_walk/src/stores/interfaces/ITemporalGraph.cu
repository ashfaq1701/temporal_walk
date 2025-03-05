#include "ITemporalGraph.cuh"

template<GPUUsageMode GPUUsage>
HOST void ITemporalGraph<GPUUsage>::update_temporal_weights() {
    this->edges->update_temporal_weights(this->timescale_bound);
    this->node_index->update_temporal_weights(this->edges, this->timescale_bound);
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalGraph<GPUUsage>::IntVector ITemporalGraph<GPUUsage>::get_node_ids() const {
    return this->node_mapping->get_active_node_ids();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalGraph<GPUUsage>::EdgeVector ITemporalGraph<GPUUsage>::get_edges() {
    return this->edges->get_edges();
}

template<GPUUsageMode GPUUsage>
HOST size_t ITemporalGraph<GPUUsage>::get_total_edges() const {
    return this->edges->size();
}

template<GPUUsageMode GPUUsage>
HOST size_t ITemporalGraph<GPUUsage>::get_node_count() const {
    return this->node_mapping->active_size();
}

template<GPUUsageMode GPUUsage>
HOST int64_t ITemporalGraph<GPUUsage>::get_latest_timestamp() {
    return this->latest_timestamp;
}
