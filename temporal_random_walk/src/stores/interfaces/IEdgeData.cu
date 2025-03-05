#include "IEdgeData.cuh"

#include <cstdint>

template<GPUUsageMode GPUUsage>
HOST void IEdgeData<GPUUsage>::reserve(size_t size) {
    this->sources.reserve(size);
    this->targets.reserve(size);
    this->timestamps.reserve(size);
    this->timestamp_group_offsets.reserve(size);  // Estimate for group count
    this->unique_timestamps.reserve(size);
}

template<GPUUsageMode GPUUsage>
HOST void IEdgeData<GPUUsage>::clear() {
    this->sources.clear();
    this->targets.clear();
    this->timestamps.clear();
    this->timestamp_group_offsets.clear();
    this->unique_timestamps.clear();
}

template<GPUUsageMode GPUUsage>
HOST size_t IEdgeData<GPUUsage>::size() const {
    return this->timestamps.size();
}

template<GPUUsageMode GPUUsage>
HOST bool IEdgeData<GPUUsage>::empty() const {
    return this->timestamps.empty();
}

template<GPUUsageMode GPUUsage>
void IEdgeData<GPUUsage>::resize(size_t new_size) {
    this->sources.resize(new_size);
    this->targets.resize(new_size);
    this->timestamps.resize(new_size);
}

template<GPUUsageMode GPUUsage>
HOST void IEdgeData<GPUUsage>::add_edges(int* src, int* tgt, int64_t* ts, size_t size) {
    this->sources.insert(this->sources.end(), src, src + size);
    this->targets.insert(this->targets.end(), tgt, tgt + size);
    this->timestamps.insert(this->timestamps.end(), ts, ts + size);
}

template<GPUUsageMode GPUUsage>
HOST void IEdgeData<GPUUsage>::push_back(int src, int tgt, int64_t ts) {
    this->sources.push_back(src);
    this->targets.push_back(tgt);
    this->timestamps.push_back(ts);
}

template<GPUUsageMode GPUUsage>
HOST typename IEdgeData<GPUUsage>::EdgeVector IEdgeData<GPUUsage>::get_edges() {
    EdgeVector accumulated_edges;
    accumulated_edges.reserve(this->sources.size());

    for (int i = 0; i < this->sources.size(); i++) {
        accumulated_edges.push_back(Edge(this->sources[i], this->targets[i], this->timestamps[i]));
    }

    return accumulated_edges;
}

template<GPUUsageMode GPUUsage>
HOST void IEdgeData<GPUUsage>::update_temporal_weights(const double timescale_bound) {
    if (this->timestamps.empty()) {
        this->forward_cumulative_weights_exponential.clear();
        this->backward_cumulative_weights_exponential.clear();
        return;
    }

    compute_temporal_weights(timescale_bound);
}

template<GPUUsageMode GPUUsage>
HOST SizeRange IEdgeData<GPUUsage>::get_timestamp_group_range(size_t group_idx) const {
    if (group_idx >= this->unique_timestamps.size()) {
        return SizeRange{0, 0};
    }
    return SizeRange{this->timestamp_group_offsets[group_idx], this->timestamp_group_offsets[group_idx + 1]};
}

template<GPUUsageMode GPUUsage>
HOST size_t IEdgeData<GPUUsage>::get_timestamp_group_count() const {
    return this->unique_timestamps.size();
}

template class IEdgeData<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class IEdgeData<GPUUsageMode::ON_GPU>;
#endif
