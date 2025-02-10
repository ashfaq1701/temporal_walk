#include "NodeEdgeIndex.cuh"

#include <iostream>

NodeEdgeIndex::NodeEdgeIndex(bool use_gpu): use_gpu(use_gpu) {
    outbound_offsets = VectorTypes<size_t>::select(use_gpu);
    outbound_indices = VectorTypes<size_t>::select(use_gpu);
    outbound_timestamp_group_offsets = VectorTypes<size_t>::select(use_gpu);
    outbound_timestamp_group_indices = VectorTypes<size_t>::select(use_gpu);

    inbound_offsets = VectorTypes<size_t>::select(use_gpu);
    inbound_indices = VectorTypes<size_t>::select(use_gpu);
    inbound_timestamp_group_offsets = VectorTypes<size_t>::select(use_gpu);
    inbound_timestamp_group_indices = VectorTypes<size_t>::select(use_gpu);

    outbound_forward_cumulative_weights_exponential = VectorTypes<double>::select(use_gpu);
    outbound_backward_cumulative_weights_exponential = VectorTypes<double>::select(use_gpu);
    inbound_backward_cumulative_weights_exponential = VectorTypes<double>::select(use_gpu);
}


void NodeEdgeIndex::clear() {
    std::visit([&](auto& outbound_offsets_vec, auto& outbound_indices_vec, auto& outbound_timestamp_group_offsets_vec,
        auto& outbound_timestamp_group_indices_vec, auto& inbound_offsets_vec, auto& inbound_indices_vec,
        auto& inbound_timestamp_group_offsets_vec, auto& inbound_timestamp_group_indices_vec)
    {
        // Clear edge CSR structures
        outbound_offsets_vec.clear();
        outbound_indices_vec.clear();
        outbound_timestamp_group_offsets_vec.clear();
        outbound_timestamp_group_indices_vec.clear();

        // Clear inbound structures
        inbound_offsets_vec.clear();
        inbound_indices_vec.clear();
        inbound_timestamp_group_offsets_vec.clear();
        inbound_timestamp_group_indices_vec.clear();
    }, outbound_offsets, outbound_indices, outbound_timestamp_group_offsets, outbound_timestamp_group_indices,
    inbound_offsets, inbound_indices, inbound_timestamp_group_offsets, inbound_timestamp_group_indices);
}

void NodeEdgeIndex::rebuild(
   const EdgeData& edges,
   const NodeMapping& mapping,
   const bool is_directed) {

   const size_t num_nodes = mapping.size();

    std::visit([&](auto& sources_vec, auto& targets_vec, auto& timestamps_vec,
        auto& outbound_offsets_vec, auto& inbound_offsets_vec, auto& outbound_indices_vec, auto& inbound_indices_vec,
        auto& outbound_timestamp_group_offsets_vec, auto& inbound_timestamp_group_offsets_vec,
        auto& outbound_timestamp_group_indices_vec, auto& inbound_timestamp_group_indices_vec)
    {
        // Initialize base CSR structures
        outbound_offsets_vec.assign(num_nodes + 1, 0);
        outbound_timestamp_group_offsets_vec.assign(num_nodes + 1, 0);

        if (is_directed) {
            inbound_offsets_vec.assign(num_nodes + 1, 0);
            inbound_timestamp_group_offsets_vec.assign(num_nodes + 1, 0);
        }

        // First pass: count edges per node
        for (size_t i = 0; i < edges.size(); i++) {
            const int src_idx = mapping.to_dense(sources_vec[i]);
            const int tgt_idx = mapping.to_dense(targets_vec[i]);

            ++outbound_offsets_vec[src_idx + 1];
            if (is_directed) {
                ++inbound_offsets_vec[tgt_idx + 1];
            } else {
                ++outbound_offsets_vec[tgt_idx + 1];
            }
        }

        // Calculate prefix sums for edge offsets
        for (size_t i = 1; i <= num_nodes; i++) {
            outbound_offsets_vec[i] += outbound_offsets_vec[i-1];
            if (is_directed) {
                inbound_offsets_vec[i] += inbound_offsets_vec[i-1];
            }
        }

        // Allocate edge index arrays
        outbound_indices_vec.resize(outbound_offsets_vec.back());
        if (is_directed) {
            inbound_indices_vec.resize(inbound_offsets_vec.back());
        }

        // Second pass: fill edge indices
        using SizeTVecType = typename RebindVectorT<size_t, std::remove_reference_t<decltype(outbound_offsets_vec)>>::Type;
        SizeTVecType outbound_current(num_nodes, 0);
        SizeTVecType inbound_current;
        if (is_directed) {
            inbound_current.resize(num_nodes, 0);
        }

        for (size_t i = 0; i < edges.size(); i++) {
            const int src_idx = mapping.to_dense(sources_vec[i]);
            const int tgt_idx = mapping.to_dense(targets_vec[i]);

            const size_t out_pos = outbound_offsets_vec[src_idx] + outbound_current[src_idx]++;
            outbound_indices_vec[out_pos] = i;

            if (is_directed) {
                const size_t in_pos = inbound_offsets_vec[tgt_idx] + inbound_current[tgt_idx]++;
                inbound_indices_vec[in_pos] = i;
            } else {
                const size_t out_pos2 = outbound_offsets_vec[tgt_idx] + outbound_current[tgt_idx]++;
                outbound_indices_vec[out_pos2] = i;
            }
        }

        // Third pass: count timestamp groups
        SizeTVecType outbound_group_count(num_nodes, 0);
        SizeTVecType inbound_group_count;
        if (is_directed) {
            inbound_group_count.resize(num_nodes, 0);
        }

        for (size_t node = 0; node < num_nodes; node++) {
            size_t start = outbound_offsets_vec[node];
            size_t end = outbound_offsets_vec[node + 1];

            if (start < end) {
                outbound_group_count[node] = 1;  // First group
                for (size_t i = start + 1; i < end; i++) {
                    if (timestamps_vec[outbound_indices_vec[i]] !=
                        timestamps_vec[outbound_indices_vec[i-1]]) {
                        outbound_group_count[node]++;
                    }
                }
            }

            if (is_directed) {
                start = inbound_offsets_vec[node];
                end = inbound_offsets_vec[node + 1];

                if (start < end) {
                    inbound_group_count[node] = 1;  // First group
                    for (size_t i = start + 1; i < end; i++) {
                        if (timestamps_vec[inbound_indices_vec[i]] !=
                            timestamps_vec[inbound_indices_vec[i-1]]) {
                            inbound_group_count[node]++;
                        }
                    }
                }
            }
        }

        // Calculate prefix sums for group offsets
        for (size_t i = 0; i < num_nodes; i++) {
            outbound_timestamp_group_offsets_vec[i + 1] = outbound_timestamp_group_offsets_vec[i] + outbound_group_count[i];
            if (is_directed) {
                inbound_timestamp_group_offsets_vec[i + 1] = inbound_timestamp_group_offsets_vec[i] + inbound_group_count[i];
            }
        }

        // Allocate and fill group indices
        outbound_timestamp_group_indices_vec.resize(outbound_timestamp_group_offsets_vec.back());
        if (is_directed) {
            inbound_timestamp_group_indices_vec.resize(inbound_timestamp_group_offsets_vec.back());
        }

        // Final pass: fill group indices
        for (size_t node = 0; node < num_nodes; node++) {
            size_t start = outbound_offsets_vec[node];
            size_t end = outbound_offsets_vec[node + 1];
            size_t group_pos = outbound_timestamp_group_offsets_vec[node];

            if (start < end) {
                outbound_timestamp_group_indices_vec[group_pos++] = start;
                for (size_t i = start + 1; i < end; i++) {
                    if (timestamps_vec[outbound_indices_vec[i]] !=
                        timestamps_vec[outbound_indices_vec[i-1]]) {
                        outbound_timestamp_group_indices_vec[group_pos++] = i;
                    }
                }
            }

            if (is_directed) {
                start = inbound_offsets_vec[node];
                end = inbound_offsets_vec[node + 1];
                group_pos = inbound_timestamp_group_offsets_vec[node];

                if (start < end) {
                    inbound_timestamp_group_indices_vec[group_pos++] = start;
                    for (size_t i = start + 1; i < end; i++) {
                        if (timestamps_vec[inbound_indices_vec[i]] !=
                            timestamps_vec[inbound_indices_vec[i-1]]) {
                            inbound_timestamp_group_indices_vec[group_pos++] = i;
                        }
                    }
                }
            }
        }
    }, edges.sources, edges.targets, edges.timestamps, outbound_offsets, inbound_offsets,
    outbound_indices, inbound_indices, outbound_timestamp_group_offsets, inbound_timestamp_group_offsets,
    outbound_timestamp_group_indices, inbound_timestamp_group_indices);
}

void NodeEdgeIndex::update_temporal_weights(const EdgeData& edges, double timescale_bound) {
    const size_t num_nodes = std::visit([](const auto& outbound_offsets_vec)
    {
        return outbound_offsets_vec.size() - 1;
    }, outbound_offsets);

    std::visit([&](const auto& inbound_offsets_vec, const auto& outbound_timestamp_group_indices_vec,
        const auto& inbound_timestamp_group_indices_vec, auto& outbound_forward_cumulative_weights_exponential_vec,
        auto& outbound_backward_cumulative_weights_exponential_vec, auto& inbound_backward_cumulative_weights_exponential_vec)
    {
        outbound_forward_cumulative_weights_exponential_vec.resize(outbound_timestamp_group_indices_vec.size());
        outbound_backward_cumulative_weights_exponential_vec.resize(outbound_timestamp_group_indices_vec.size());
        if (!inbound_offsets_vec.empty()) {
            inbound_backward_cumulative_weights_exponential_vec.resize(inbound_timestamp_group_indices_vec.size());
        }
    }, inbound_offsets, outbound_timestamp_group_indices, inbound_timestamp_group_indices,
    outbound_forward_cumulative_weights_exponential, outbound_backward_cumulative_weights_exponential,
    inbound_backward_cumulative_weights_exponential);

    // Process each node
    for (size_t node = 0; node < num_nodes; node++) {
        std::visit([&](const auto& timestamps_vec, const auto& inbound_offsets_vec, const auto& outbound_indices_vec,
            const auto& inbound_indices_vec, const auto& outbound_timestamp_group_indices_vec,
            const auto& inbound_timestamp_group_indices_vec, auto& outbound_forward_cumulative_weights_exponential_vec,
            auto& outbound_backward_cumulative_weights_exponential_vec, auto& inbound_backward_cumulative_weights_exponential_vec)
        {
            // Outbound weights
            const auto& outbound_timestamp_offsets = get_timestamp_offset_vector(true, false);
            const size_t out_start = std::visit([node](const auto& outbound_timestamp_offsets_vec)
            {
                return outbound_timestamp_offsets_vec[node];
            }, outbound_timestamp_offsets);
            const size_t out_end = std::visit([node](const auto& outbound_timestamp_offsets_vec)
            {
                return outbound_timestamp_offsets_vec[node + 1];
            }, outbound_timestamp_offsets);

            if (out_start < out_end) {
                const size_t first_group_start = outbound_timestamp_group_indices_vec[out_start];
                const size_t last_group_start = outbound_timestamp_group_indices_vec[out_end - 1];
                const int64_t min_ts = timestamps_vec[outbound_indices_vec[first_group_start]];
                const int64_t max_ts = timestamps_vec[outbound_indices_vec[last_group_start]];
                const auto time_diff = static_cast<double>(max_ts - min_ts);
                const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                    timescale_bound / time_diff : 1.0;

                double forward_sum = 0.0;
                double backward_sum = 0.0;

                // Calculate weights and sums
                for (size_t pos = out_start; pos < out_end; ++pos) {
                    const size_t edge_start = outbound_timestamp_group_indices_vec[pos];
                    const int64_t group_ts = timestamps_vec[outbound_indices_vec[edge_start]];

                    const auto time_diff_forward = static_cast<double>(max_ts - group_ts);
                    const auto time_diff_backward = static_cast<double>(group_ts - min_ts);

                    const double forward_scaled = timescale_bound > 0 ?
                        time_diff_forward * time_scale : time_diff_forward;
                    const double backward_scaled = timescale_bound > 0 ?
                        time_diff_backward * time_scale : time_diff_backward;

                    const double forward_weight = exp(forward_scaled);
                    outbound_forward_cumulative_weights_exponential_vec[pos] = forward_weight;
                    forward_sum += forward_weight;

                    const double backward_weight = exp(backward_scaled);
                    outbound_backward_cumulative_weights_exponential_vec[pos] = backward_weight;
                    backward_sum += backward_weight;
                }

                // Normalize and compute cumulative sums
                double forward_cumsum = 0.0, backward_cumsum = 0.0;
                for (size_t pos = out_start; pos < out_end; ++pos) {
                    outbound_forward_cumulative_weights_exponential_vec[pos] /= forward_sum;
                    outbound_backward_cumulative_weights_exponential_vec[pos] /= backward_sum;

                    forward_cumsum += outbound_forward_cumulative_weights_exponential_vec[pos];
                    backward_cumsum += outbound_backward_cumulative_weights_exponential_vec[pos];

                    outbound_forward_cumulative_weights_exponential_vec[pos] = forward_cumsum;
                    outbound_backward_cumulative_weights_exponential_vec[pos] = backward_cumsum;
                }
            }

            // Inbound weights
            if (!inbound_offsets_vec.empty()) {
                const auto& inbound_timestamp_offsets = get_timestamp_offset_vector(false, true);
                const size_t in_start = std::visit([node](const auto& inbound_timestamp_offsets_vec)
                {
                    return inbound_timestamp_offsets_vec[node];
                }, inbound_timestamp_offsets);
                const size_t in_end = std::visit([node](const auto& inbound_timestamp_offsets_vec)
                {
                    return inbound_timestamp_offsets_vec[node + 1];
                }, inbound_timestamp_offsets);

                if (in_start < in_end) {
                    const size_t first_group_start = inbound_timestamp_group_indices_vec[in_start];
                    const size_t last_group_start = inbound_timestamp_group_indices_vec[in_end - 1];
                    const int64_t min_ts = timestamps_vec[inbound_indices_vec[first_group_start]];
                    const int64_t max_ts = timestamps_vec[inbound_indices_vec[last_group_start]];
                    const auto time_diff = static_cast<double>(max_ts - min_ts);
                    const double time_scale = (timescale_bound > 0 && time_diff > 0) ?
                        timescale_bound / time_diff : 1.0;

                    double backward_sum = 0.0;

                    // Calculate weights and sum
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        const size_t edge_start = inbound_timestamp_group_indices_vec[pos];
                        const int64_t group_ts = timestamps_vec[inbound_indices_vec[edge_start]];

                        const auto time_diff_backward = static_cast<double>(group_ts - min_ts);
                        const double backward_scaled = timescale_bound > 0 ?
                            time_diff_backward * time_scale : time_diff_backward;

                        const double backward_weight = exp(backward_scaled);
                        inbound_backward_cumulative_weights_exponential_vec[pos] = backward_weight;
                        backward_sum += backward_weight;
                    }

                    // Normalize and compute cumulative sum
                    double backward_cumsum = 0.0;
                    for (size_t pos = in_start; pos < in_end; ++pos) {
                        inbound_backward_cumulative_weights_exponential_vec[pos] /= backward_sum;
                        backward_cumsum += inbound_backward_cumulative_weights_exponential_vec[pos];
                        inbound_backward_cumulative_weights_exponential_vec[pos] = backward_cumsum;
                    }
                }
            }
        }, edges.timestamps, inbound_offsets, outbound_indices, inbound_indices,
        outbound_timestamp_group_indices, inbound_timestamp_group_indices,
        outbound_forward_cumulative_weights_exponential, outbound_backward_cumulative_weights_exponential,
        inbound_backward_cumulative_weights_exponential);
    }
}

std::pair<size_t, size_t> NodeEdgeIndex::get_edge_range(
   int dense_node_id,
   bool forward,
   bool is_directed) const {

    if (is_directed) {
        const auto& offsets = forward ? outbound_offsets : inbound_offsets;

        return std::visit([&](const auto& offsets_vec)
        {
            if (dense_node_id < 0 || dense_node_id >= offsets_vec.size() - 1) {
                return std::pair<size_t, size_t>{0, 0};
            }
            return std::pair<size_t, size_t>{offsets_vec[dense_node_id], offsets_vec[dense_node_id + 1]};
        }, offsets);
    } else {
        return std::visit([&](const auto& outbound_offsets_vec)
        {
            if (dense_node_id < 0 || dense_node_id >= outbound_offsets_vec.size() - 1) {
                return std::pair<size_t, size_t>{0, 0};
            }
            return std::pair<size_t, size_t>{outbound_offsets_vec[dense_node_id], outbound_offsets_vec[dense_node_id + 1]};
        }, outbound_offsets);
    }
}

std::pair<size_t, size_t> NodeEdgeIndex::get_timestamp_group_range(
   int dense_node_id,
   size_t group_idx,
   bool forward,
   bool is_directed) const {

   const auto& group_offsets = (is_directed && !forward) ?
       inbound_timestamp_group_offsets : outbound_timestamp_group_offsets;
   const auto& group_indices = (is_directed && !forward) ?
       inbound_timestamp_group_indices : outbound_timestamp_group_indices;
   const auto& edge_offsets = (is_directed && !forward) ?
       inbound_offsets : outbound_offsets;

    return std::visit([&](const auto& group_offsets_vec, const auto& group_indices_vec, const auto edge_offsets_vec)
    {
        if (dense_node_id < 0 || dense_node_id >= group_offsets_vec.size() - 1) {
            return std::pair<size_t, size_t>{0, 0};
        }

        size_t num_groups = group_offsets_vec[dense_node_id + 1] - group_offsets_vec[dense_node_id];
        if (group_idx >= num_groups) {
            return std::pair<size_t, size_t>{0, 0};
        }

        size_t group_start_idx = group_offsets_vec[dense_node_id] + group_idx;
        size_t group_start = group_indices_vec[group_start_idx];

        // Group end is either next group's start or node's edge range end
        size_t group_end;
        if (group_idx == num_groups - 1) {
            group_end = edge_offsets_vec[dense_node_id + 1];
        } else {
            group_end = group_indices_vec[group_start_idx + 1];
        }

        return std::pair<size_t, size_t>{group_start, group_end};
    }, group_offsets, group_indices, edge_offsets);
}

size_t NodeEdgeIndex::get_timestamp_group_count(
   int dense_node_id,
   bool forward,
   bool directed) const {

   const auto& offsets = get_timestamp_offset_vector(forward, directed);

    return std::visit([&](const auto& offsets_vec)
    {
        if (dense_node_id < 0 || dense_node_id >= offsets_vec.size() - 1) {
            return static_cast<size_t>(0);
        }

        return static_cast<size_t>(offsets_vec[dense_node_id + 1] - offsets_vec[dense_node_id]);
    }, offsets);
}

[[nodiscard]] VectorTypes<size_t>::Vector NodeEdgeIndex::get_timestamp_offset_vector(
    bool forward,
    bool directed) const {
    return (directed && !forward) ? inbound_timestamp_group_offsets : outbound_timestamp_group_offsets;
}
