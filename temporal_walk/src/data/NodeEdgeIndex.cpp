#include "NodeEdgeIndex.h"

void NodeEdgeIndex::clear() {
    outbound_offsets.clear();
    outbound_indices.clear();
    inbound_offsets.clear();
    inbound_indices.clear();
}

std::pair<size_t, size_t> NodeEdgeIndex::get_edge_range(
    int dense_node_id, bool forward, bool is_directed) const {

    if (is_directed) {
        const auto& offsets = forward ? outbound_offsets : inbound_offsets;
        if (dense_node_id < 0 || dense_node_id >= offsets.size() - 1) {
            return {0, 0};
        }
        return {offsets[dense_node_id], offsets[dense_node_id + 1]};
    } else {
        if (dense_node_id < 0 || dense_node_id >= outbound_offsets.size() - 1) {
            return {0, 0};
        }
        return {outbound_offsets[dense_node_id], outbound_offsets[dense_node_id + 1]};
    }
}

void NodeEdgeIndex::rebuild(
    const EdgeData& edges, const NodeMapping& mapping, bool is_directed) {

    const size_t num_nodes = mapping.size();
    outbound_offsets.assign(num_nodes + 1, 0);
    if (is_directed) {
        inbound_offsets.assign(num_nodes + 1, 0);
    }

    // Count edges per node
    for (size_t i = 0; i < edges.size(); i++) {
        int src_idx = mapping.to_dense(edges.sources[i]);
        int tgt_idx = mapping.to_dense(edges.targets[i]);

        if (is_directed) {
            outbound_offsets[src_idx + 1]++;
            inbound_offsets[tgt_idx + 1]++;
        } else {
            outbound_offsets[src_idx + 1]++;
            outbound_offsets[tgt_idx + 1]++;
        }
    }

    // Calculate offsets
    for (size_t i = 1; i < outbound_offsets.size(); i++) {
        outbound_offsets[i] += outbound_offsets[i-1];
    }

    if (is_directed) {
        for (size_t i = 1; i < inbound_offsets.size(); i++) {
            inbound_offsets[i] += inbound_offsets[i-1];
        }
    }

    // Allocate indices arrays
    outbound_indices.resize(outbound_offsets.back());
    if (is_directed) {
        inbound_indices.resize(inbound_offsets.back());
    }

    std::vector<size_t> out_current(num_nodes, 0);
    std::vector<size_t> in_current;
    if (is_directed) {
        in_current.resize(num_nodes, 0);
    }

    // Fill indices
    for (size_t i = 0; i < edges.size(); i++) {
        int src_idx = mapping.to_dense(edges.sources[i]);
        int tgt_idx = mapping.to_dense(edges.targets[i]);

        if (is_directed) {
            size_t out_pos = outbound_offsets[src_idx] + out_current[src_idx]++;
            size_t in_pos = inbound_offsets[tgt_idx] + in_current[tgt_idx]++;
            outbound_indices[out_pos] = i;
            inbound_indices[in_pos] = i;
        } else {
            size_t pos1 = outbound_offsets[src_idx] + out_current[src_idx]++;
            size_t pos2 = outbound_offsets[tgt_idx] + out_current[tgt_idx]++;
            outbound_indices[pos1] = i;
            outbound_indices[pos2] = i;
        }
    }

    // Sort indices by timestamp
    auto sort_range = [&edges](auto begin, auto end) {
        std::sort(begin, end, [&edges](size_t a, size_t b) {
            return edges.timestamps[a] < edges.timestamps[b];
        });
    };

    for (size_t node = 0; node < num_nodes; node++) {
        if (is_directed) {
            sort_range(
                inbound_indices.begin() + inbound_offsets[node],
                inbound_indices.begin() + inbound_offsets[node + 1]
            );
            sort_range(
                outbound_indices.begin() + outbound_offsets[node],
                outbound_indices.begin() + outbound_offsets[node + 1]
            );
        } else {
            sort_range(
                outbound_indices.begin() + outbound_offsets[node],
                outbound_indices.begin() + outbound_offsets[node + 1]
            );
        }
    }
}