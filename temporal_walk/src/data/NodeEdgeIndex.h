#ifndef NODEEDGEINDEX_H
#define NODEEDGEINDEX_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.h"

struct NodeEdgeIndex
{
private:
    static void create_alias_table(
        const std::vector<double>& weights,
        std::vector<double>& probs,
        std::vector<int>& alias);

public:
    // Base CSR format for edges
    std::vector<size_t> outbound_offsets; // Size: num_nodes + 1
    std::vector<size_t> outbound_indices; // Size: num_edges

    // CSR format for timestamp groups
    std::vector<size_t> outbound_timestamp_group_offsets; // Size: num_nodes + 1
    std::vector<size_t> outbound_timestamp_group_indices; // Each group's start position in outbound_indices

    // Mirror structures for directed graphs
    std::vector<size_t> inbound_offsets;
    std::vector<size_t> inbound_indices;
    std::vector<size_t> inbound_timestamp_group_offsets;
    std::vector<size_t> inbound_timestamp_group_indices;

    // Per-node timestamp selection
    std::vector<double> outbound_ts_prob; // exp(tmax - T(e))/sum
    std::vector<int> outbound_ts_alias; // For outbound edges - favors earlier timestamps
    std::vector<double> inbound_ts_prob; // exp(T(e) - tmin)/sum
    std::vector<int> inbound_ts_alias; // For inbound edges - favors later timestamps

    void clear();
    void rebuild(const EdgeData& edges, const NodeMapping& mapping, bool is_directed);

    // Core access methods
    [[nodiscard]] std::pair<size_t, size_t> get_edge_range(int dense_node_id, bool forward, bool is_directed) const;
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(int dense_node_id, size_t group_idx, bool forward,
                                                                      bool is_directed) const;
    [[nodiscard]] size_t get_timestamp_group_count(int dense_node_id, bool forward, bool is_directed) const;

    void update_temporal_weights(const EdgeData& edges);
};

#endif //NODEEDGEINDEX_H
