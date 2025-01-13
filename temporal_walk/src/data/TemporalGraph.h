#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.h"
#include "NodeEdgeIndex.h"
#include "../utils/utils.h"

// In header file (TemporalGraph.h):
class TemporalGraph {
private:
    bool is_directed;
    int64_t time_window;         // Time duration to keep edges (-1 means keep all)
    int64_t latest_timestamp;    // Track latest edge timestamp
    EdgeData edges;              // Main edge storage
    NodeMapping node_mapping;    // Sparse to dense node ID mapping
    NodeEdgeIndex node_index;    // Node to edge mappings

    void sort_and_merge_edges(size_t start_idx);
    void delete_old_edges();

public:
    explicit TemporalGraph(bool directed, int64_t window = -1);

    // Edge addition
    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& new_edges);

    // Timestamp group counting
    size_t count_timestamps_less_than(int64_t timestamp) const;
    size_t count_timestamps_greater_than(int64_t timestamp) const;
    size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;
    size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    // Edge selection
    std::tuple<int, int, int64_t> get_edge_at(size_t index, int64_t timestamp = -1, bool forward = true) const;
    std::tuple<int, int, int64_t> get_node_edge_at(int node_id, size_t index, int64_t timestamp = -1, bool forward = true) const;

    // Utility methods
    size_t get_total_edges() const { return edges.size(); }
    size_t get_node_count() const { return node_mapping.size(); }
    int64_t get_latest_timestamp() const { return latest_timestamp; }
};

#endif //TEMPORALGRAPH_H
