#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <vector>
#include <cstdint>
#include <tuple>
#include "NodeMapping.h"
#include "NodeEdgeIndex.h"
#include "../utils/utils.h"

class TemporalGraph
{
private:
    bool is_directed;
    int64_t time_window; // Time duration to keep edges (-1 means keep all)
    int64_t latest_timestamp; // Track latest edge timestamp
    EdgeData edges; // Main edge storage
    NodeMapping node_mapping; // Sparse to dense node ID mapping
    NodeEdgeIndex node_index; // Node to edge mappings

    void sort_and_merge_edges(size_t start_idx);
    void delete_old_edges();

public:
    explicit TemporalGraph(bool directed, int64_t window = -1);

    // Edge addition
    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& new_edges);

    // Timestamp group counting
    [[nodiscard]] size_t count_timestamps_less_than(int64_t timestamp) const;
    [[nodiscard]] size_t count_timestamps_greater_than(int64_t timestamp) const;
    [[nodiscard]] size_t count_node_timestamps_less_than(int node_id, int64_t timestamp) const;
    [[nodiscard]] size_t count_node_timestamps_greater_than(int node_id, int64_t timestamp) const;

    // Edge selection
    [[nodiscard]] std::tuple<int, int, int64_t> get_edge_at(
        std::function<size_t(int, int, bool)> index_selector, int64_t timestamp = -1,
        bool forward = true) const;
    [[nodiscard]] std::tuple<int, int, int64_t> get_node_edge_at(int node_id,
                                                                 std::function<size_t(int, int, bool)>
                                                                 index_selector, int64_t timestamp = -1,
                                                                 bool forward = true) const;

    // Utility methods
    [[nodiscard]] size_t get_total_edges() const { return edges.size(); }
    [[nodiscard]] size_t get_node_count() const { return node_mapping.active_size(); }
    [[nodiscard]] int64_t get_latest_timestamp() const { return latest_timestamp; }
    [[nodiscard]] std::vector<int> get_node_ids() const;
    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges();
};

#endif //TEMPORALGRAPH_H
