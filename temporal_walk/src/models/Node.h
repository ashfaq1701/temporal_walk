#ifndef NODE_H
#define NODE_H

#include <map>
#include <vector>
#include <cstdint>
#include <memory>
#include "TemporalEdge.h"
#include "TimestampGroupedEdges.h"
#include "../random/RandomPicker.h"

class TemporalEdge;

class Node {
public:
    int id;

    std::vector<std::shared_ptr<TimestampGroupedEdges>> edges_as_dm;
    std::map<int64_t, std::shared_ptr<TimestampGroupedEdges>> edges_as_dm_index;

    std::vector<std::shared_ptr<TimestampGroupedEdges>> edges_as_um;
    std::map<int64_t, std::shared_ptr<TimestampGroupedEdges>> edges_as_um_index;

    explicit Node(int nodeId);

    void add_edges_as_dm(const std::shared_ptr<TemporalEdge>& edge);

    void add_edges_as_um(const std::shared_ptr<TemporalEdge>& edge);

    void sort_edges();

    void delete_edges_less_than_time(int64_t timestamp);

    [[nodiscard]] size_t count_timestamps_less_than_given(int64_t given_timestamp) const;

    [[nodiscard]] size_t count_timestamps_greater_than_given(int64_t given_timestamp) const;

    TemporalEdge* pick_temporal_edge(RandomPicker* random_picker, bool prioritize_end, int64_t given_timestamp=-1) const;
    bool is_empty() const;
};

#endif //NODE_H
