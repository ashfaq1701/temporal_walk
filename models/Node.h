#ifndef NODE_H
#define NODE_H

#include <map>
#include "TemporalEdge.h"
#include "../random/RandomPicker.h"

class TemporalEdge;

class Node {
public:
    int id;
    std::map<int64_t, std::vector<std::shared_ptr<TemporalEdge>>> edges_as_dm;

    explicit Node(int nodeId);

    void add_edges_as_dm(const std::shared_ptr<TemporalEdge>& edge);

    [[nodiscard]] size_t count_timestamps_less_than_given(int64_t given_timestamp) const;

    TemporalEdge* pick_temporal_edge(RandomPicker* random_picker, int64_t given_timestamp=-1);
};

#endif //NODE_H
