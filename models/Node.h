#ifndef NODE_H
#define NODE_H

#include <map>
#include "TemporalEdge.h"

class TemporalEdge;

class Node {
public:
    int id;
    std::map<int64_t, std::vector<TemporalEdge*>> edges_as_dm;

    explicit Node(int nodeId);

    void add_edges_as_dm(TemporalEdge* edge);

    [[nodiscard]] size_t count_timestamps_less_than_given(int64_t given_timestamp) const;
};



#endif //NODE_H
