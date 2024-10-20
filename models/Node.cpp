#include "Node.h"
#include "../utils.h"

Node::Node(const int nodeId) : id(nodeId) {}

void Node::add_edges_as_dm(TemporalEdge* edge) {
    if (edges_as_dm.find(edge->timestamp) == edges_as_dm.end()) {
        edges_as_dm[edge->timestamp] = std::vector<TemporalEdge*>();
    }
    edges_as_dm[edge->timestamp].push_back(edge);
}

std::vector<int64_t> Node::get_timestamps_less_than_given(int64_t given_timestamp) const {
    return getKeysLessThan(edges_as_dm, given_timestamp);
}
