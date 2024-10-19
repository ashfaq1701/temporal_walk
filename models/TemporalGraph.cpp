#include "TemporalGraph.h"

void TemporalGraph::addNode(const int id) {
    if (nodes.find(id) == nodes.end()) {
        nodes[id] = std::make_shared<Node>(id);
    }
}

Node* TemporalGraph::getNode(const int id) {
    if (nodes.find(id) == nodes.end()) {
        addNode(id);
    }
    return nodes[id].get();
}

void TemporalGraph::addEdge(const int id1, const int id2, int64_t timestamp) {
    Node* node1 = getNode(id1);
    Node* node2 = getNode(id2);

    auto edge = std::make_shared<TemporalEdge>(node1, node2, timestamp);
    node2->add_edges_as_dm(edge.get());

    edges.push_back(std::move(edge));
}

size_t TemporalGraph::get_node_count() const {
    return nodes.size();
}

size_t TemporalGraph::get_edge_count() const {
    return edges.size();
}
