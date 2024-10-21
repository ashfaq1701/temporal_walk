#include "TemporalGraph.h"
#include "../utils.h"

void TemporalGraph::add_node(const int id) {
    if (nodes.find(id) == nodes.end()) {
        nodes[id] = std::make_shared<Node>(id);
    }
}

Node* TemporalGraph::get_node(const int id) {
    if (nodes.find(id) == nodes.end()) {
        add_node(id);
    }
    return nodes[id].get();
}

Node* TemporalGraph::get_random_node() {
    if (nodes.empty()) {
        return nullptr;
    }

    const int random_idx = get_random_number(static_cast<int>(nodes.size()));
    auto it = nodes.begin();
    std::advance(it, random_idx);
    return it->second.get();
}

void TemporalGraph::add_edge(const int id1, const int id2, int64_t timestamp) {
    Node* node1 = get_node(id1);
    Node* node2 = get_node(id2);

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
