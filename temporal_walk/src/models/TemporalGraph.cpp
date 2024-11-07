#include "TemporalGraph.h"

#include <iostream>

#include "../utils/utils.h"
#include "../random/RandomPicker.h"

void TemporalGraph::add_node(const int id) {
    if (nodes.find(id) == nodes.end()) {
        nodes[id] = std::make_shared<Node>(id);
    }
}

Node* TemporalGraph::get_or_create_node(const int id) {
    if (nodes.find(id) == nodes.end()) {
        add_node(id);
    }
    return nodes[id].get();
}

Node* TemporalGraph::get_node(const int id) {
    if (nodes.find(id) == nodes.end()) {
        return nullptr;
    }

    return nodes[id].get();
}

Node* TemporalGraph::get_random_node(RandomPicker* random_picker, const bool begin_from_end) {
    if (edges.empty()) {
        return nullptr;
    }

    const int picked_idx = random_picker->pick_random(0, static_cast<int>(edges.size()), begin_from_end);
    auto it = edges.begin();
    std::advance(it, picked_idx);

    if (it == edges.end() || it->second.empty()) {
        return nullptr;
    }

    const int random_edge_idx = get_random_number(static_cast<int>(it->second.size()));
    return begin_from_end ? it->second[random_edge_idx]->i : it->second[random_edge_idx]->u;
}

void TemporalGraph::add_edge(const int id1, const int id2, int64_t timestamp) {
    Node* node1 = get_or_create_node(id1);
    Node* node2 = get_or_create_node(id2);

    const auto edge = std::make_shared<TemporalEdge>(node1, node2, timestamp);
    node1->add_edges_as_um(edge);
    node2->add_edges_as_dm(edge);

    if (edges.find(edge->timestamp) == edges.end()) {
        edges[edge->timestamp] = std::vector<std::shared_ptr<TemporalEdge>>();;
    }

    edges[edge->timestamp].push_back(edge);
}

void TemporalGraph::delete_edges_less_than_time(const int64_t timestamp) {
    delete_items_less_than_key(edges, timestamp);

    for (auto it = nodes.begin(); it != nodes.end(); ) {
        if (const auto& node_ptr = it->second) {
            node_ptr->delete_edges_less_than_time(timestamp);

            if (node_ptr->is_empty()) {
                it = nodes.erase(it);
                continue;
            }
        }

        ++it;
    }

}

size_t TemporalGraph::get_node_count() const {
    return nodes.size();
}

size_t TemporalGraph::get_edge_count() const {
    size_t total_edges = 0;

    for (const auto& [key, edge_vector] : edges) {
        total_edges += edge_vector.size();
    }

    return total_edges;
}

std::vector<int> TemporalGraph::get_node_ids() {
    std::vector<int> ids;
    ids.reserve(nodes.size());

    for (const auto& [node_id, node_ptr] : nodes) {
        ids.push_back(node_id);
    }

    return ids;
}
