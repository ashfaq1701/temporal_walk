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

    if (it == edges.end() || it->get()->empty()) {
        return nullptr;
    }

    const auto random_edge = it->get()->select_random_edge();
    return begin_from_end ? random_edge->i : random_edge->u;
}

void TemporalGraph::add_edge(const int id1, const int id2, int64_t timestamp) {
    Node* node1 = get_or_create_node(id1);
    Node* node2 = get_or_create_node(id2);

    const auto edge = std::make_shared<TemporalEdge>(node1, node2, timestamp);
    node1->add_edges_as_um(edge);
    node2->add_edges_as_dm(edge);

    if (edge_index.find(timestamp) == edge_index.end()) {
        const auto group = std::make_shared<TimestampGroupedEdges>(timestamp);
        edge_index[timestamp] = group;
        edges.push_back(group);
    }

    edge_index[edge->timestamp]->add_edge(edge);
}

void TemporalGraph::sort_edges() {
    std::sort(edges.begin(), edges.end(), TimestampGroupedEdgesComparator());

    for (auto & [node_id, node] : nodes) {
        node->sort_edges();
    }
}

void TemporalGraph::delete_edges_less_than_time(const int64_t timestamp) {
    delete_items_less_than_key(edge_index, timestamp);
    delete_items_less_than(edges, timestamp, TimestampGroupedEdgesComparator());

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

    for (const auto& [key, edge_group] : edge_index) {
        total_edges += edge_group->size();
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
