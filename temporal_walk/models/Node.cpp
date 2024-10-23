#include "Node.h"

#include <iostream>

#include "../utils/utils.h"
#include "../random/RandomPicker.h"

Node::Node(const int nodeId) : id(nodeId) {}

void Node::add_edges_as_dm(const std::shared_ptr<TemporalEdge>& edge) {
    if (edges_as_dm.find(edge->timestamp) == edges_as_dm.end()) {
        edges_as_dm[edge->timestamp] = std::vector<std::shared_ptr<TemporalEdge>>();
    }
    edges_as_dm[edge->timestamp].push_back(edge);
}

void Node::add_edges_as_um(const std::shared_ptr<TemporalEdge>& edge) {
    if (edges_as_um.find(edge->timestamp) == edges_as_um.end()) {
        edges_as_um[edge->timestamp] = std::vector<std::shared_ptr<TemporalEdge>>();
    }
    edges_as_um[edge->timestamp].push_back(edge);
}

size_t Node::count_timestamps_less_than_given(const int64_t given_timestamp) const {
    return countKeysLessThan(edges_as_dm, given_timestamp);
}

size_t Node::count_timestamps_greater_than_given(const int64_t given_timestamp) const {
    return countKeysGreaterThan(edges_as_um, given_timestamp);
}

TemporalEdge* Node::pick_temporal_edge(RandomPicker* random_picker, const bool prioritize_end, const int64_t given_timestamp) const {
    const auto map_to_use = prioritize_end ? edges_as_dm : edges_as_um;

    size_t count_edge_times = map_to_use.size();
    if (given_timestamp != -1) {
        if (prioritize_end) {
            count_edge_times = count_timestamps_less_than_given(given_timestamp);
        } else {
            count_edge_times = count_timestamps_greater_than_given(given_timestamp);
        }
    }

    if (count_edge_times == 0) {
        return nullptr;
    }

    const int random_timestamp_idx = random_picker->pick_random(0, static_cast<int>(count_edge_times), prioritize_end);

    auto it = prioritize_end ? map_to_use.begin() : map_to_use.end();

    if (prioritize_end) {
        std::advance(it, random_timestamp_idx);
    } else {
        std::advance(it, -(count_edge_times - random_timestamp_idx));
    }
    const auto edges_at_chosen_timestamp = it->second;

    const int random_edge_idx = get_random_number(static_cast<int>(edges_at_chosen_timestamp.size()));
    return edges_at_chosen_timestamp[random_edge_idx].get();
}
