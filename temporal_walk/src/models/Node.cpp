#include "Node.h"

#include <iostream>

#include "../utils/utils.h"
#include "../random/RandomPicker.h"

Node::Node(const int nodeId) : id(nodeId) {}

void Node::add_edges_as_dm(const std::shared_ptr<TemporalEdge>& edge) {
    if (edges_as_dm_index.find(edge->timestamp) == edges_as_dm_index.end()) {
        const auto group = std::make_shared<TimestampGroupedEdges>(edge->timestamp);
        edges_as_dm_index[edge->timestamp] = group;
        edges_as_dm.push_back(group);
    }
    edges_as_dm_index[edge->timestamp]->add_edge(edge);
}

void Node::add_edges_as_um(const std::shared_ptr<TemporalEdge>& edge) {
    if (edges_as_um_index.find(edge->timestamp) == edges_as_um_index.end()) {
        const auto group = std::make_shared<TimestampGroupedEdges>(edge->timestamp);
        edges_as_um_index[edge->timestamp] = group;
        edges_as_um.push_back(group);
    }
    edges_as_um_index[edge->timestamp]->add_edge(edge);
}

void Node::sort_edges() {
    std::sort(edges_as_dm.begin(), edges_as_dm.end(), TimestampGroupedEdgesComparator());
    std::sort(edges_as_um.begin(), edges_as_um.end(), TimestampGroupedEdgesComparator());
}


void Node::delete_edges_less_than_time(const int64_t timestamp) {
    delete_items_less_than_key(edges_as_dm_index, timestamp);
    delete_items_less_than(edges_as_dm, timestamp, TimestampGroupedEdgesComparator());

    delete_items_less_than_key(edges_as_um_index, timestamp);
    delete_items_less_than(edges_as_um, timestamp, TimestampGroupedEdgesComparator());
}

size_t Node::count_timestamps_less_than_given(const int64_t given_timestamp) const {
    return count_elements_less_than(edges_as_dm, given_timestamp, TimestampGroupedEdgesComparator());
}

size_t Node::count_timestamps_greater_than_given(const int64_t given_timestamp) const {
    return count_elements_greater_than(edges_as_um, given_timestamp, TimestampGroupedEdgesComparator());
}

TemporalEdge* Node::pick_temporal_edge(RandomPicker* random_picker, const bool prioritize_end, const int64_t given_timestamp) const {
    const auto list_to_use = prioritize_end ? edges_as_dm : edges_as_um;

    size_t count_edge_times = list_to_use.size();
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

    auto it = prioritize_end ? list_to_use.begin() : list_to_use.end();

    if (prioritize_end) {
        std::advance(it, random_timestamp_idx);
    } else {
        std::advance(it, -(count_edge_times - random_timestamp_idx));
    }
    return it->get()->select_random_edge();
}

bool Node::is_empty() const {
    return edges_as_dm.empty() && edges_as_um.empty();
}
