#include "TemporalWalk.h"

#include <iostream>
#include "../utils/utils.h"
#include "../random/UniformRandomPicker.h"
#include "../random/LinearRandomPicker.h"
#include "../random/ExponentialRandomPicker.h"

EdgeInfo::EdgeInfo(const int u, const int i, const int64_t t): u(u), i(i), t(t) {}

TemporalWalk::TemporalWalk(const int num_walks, const int len_walk, const RandomPickerType picker_type, int64_t max_time_capacity)
    : num_walks(num_walks), len_walk(len_walk), max_time_capacity(max_time_capacity) {
    temporal_graph = std::make_unique<TemporalGraph>();

    switch (picker_type) {
        case Uniform:
            random_picker = std::make_unique<UniformRandomPicker>();
        break;
        case Linear:
            random_picker = std::make_unique<LinearRandomPicker>();
        break;
        case Exponential:
            random_picker = std::make_unique<ExponentialRandomPicker>();
        break;
        default:
            throw std::invalid_argument("Invalid picker type");
    }
}

std::vector<std::vector<NodeWithTime>> TemporalWalk::get_random_walks_with_times(const WalkStartAt walk_start_at, const int end_node) {
    std::vector walks(num_walks, std::vector<NodeWithTime>());
    for (auto & walk : walks) {
        walk.reserve(len_walk);
    }

    auto get_if_begin_from_end = [walk_start_at]() {
        bool begin_from_end = true;
        switch (walk_start_at) {
            case WalkStartAt::End:
                begin_from_end = true;
            break;
            case WalkStartAt::Begin:
                begin_from_end = false;
            break;
            case WalkStartAt::Random:
                begin_from_end = get_random_boolean();
            break;
        }

        return begin_from_end;
    };

    const size_t batch_size = std::max<size_t>(
        1,
        num_walks / (4 * thread_pool.get_thread_count())
    );

    std::vector<std::future<void>> results;

    for (size_t i = 0; i < walks.size(); i += batch_size) {
        const size_t end = std::min(i + batch_size, walks.size());

        results.emplace_back(
            thread_pool.submit_task([&walks, i, end, &get_if_begin_from_end, this, end_node] {
                for (size_t j = i; j < end; ++j) {
                    auto& walk = walks[j];
                    const bool begin_from_end = get_if_begin_from_end();

                    generate_random_walk_with_time(&walk, begin_from_end, end_node);

                    if (begin_from_end) {
                        std::reverse(walk.begin(), walk.end());
                    }
                }
            })
        );
    }

    for (auto& future : results) {
        future.wait();
    }

    return walks;
}

std::unordered_map<int, std::vector<std::vector<NodeWithTime>>> TemporalWalk::get_random_walks_for_nodes_with_times(const WalkStartAt walk_start_at, const std::vector<int>& end_nodes) {
    std::unordered_map<int, std::vector<std::vector<NodeWithTime>>> walk_for_nodes_with_times;

    for (int end_node : end_nodes) {
        walk_for_nodes_with_times[end_node] = get_random_walks_with_times(walk_start_at, end_node);
    }

    return walk_for_nodes_with_times;
}

void TemporalWalk::generate_random_walk_with_time(std::vector<NodeWithTime>* walk, const bool begin_from_end, const int end_node) const {
    Node* graph_node;

    if (end_node != -1) {
        graph_node = temporal_graph->get_node(end_node);
    } else {
        graph_node = temporal_graph->get_random_node(random_picker.get(), begin_from_end);
    }

    if (graph_node == nullptr) {
        return;
    }

    auto current_node = graph_node;
    auto current_timestamp = begin_from_end ? INT64_MAX : INT64_MIN;

    while (walk->size() < len_walk && current_node != nullptr) {
        walk->emplace_back(NodeWithTime {current_node->id, current_timestamp});
        const auto picked_edge = current_node->pick_temporal_edge(
            random_picker.get(),
            begin_from_end,
            current_timestamp);

        if (picked_edge == nullptr) {
            current_node = nullptr;
            continue;
        }

        current_node = begin_from_end ? picked_edge->u : picked_edge->i;
        current_timestamp = picked_edge->timestamp;
    }
}

std::vector<std::vector<int>> TemporalWalk::get_random_walks(const WalkStartAt walk_start_at, const int end_node) {
    std::vector<std::vector<int>> walks;

    const auto walks_with_times = get_random_walks_with_times(walk_start_at, end_node);
    for (auto & walk_with_time : walks_with_times)
    {
        std::vector<int> walk;

        for (const auto & [node, time] : walk_with_time)
        {
            walk.push_back(node); // NOLINT(*-inefficient-vector-operation)
        }

        walks.push_back(walk);
    }

    return walks;
}

std::unordered_map<int, std::vector<std::vector<int>>> TemporalWalk::get_random_walks_for_nodes(const WalkStartAt walk_start_at, const std::vector<int>& end_nodes) {
    std::unordered_map<int, std::vector<std::vector<int>>> walk_for_nodes;

    for (int end_node : end_nodes) {
        walk_for_nodes[end_node] = get_random_walks(walk_start_at, end_node);
    }

    return walk_for_nodes;
}


void TemporalWalk::add_edge(const int u, const int i, const int64_t t) {
    temporal_graph->add_edge(u, i, t);
    max_edge_time = std::max(max_edge_time, t);
}

void TemporalWalk::add_multiple_edges(const std::vector<EdgeInfo>& edge_infos) {
    for (const auto& [u, i, t] : edge_infos) {
        add_edge(u, i, t);
    }

    temporal_graph->sort_edges();

    if (max_time_capacity != -1) {
        const int64_t min_edge_time = std::max(max_edge_time - max_time_capacity + 1, static_cast<int64_t>(0));
        temporal_graph->delete_edges_less_than_time(min_edge_time);
    }
}

int TemporalWalk::get_len_walk() const {
    return len_walk;
}

size_t TemporalWalk::get_node_count() const {
    return temporal_graph->get_node_count();
}

size_t TemporalWalk::get_edge_count() const {
    return temporal_graph->get_edge_count();
}

std::vector<int> TemporalWalk::get_node_ids() const {
    return temporal_graph->get_node_ids();
}
