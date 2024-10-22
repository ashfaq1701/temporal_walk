#include "TemporalWalk.h"

#include <iostream>

#include "random/UniformRandomPicker.h"
#include "random/LinearRandomPicker.h"
#include "random/ExponentialRandomPicker.h"

TemporalWalk::TemporalWalk(const int num_walks, const int len_walk, RandomPickerType picker_type)
    : num_walks(num_walks), len_walk(len_walk), thread_pool(ThreadPool(std::thread::hardware_concurrency())) {
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

std::vector<std::vector<int>> TemporalWalk::get_random_walks(const int end_node) {
    std::vector walks(num_walks, std::vector<int>());
    for (auto & walk : walks) {
        walk.reserve(len_walk);
    }

    std::vector<std::future<int>> results;

    auto prepare_walk = [&](std::vector<int>* walk) {
        generate_random_walk(walk, end_node);
        return 1;
    };

    for (auto & walk : walks) {
        results.emplace_back(thread_pool.enqueue(prepare_walk, &walk)); // NOLINT(*-inefficient-vector-operation)
    }

    for (auto& future : results) {
        future.wait();
    }

    for (auto & walk : walks) {
        std::reverse(walk.begin(), walk.end());
    }

    return walks;
}

std::unordered_map<int, std::vector<std::vector<int>>> TemporalWalk::get_random_walks_for_nodes(std::vector<int> end_nodes) {
    std::unordered_map<int, std::vector<std::vector<int>>> walk_for_nodes;
    std::vector<std::future<std::vector<std::vector<int>>>> results;

    for (int end_node : end_nodes) {
        walk_for_nodes[end_node] = get_random_walks(end_node);
    }

    return walk_for_nodes;
}

void TemporalWalk::generate_random_walk(std::vector<int>* walk, const int end_node) const {
    Node* end_graph_node;

    if (end_node != -1) {
        end_graph_node = temporal_graph->get_node(end_node);
    } else {
        end_graph_node = temporal_graph->get_random_node(random_picker.get());
    }

    if (end_graph_node == nullptr) {
        return;
    }

    auto current_node = end_graph_node;
    auto current_timestamp = INT64_MAX;

    while (walk->size() < len_walk && current_node != nullptr) {
        walk->push_back(current_node->id);
        const auto picked_edge = current_node->pick_temporal_edge(
            random_picker.get(),
            current_timestamp);

        if (picked_edge == nullptr) {
            current_node = nullptr;
            continue;
        }

        current_node = picked_edge->u;
        current_timestamp = picked_edge->timestamp;
    }
}

void TemporalWalk::add_edge(const int u, const int i, const int64_t t) const {
    temporal_graph->add_edge(u, i, t);
}

void TemporalWalk::add_multiple_edges(const std::vector<EdgeInfo>& edge_infos) const {
    for (const auto& [u, i, t] : edge_infos) {
        add_edge(u, i, t);
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
