#include "TemporalWalk.h"
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

std::vector<std::vector<int>> TemporalWalk::get_random_walks(const int start_node) {
    std::vector walks(num_walks, std::vector<int>());
    for (auto & walk : walks) {
        walk.reserve(len_walk);
    }

    std::vector<std::future<int>> results;

    auto prepare_walk = [&](std::vector<int>* walk) {
        generate_random_walk(walk, start_node);
        return 1;
    };

    for (auto & walk : walks) {
        results.emplace_back(thread_pool.enqueue(prepare_walk, &walk));
        generate_random_walk(&walk, start_node);
    }

    for (auto& future : results) {
        future.wait();
    }

    return walks;
}

void TemporalWalk::generate_random_walk(std::vector<int>* walk, const int start_node) const {
    Node* start_graph_node;

    if (start_node != -1) {
        start_graph_node = temporal_graph->get_node(start_node);
    } else {
        start_graph_node = temporal_graph->get_random_node(random_picker.get());
    }

    if (start_graph_node == nullptr) {
        return;
    }

    auto current_node = start_graph_node;
    auto current_timestamp = INT64_MAX;

    while (walk->size() < len_walk && current_node != nullptr) {
        walk->push_back(current_node->id);
        const auto picked_edge = current_node->pick_temporal_edge(
            random_picker.get(),
            current_timestamp);
        current_node = picked_edge->u;
        current_timestamp = picked_edge->timestamp;
    }
}

void TemporalWalk::add_edge(const int u, const int v, const int64_t t) const {
    temporal_graph->add_edge(u, v, t);
}

void TemporalWalk::add_multiple_edges(const std::vector<EdgeInfo>& edge_infos) const {
    for (const auto& [u, v, t] : edge_infos) {
        add_edge(u, v, t);
    }
}
