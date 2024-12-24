#include "TemporalWalk.h"

#include <iostream>
#include "../utils/utils.h"
#include "../random/UniformRandomPicker.h"
#include "../random/LinearRandomPicker.h"
#include "../random/ExponentialRandomPicker.h"


constexpr int DEFAULT_CONTEXT_WINDOW_LEN = 2;
constexpr int DEFAULT_NUM_WALKS_PER_THREAD = 500;

EdgeInfo::EdgeInfo(const int u, const int i, const int64_t t): u(u), i(i), t(t) {}

TemporalWalk::TemporalWalk(int64_t max_time_capacity): max_time_capacity(max_time_capacity) {
    temporal_graph = std::make_unique<TemporalGraph>();
}

std::shared_ptr<RandomPicker> TemporalWalk::get_random_picker(const RandomPickerType* picker_type) {
    if (!picker_type) {
        throw std::invalid_argument("picker_type cannot be nullptr");
    }

    switch (*picker_type) {
    case Uniform:
        return std::make_shared<UniformRandomPicker>();
    case Linear:
        return std::make_shared<LinearRandomPicker>();
    case Exponential:
        return std::make_shared<ExponentialRandomPicker>();
    default:
        throw std::invalid_argument("Invalid picker type");
    }
}

std::vector<std::vector<NodeWithTime>> TemporalWalk::get_random_walks_with_times(
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const RandomPickerType* initial_edge_bias,
    const long num_cw,
    const int num_walks_per_node,
    const WalkDirection walk_direction,
    const WalkInitEdgeTimeBias walk_init_edge_time_bias,
    const int context_window_len,
    const float p_walk_success_threshold) {

    const std::shared_ptr<RandomPicker> edge_picker = get_random_picker(walk_bias);
    std::shared_ptr<RandomPicker> start_picker;
    if (!initial_edge_bias) {
        start_picker = get_random_picker(initial_edge_bias);
    } else {
        start_picker = edge_picker;
    }

    auto get_init_edge_picker_end_prioritization = [walk_init_edge_time_bias]() {
        switch (walk_init_edge_time_bias) {
        case WalkInitEdgeTimeBias::Bias_Earliest_Time:
            return false;
        case WalkInitEdgeTimeBias::Bias_Latest_Time:
            return true;
        default:
            throw std::invalid_argument("Invalid walk init edge time bias");
        }
    };

    auto get_should_walk_forward = [walk_direction] {
        switch (walk_direction)
        {
        case WalkDirection::Forward_In_Time:
            return true;
        case WalkDirection::Backward_In_Time:
            return false;
        default:
            throw std::invalid_argument("Invalid walk direction");
        }
    };

    int min_walk_len = DEFAULT_CONTEXT_WINDOW_LEN;
    if (context_window_len != -1) {
        min_walk_len = context_window_len;
    }

    long cw_count = num_cw;
    if (num_cw == -1 && num_walks_per_node == -1) {
        throw std::invalid_argument("One of num_cw and num_walks_per_node must be specified.");
    }

    if (num_cw == -1) {
        cw_count = static_cast<long>(get_node_count()) * num_walks_per_node * (max_walk_len - min_walk_len + 1);
    }

    std::atomic<size_t> num_cw_curr{0};
    std::atomic<size_t> successes{0};
    std::atomic<size_t> failures{0};

    // Thread-safe vector for collecting walks
    std::mutex walks_mutex;
    std::vector<std::vector<NodeWithTime>> walks;

    auto generate_walks_thread = [&]()
    {
        while (num_cw_curr < cw_count) {
            const bool init_edge_picker_end_prioritization = get_init_edge_picker_end_prioritization();
            const bool should_walk_forward = get_should_walk_forward();

            std::vector<NodeWithTime> walk;
            walk.reserve(max_walk_len);

            generate_random_walk_with_time(
                &walk,
                edge_picker,
                start_picker,
                max_walk_len,
                should_walk_forward,
                init_edge_picker_end_prioritization);

            if (walk.size() >= min_walk_len) {
                const size_t new_cw = walk.size() - min_walk_len + 1;
                size_t curr_cw = num_cw_curr.fetch_add(new_cw);

                if (curr_cw < cw_count) {
                    std::lock_guard<std::mutex> lock(walks_mutex);
                    walks.push_back(std::move(walk));
                }

                ++successes;
            } else {
                ++failures;

                size_t total = successes + failures;
                if (total > 100) {
                    float success_rate = static_cast<float>(successes) / static_cast<float>(total);
                    if (success_rate < p_walk_success_threshold) {
                        throw std::runtime_error("Too many walks being discarded. Consider using a smaller context window size.");
                    }
                }
            }
        }
    };

    std::vector<std::future<void>> futures;
    const size_t num_threads = std::thread::hardware_concurrency();

    for (size_t i = 0; i < num_threads; ++i)
    {
        futures.push_back(thread_pool.submit_task(generate_walks_thread));
    }

    for (auto& future : futures) {
        try {
            future.get();
        } catch (const std::exception& e) {
            for (auto& f : futures) {
                if (f.valid()) {
                    f.wait();
                }
            }
            throw;
        }
    }

    return walks;
}

std::vector<std::vector<int>> TemporalWalk::get_random_walks(
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const RandomPickerType* initial_edge_bias,
    const long num_cw,
    const int num_walks_per_node,
    const WalkDirection walk_direction,
    const WalkInitEdgeTimeBias walk_init_edge_time_bias,
    const int context_window_len,
    const float p_walk_success_threshold) {

    std::vector<std::vector<NodeWithTime>> walks_with_times = get_random_walks_with_times(max_walk_len, walk_bias, initial_edge_bias, num_cw, num_walks_per_node, walk_direction, walk_init_edge_time_bias, context_window_len, p_walk_success_threshold);
    std::vector<std::vector<int>> walks;

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

void TemporalWalk::generate_random_walk_with_time(
    std::vector<NodeWithTime>* walk,
    const std::shared_ptr<RandomPicker>& edge_picker,
    const std::shared_ptr<RandomPicker>& start_picker,
    const int max_walk_len,
    const bool should_walk_forward,
    const bool init_edge_picker_end_prioritization) const {
    Node* graph_node = temporal_graph->get_random_node(
        start_picker.get(),
        should_walk_forward,
        init_edge_picker_end_prioritization);

    if (graph_node == nullptr) {
        return;
    }

    auto current_node = graph_node;
    auto current_timestamp = should_walk_forward ? INT64_MIN : INT64_MAX;

    while (walk->size() < max_walk_len && current_node != nullptr) {
        walk->emplace_back(NodeWithTime {current_node->id, current_timestamp});
        const auto picked_edge = current_node->pick_temporal_edge(
            edge_picker.get(),
            should_walk_forward,
            current_timestamp);

        if (picked_edge == nullptr) {
            current_node = nullptr;
            continue;
        }

        current_node = should_walk_forward ? picked_edge->i : picked_edge->u;
        current_timestamp = picked_edge->timestamp;
    }
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

size_t TemporalWalk::get_node_count() const {
    return temporal_graph->get_node_count();
}

size_t TemporalWalk::get_edge_count() const {
    return temporal_graph->get_edge_count();
}

std::vector<int> TemporalWalk::get_node_ids() const {
    return temporal_graph->get_node_ids();
}

std::vector<EdgeInfo> TemporalWalk::get_edges() const {
    std::vector<EdgeInfo> edges;

    for (const auto& edge : temporal_graph->get_edges()) {
        edges.emplace_back( edge->u->id, edge->i->id, edge->timestamp );
    }

    return edges;
}

void TemporalWalk::clear() {
    temporal_graph = std::make_unique<TemporalGraph>();
}
