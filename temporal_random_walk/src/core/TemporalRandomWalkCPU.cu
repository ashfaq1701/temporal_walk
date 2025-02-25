#include "TemporalRandomWalkCPU.cuh"

#include <iostream>
#include "../utils/utils.h"
#include "../random/UniformRandomPicker.cuh"
#include "../random/LinearRandomPicker.cuh"
#include "../data/cpu/TemporalGraphCPU.cuh"
#include "../random/ExponentialIndexRandomPicker.cuh"


constexpr int DEFAULT_CONTEXT_WINDOW_LEN = 2;

template<GPUUsageMode GPUUsage>
TemporalRandomWalkCPU<GPUUsage>::TemporalRandomWalkCPU(
    bool is_directed,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound,
    size_t n_threads):
    ITemporalRandomWalk<GPUUsage>(is_directed, max_time_capacity, enable_weight_computation, timescale_bound),
    n_threads(static_cast<int>(n_threads)), thread_pool(n_threads)
{
    this->temporal_graph = new TemporalGraphCPU<GPUUsage>(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
}

bool get_should_walk_forward(const WalkDirection walk_direction) {
    switch (walk_direction)
    {
    case WalkDirection::Forward_In_Time:
        return true;
    case WalkDirection::Backward_In_Time:
        return false;
    default:
        throw std::invalid_argument("Invalid walk direction");
    }
}

template<GPUUsageMode GPUUsage>
HOST RandomPicker* TemporalRandomWalkCPU<GPUUsage>::get_random_picker(const RandomPickerType* picker_type) const {
    if (!picker_type) {
        throw std::invalid_argument("picker_type cannot be nullptr");
    }

    switch (*picker_type) {
    case Uniform:
        return new UniformRandomPicker<GPUUsage>();
    case Linear:
        return new LinearRandomPicker<GPUUsage>();
    case ExponentialIndex:
        return new ExponentialIndexRandomPicker<GPUUsage>();
    case ExponentialWeight:
        if (!this->enable_weight_computation) {
            throw std::invalid_argument("To enable weight based random pickers, set enable_weight_computation constructor argument to true.");
        }
        return new WeightBasedRandomPicker<GPUUsage>();
    default:
        throw std::invalid_argument("Invalid picker type");
    }
}

template<GPUUsageMode GPUUsage>
std::vector<std::vector<NodeWithTime>> TemporalRandomWalk<GPUUsage>::get_random_walks_and_times_for_all_nodes(
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {

    const std::shared_ptr<RandomPicker> edge_picker = get_random_picker(walk_bias);
    std::shared_ptr<RandomPicker> start_picker;
    if (initial_edge_bias) {
        start_picker = get_random_picker(initial_edge_bias);
    } else {
        start_picker = edge_picker;
    }

    std::vector<int> repeated_node_ids = repeat_elements(get_node_ids(), num_walks_per_node);
    shuffle_vector(repeated_node_ids);
    std::vector<std::vector<int>> distributed_node_ids = divide_vector(repeated_node_ids, n_threads);

    auto generate_walks_thread = [&](const std::vector<int>& start_node_ids) -> std::vector<std::vector<NodeWithTime>> {
        std::vector<std::vector<NodeWithTime>> walks_internal;
        walks_internal.reserve(start_node_ids.size());

        for (const int start_node_id : start_node_ids) {
            const bool should_walk_forward = get_should_walk_forward(walk_direction);

            std::vector<NodeWithTime> walk;
            walk.reserve(max_walk_len);

            generate_random_walk_and_time(
                &walk,
                edge_picker,
                start_picker,
                max_walk_len,
                should_walk_forward,
                start_node_id);

            if (!walk.empty()) {
                walks_internal.emplace_back(std::move(walk));
            }
        }

        return walks_internal;
    };

    std::vector<std::future<std::vector<std::vector<NodeWithTime>>>> futures;
    futures.reserve(distributed_node_ids.size());

    for (auto & node_ids : distributed_node_ids)
    {
        futures.push_back(thread_pool.enqueue(generate_walks_thread, node_ids));
    }

    std::vector<std::vector<NodeWithTime>> walks;

    for (auto& future : futures) {
        try {
            auto walks_in_thread = future.get();
            walks.insert(walks.end(),
                std::make_move_iterator(walks_in_thread.begin()),
                std::make_move_iterator(walks_in_thread.end()));
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

template<GPUUsageMode GPUUsage>
std::vector<std::vector<int>> TemporalRandomWalk<GPUUsage>::get_random_walks_for_all_nodes(
        const int max_walk_len,
        const RandomPickerType* walk_bias,
        const int num_walks_per_node,
        const RandomPickerType* initial_edge_bias,
        const WalkDirection walk_direction) {

    std::vector<std::vector<NodeWithTime>> walks_with_times = get_random_walks_and_times_for_all_nodes(
        max_walk_len,
        walk_bias,
        num_walks_per_node,
        initial_edge_bias,
        walk_direction);

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

template<GPUUsageMode GPUUsage>
std::vector<std::vector<NodeWithTime>> TemporalRandomWalk<GPUUsage>::get_random_walks_and_times(
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const int num_walks_total,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction) {

    const std::shared_ptr<RandomPicker> edge_picker = get_random_picker(walk_bias);
    std::shared_ptr<RandomPicker> start_picker;
    if (initial_edge_bias) {
        start_picker = get_random_picker(initial_edge_bias);
    } else {
        start_picker = edge_picker;
    }

    auto generate_walks_thread = [&](int n_walks) -> std::vector<std::vector<NodeWithTime>> {
        std::vector<std::vector<NodeWithTime>> walks_internal;
        walks_internal.reserve(n_walks);

        int remaining_walks = n_walks;
        while (remaining_walks > 0) {
            const bool should_walk_forward = get_should_walk_forward(walk_direction);

            std::vector<NodeWithTime> walk;
            walk.reserve(max_walk_len);

            generate_random_walk_and_time(
                &walk,
                edge_picker,
                start_picker,
                max_walk_len,
                should_walk_forward);

            if (!walk.empty()) {
                walks_internal.emplace_back(std::move(walk));
                remaining_walks--;
            }
        }

        return walks_internal;
    };

    std::vector<std::future<std::vector<std::vector<NodeWithTime>>>> futures;
    futures.reserve(n_threads);

    auto walks_per_thread = divide_number(num_walks_total, n_threads);
    for (auto & number_of_walks : walks_per_thread)
    {
        futures.push_back(thread_pool.enqueue(generate_walks_thread, number_of_walks));
    }

    std::vector<std::vector<NodeWithTime>> walks;

    for (auto& future : futures) {
        try {
            auto walks_in_thread = future.get();
            walks.insert(walks.end(),
                std::make_move_iterator(walks_in_thread.begin()),
                std::make_move_iterator(walks_in_thread.end()));
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

template<GPUUsageMode GPUUsage>
std::vector<std::vector<int>> TemporalRandomWalk<GPUUsage>::get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias,
        WalkDirection walk_direction) {

    std::vector<std::vector<NodeWithTime>> walks_with_times = get_random_walks_and_times(
        max_walk_len,
        walk_bias,
        num_walks_total,
        initial_edge_bias,
        walk_direction);

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

template<GPUUsageMode GPUUsage>
void TemporalRandomWalk<GPUUsage>::generate_random_walk_and_time(
    std::vector<NodeWithTime>* walk,
    const std::shared_ptr<RandomPicker>& edge_picker,
    const std::shared_ptr<RandomPicker>& start_picker,
    const int max_walk_len,
    const bool should_walk_forward,
    const int start_node_id) const {

    std::tuple<int, int, int64_t> start_edge;
    if (start_node_id == -1) {
        start_edge = temporal_graph->get_edge_at(
            *start_picker,
            -1,
            should_walk_forward);
    } else {
        start_edge = temporal_graph->get_node_edge_at(
            start_node_id,
            *start_picker,
            -1,
            should_walk_forward
        );
    }

    if (std::get<2>(start_edge) == -1) {
        return;
    }

    int current_node = -1;
    auto current_timestamp = should_walk_forward ? INT64_MIN : INT64_MAX;
    auto [start_src, start_dst, start_ts] = start_edge;

    if (is_directed) {
        if (should_walk_forward) {
            walk->emplace_back(NodeWithTime { start_src, current_timestamp });
            current_node = start_dst;
        } else {
            walk->emplace_back(NodeWithTime { start_dst, current_timestamp });
            current_node = start_src;
        }
    } else {
        const int picked_node = start_node_id;
        walk->emplace_back(NodeWithTime { picked_node, current_timestamp });
        current_node = pick_other_number({start_src, start_dst}, picked_node);
    }

    current_timestamp = start_ts;

    while (walk->size() < max_walk_len && current_node != -1) {
        walk->emplace_back(NodeWithTime {current_node, current_timestamp});

        auto [picked_src, picked_dst, picked_ts] = temporal_graph->get_node_edge_at(
            current_node,
            *edge_picker,
            current_timestamp,
            should_walk_forward
        );

        if (picked_ts == -1) {
            current_node = -1;
            continue;
        }

        if (is_directed) {
            current_node = should_walk_forward ? picked_dst : picked_src;
        } else {
            current_node = pick_other_number({picked_src, picked_dst}, current_node);
        }

        current_timestamp = picked_ts;
    }

    if (!should_walk_forward) {
        std::reverse(walk->begin(), walk->end());
    }
}

template<GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCPU<GPUUsage>::add_multiple_edges(const typename ITemporalRandomWalk<GPUUsage>::EdgeVector& edge_infos) const {
    this->temporal_graph->add_multiple_edges(edge_infos);
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalRandomWalkCPU<GPUUsage>::get_node_count() const {
    return this->temporal_graph->get_node_count();
}

template<GPUUsageMode GPUUsage>
HOST size_t TemporalRandomWalkCPU<GPUUsage>::get_edge_count() const {
    return this->temporal_graph->get_total_edges();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalRandomWalk<GPUUsage>::IntVector TemporalRandomWalkCPU<GPUUsage>::get_node_ids() const {
    return this->temporal_graph->get_node_ids();
}

template<GPUUsageMode GPUUsage>
HOST typename ITemporalRandomWalk<GPUUsage>::EdgeVector TemporalRandomWalkCPU<GPUUsage>::get_edges() const {
    return this->temporal_graph->get_edges();
}

template<GPUUsageMode GPUUsage>
HOST bool TemporalRandomWalkCPU<GPUUsage>::get_is_directed() const {
    return this->is_directed;
}

template<GPUUsageMode GPUUsage>
HOST void TemporalRandomWalkCPU<GPUUsage>::clear() {
    this->temporal_graph = new TemporalGraphCPU<GPUUsage>(
        this->is_directed, this->max_time_capacity,
        this->enable_weight_computation, this->timescale_bound);
}

template class TemporalRandomWalkCPU<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalRandomWalk<GPUUsageMode::ON_GPU>;
#endif
