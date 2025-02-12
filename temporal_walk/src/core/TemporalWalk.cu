#include "TemporalWalk.cuh"

#include <iostream>
#include "../utils/utils.h"
#include "../random/UniformRandomPicker.cuh"
#include "../random/LinearRandomPicker.cuh"
#include "../random/ExponentialIndexRandomPicker.cuh"
#include "../random/WeightBasedRandomPicker.cuh"


constexpr int DEFAULT_CONTEXT_WINDOW_LEN = 2;

template<GPUUsageMode GPUUsage>
TemporalWalk<GPUUsage>::TemporalWalk(
    bool is_directed,
    int64_t max_time_capacity,
    bool enable_weight_computation,
    double timescale_bound,
    size_t n_threads):
    is_directed(is_directed), max_time_capacity(max_time_capacity),
    n_threads(static_cast<int>(n_threads)), enable_weight_computation(enable_weight_computation),
    timescale_bound(timescale_bound), thread_pool(n_threads)
{
    #ifndef HAS_CUDA
    if (GPUUsage != ON_CPU) {
        throw std::runtime_error("GPU support is not available, only \"ON_CPU\" version is available.");
    }
    #endif

    #ifdef HAS_CUDA
    temporal_graph = std::make_unique<TemporalGraphCUDA<GPUUsage>>(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
    #else
    temporal_graph = std::make_unique<TemporalGraph<GPUUsageMode::ON_CPU>>(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
    #endif
}

bool get_should_walk_forward(WalkDirection walk_direction) {
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
std::shared_ptr<RandomPicker> TemporalWalk<GPUUsage>::get_random_picker(const RandomPickerType* picker_type) const {
    if (!picker_type) {
        throw std::invalid_argument("picker_type cannot be nullptr");
    }

    switch (*picker_type) {
    case Uniform:
        return std::make_shared<UniformRandomPicker<GPUUsage>>();
    case Linear:
        return std::make_shared<LinearRandomPicker<GPUUsage>>();
    case ExponentialIndex:
        return std::make_shared<ExponentialIndexRandomPicker<GPUUsage>>();
    case ExponentialWeight:
        if (!enable_weight_computation) {
            throw std::invalid_argument("To enable weight based random pickers, set enable_weight_computation constructor argument to true.");
        }
        return std::make_shared<WeightBasedRandomPicker<GPUUsage>>();
    default:
        throw std::invalid_argument("Invalid picker type");
    }
}

template<GPUUsageMode GPUUsage>
std::vector<std::vector<NodeWithTime>> TemporalWalk<GPUUsage>::get_random_walks_and_times_for_all_nodes(
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
std::vector<std::vector<int>> TemporalWalk<GPUUsage>::get_random_walks_for_all_nodes(
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
std::vector<std::vector<NodeWithTime>> TemporalWalk<GPUUsage>::get_random_walks_and_times(
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
std::vector<std::vector<int>> TemporalWalk<GPUUsage>::get_random_walks(
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
std::vector<std::vector<NodeWithTime>> TemporalWalk<GPUUsage>::get_random_walks_and_times_with_specific_number_of_contexts(
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const long num_cw,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const int context_window_len,
    const float p_walk_success_threshold) {

    const std::shared_ptr<RandomPicker> edge_picker = get_random_picker(walk_bias);
    std::shared_ptr<RandomPicker> start_picker;
    if (initial_edge_bias) {
        start_picker = get_random_picker(initial_edge_bias);
    } else {
        start_picker = edge_picker;
    }

    int min_walk_len = DEFAULT_CONTEXT_WINDOW_LEN;
    if (context_window_len != -1) {
        min_walk_len = context_window_len;
    }

    long cw_count = num_cw;
    if (num_cw == -1 && num_walks_per_node == -1) {
        throw std::invalid_argument("One of num_cw and num_walks_per_node must be specified.");
    }

    if (num_cw == -1) {
        cw_count = estimate_cw_count(num_walks_per_node, max_walk_len, min_walk_len);
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
            const bool should_walk_forward = get_should_walk_forward(walk_direction);

            std::vector<NodeWithTime> walk;
            walk.reserve(max_walk_len);

            generate_random_walk_and_time(
                &walk,
                edge_picker,
                start_picker,
                max_walk_len,
                should_walk_forward);

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
                    const float posterior = compute_beta_95th_percentile(successes, failures);
                    if (posterior < p_walk_success_threshold) {
                        throw std::runtime_error("Too many walks being discarded. "
                                                 "Consider using a smaller context window size.");
                    }
                }
            }
        }
    };

    std::vector<std::future<void>> futures;
    futures.reserve(n_threads);

    for (size_t i = 0; i < n_threads; ++i)
    {
        futures.push_back(thread_pool.enqueue(generate_walks_thread));
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

template<GPUUsageMode GPUUsage>
std::vector<std::vector<int>> TemporalWalk<GPUUsage>::get_random_walks_with_specific_number_of_contexts(
    const int max_walk_len,
    const RandomPickerType* walk_bias,
    const long num_cw,
    const int num_walks_per_node,
    const RandomPickerType* initial_edge_bias,
    const WalkDirection walk_direction,
    const int context_window_len,
    const float p_walk_success_threshold) {

    std::vector<std::vector<NodeWithTime>> walks_with_times = get_random_walks_and_times_with_specific_number_of_contexts(max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
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
void TemporalWalk<GPUUsage>::generate_random_walk_and_time(
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
void TemporalWalk<GPUUsage>::add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edge_infos) const {
    temporal_graph->add_multiple_edges(edge_infos);
}

template<GPUUsageMode GPUUsage>
size_t TemporalWalk<GPUUsage>::get_node_count() const {
    return temporal_graph->get_node_count();
}

template<GPUUsageMode GPUUsage>
long TemporalWalk<GPUUsage>::estimate_cw_count(
    const int num_walks_per_node,
    const int max_walk_len,
    const int min_walk_len) const {

    return static_cast<long>(get_node_count()) * num_walks_per_node * (max_walk_len - min_walk_len + 1);
}

template<GPUUsageMode GPUUsage>
size_t TemporalWalk<GPUUsage>::get_edge_count() const {
    return temporal_graph->get_total_edges();
}

template<GPUUsageMode GPUUsage>
std::vector<int> TemporalWalk<GPUUsage>::get_node_ids() const {
    return temporal_graph->get_node_ids();
}

template<GPUUsageMode GPUUsage>
std::vector<std::tuple<int, int, int64_t>> TemporalWalk<GPUUsage>::get_edges() const {
    return temporal_graph->get_edges();
}

template<GPUUsageMode GPUUsage>
bool TemporalWalk<GPUUsage>::get_is_directed() const {
    return is_directed;
}

template<GPUUsageMode GPUUsage>
void TemporalWalk<GPUUsage>::clear() {
    #ifdef HAS_CUDA
    temporal_graph = std::make_unique<TemporalGraphCUDA<GPUUsage>>(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
    #else
    temporal_graph = std::make_unique<TemporalGraph<GPUUsageMode::ON_CPU>>(
        is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
    #endif
}

template class TemporalWalk<GPUUsageMode::ON_CPU>;
#ifdef HAS_CUDA
template class TemporalWalk<GPUUsageMode::DATA_ON_GPU>;
template class TemporalWalk<GPUUsageMode::DATA_ON_HOST>;
#endif
