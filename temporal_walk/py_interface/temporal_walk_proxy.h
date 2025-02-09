#ifndef TEMPORAL_WALK_PROXY_H
#define TEMPORAL_WALK_PROXY_H

#include "../src/core/TemporalWalk.cuh"

class TemporalWalkProxy {

    bool use_gpu;

    #ifdef USE_CUDA
    std::unique_ptr<TemporalWalk<false>> cpu_impl;
    std::unique_ptr<TemporalWalk<true>> gpu_impl;
    #else
    std::unique_ptr<TemporalWalk<false>> cpu_impl;
    #endif

public:
    explicit TemporalWalkProxy(
        bool is_directed,
        bool use_gpu=false,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND): use_gpu(use_gpu) {

        #ifdef USE_CUDA
        if (use_gpu) {
            gpu_impl = std::make_unique<TemporalWalk<true>>(
                is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
        } else {
            cpu_impl = std::make_unique<TemporalWalk<false>>(
                is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
        }
        #else
        cpu_impl = std::make_unique<TemporalWalk<false>>(
            is_directed, max_time_capacity, enable_weight_computation, timescale_bound);
        #endif
    }

    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edges) {
        #ifdef USE_CUDA
        if (use_gpu) {
            gpu_impl->add_multiple_edges(edges);
        } else {
            cpu_impl->add_multiple_edges(edges);
        }
        #else
        cpu_impl->add_multiple_edges(edges);
        #endif
    }

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_random_walks_and_times_for_all_nodes(max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);
        else return cpu_impl->get_random_walks_and_times_for_all_nodes(max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);
        #else
        return cpu_impl->get_random_walks_and_times_for_all_nodes(max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);
        #endif
    }

    std::vector<std::vector<int>> get_random_walks_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_random_walks_for_all_nodes(max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);
        else return cpu_impl->get_random_walks_for_all_nodes(max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);
        #else
        return cpu_impl->get_random_walks_for_all_nodes(max_walk_len, walk_bias, num_walks_per_node, initial_edge_bias, walk_direction);
        #endif
    }

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_random_walks_and_times(max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);
        else return cpu_impl->get_random_walks_and_times(max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);
        #else
        return cpu_impl->get_random_walks_and_times(max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);
        #endif
    }

    std::vector<std::vector<int>> get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time) {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_random_walks(max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);
        else return cpu_impl->get_random_walks(max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);
        #else
        return cpu_impl->get_random_walks(max_walk_len, walk_bias, num_walks_total, initial_edge_bias, walk_direction);
        #endif
    }

    std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD) {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_random_walks_and_times_with_specific_number_of_contexts(
            max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
        else return cpu_impl->get_random_walks_and_times_with_specific_number_of_contexts(
            max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
        #else
        return cpu_impl->get_random_walks_and_times_with_specific_number_of_contexts(
            max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
        #endif
    }

    std::vector<std::vector<int>> get_random_walks_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD) {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_random_walks_with_specific_number_of_contexts(
            max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
        else return cpu_impl->get_random_walks_with_specific_number_of_contexts(
            max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
        #else
        return cpu_impl->get_random_walks_with_specific_number_of_contexts(
            max_walk_len, walk_bias, num_cw, num_walks_per_node, initial_edge_bias, walk_direction, context_window_len, p_walk_success_threshold);
        #endif
    }

    [[nodiscard]] size_t get_node_count() const {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_node_count();
        else return cpu_impl->get_node_count();
        #else
        return cpu_impl->get_node_count();
        #endif
    }

    [[nodiscard]] size_t get_edge_count() const {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_edge_count();
        else return cpu_impl->get_edge_count();
        #else
        return cpu_impl->get_edge_count();
        #endif
    }

    [[nodiscard]] std::vector<int> get_node_ids() const {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_node_ids();
        else return cpu_impl->get_node_ids();
        #else
        return cpu_impl->get_node_ids();
        #endif
    }

    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_edges();
        else return cpu_impl->get_edges();
        #else
        return cpu_impl->get_edges();
        #endif
    }

    [[nodiscard]] bool get_is_directed() const {
        #ifdef USE_CUDA
        if (use_gpu) return gpu_impl->get_is_directed();
        else return cpu_impl->get_is_directed();
        #else
        return cpu_impl->get_is_directed();
        #endif
    }

    void clear() const {
        #ifdef USE_CUDA
        if (use_gpu) gpu_impl->clear();
        else cpu_impl->clear();
        #else
        cpu_impl->clear();
        #endif
    }
};

#endif
