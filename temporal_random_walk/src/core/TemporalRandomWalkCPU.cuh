#ifndef TEMPORAL_RANDOM_WALK_CPU_H
#define TEMPORAL_RANDOM_WALK_CPU_H

#include<vector>
#include "TemporalRandomWalk.cuh"
#include "../structs/structs.cuh"
#include "../structs/enums.h"
#include "../config/constants.h"
#include "../../libs/thread-pool/ThreadPool.h"
#include "../random/RandomPicker.h"
#include "../data/cpu/TemporalGraphCPU.cuh"
#include "../data/cuda/TemporalGraphCUDA.cuh"
#include "../random/WeightBasedRandomPicker.cuh"
#include "../random/WeightBasedRandomPickerGPU.cuh"

/**
 * @brief Main class for generating temporal random walks
 *
 * Supports both CPU and GPU computation for generating random walks on temporal graphs.
 * The walks respect temporal causality and can be biased using different strategies.
 *
 * @tparam GPUUsage enum to control data placement and computation between CPU and GPU. Possible values are ON_CPU, ON_GPU.
 */
template<GPUUsageMode GPUUsage>
class TemporalRandomWalkCPU : public TemporalRandomWalk<GPUUsage> {
    bool is_directed;

    int64_t max_time_capacity;

    int n_threads;

    bool enable_weight_computation;

    double timescale_bound;

    int64_t max_edge_time = 0;

    std::unique_ptr<TemporalGraph<GPUUsage>> temporal_graph;

    ThreadPool thread_pool;

    HOST void generate_random_walk_and_time(
        std::vector<NodeWithTime>* walk,
        const std::shared_ptr<RandomPicker>& edge_picker,
        const std::shared_ptr<RandomPicker>& start_picker,
        int max_walk_len,
        bool should_walk_forward,
        int start_node_id=-1) const;

    std::shared_ptr<RandomPicker> get_random_picker(const RandomPickerType* picker_type) const;

    [[nodiscard]] long estimate_cw_count(int num_walks_per_node, int max_walk_len, int min_walk_len) const;

public:

    /**
     * @brief Construct a temporal random walk generator
     *
     * @param is_directed Whether the graph is directed
     * @param max_time_capacity Maximum time window for edges (-1 for no limit)
     * @param enable_weight_computation Enable CTDNE weight computation for ExponentialWeight picker
     * @param timescale_bound Scale factor for temporal differences
     * @param n_threads Number of threads for parallel walk generation
     */
    explicit TemporalRandomWalk(
        bool is_directed,
        int64_t max_time_capacity=-1,
        bool enable_weight_computation=false,
        double timescale_bound=DEFAULT_TIMESCALE_BOUND,
        size_t n_threads=std::thread::hardware_concurrency());

    /**
     * @brief Generate temporal random walks starting from all nodes with timestamps
     *
     * @param max_walk_len Maximum length of each walk
     * @param walk_bias Random selection strategy for next edges
     * @param num_walks_per_node Number of walks to generate per starting node
     * @param initial_edge_bias Optional different strategy for first edge
     * @param walk_direction Forward or backward in time
     * @return Vector of walks, each containing node IDs and timestamps
     */
    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    /**
     * @brief Generate temporal random walks from all nodes (without timestamps)
     * Similar to get_random_walks_and_times_for_all_nodes but returns only node IDs.
     *
     * @param max_walk_len Maximum length of each walk
     * @param walk_bias Random selection strategy for next edges
     * @param num_walks_per_node Number of walks to generate per starting node
     * @param initial_edge_bias Optional different strategy for first edge
     * @param walk_direction Forward or backward in time
     * @return Vector of walks, each containing node IDs
     */
    [[nodiscard]] std::vector<std::vector<int>> get_random_walks_for_all_nodes(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_per_node,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    /**
     * @brief Generate temporal random walks starting from random nodes with timestamps
     *
     * @param max_walk_len Maximum length of each walk
     * @param walk_bias Random selection strategy for next edges
     * @param num_walks_total Number of total walks to generate
     * @param initial_edge_bias Optional different strategy for first edge
     * @param walk_direction Forward or backward in time
     * @return Vector of walks, each containing node IDs and timestamps
     */
    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_and_times(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    /**
     * @brief Generate temporal random walks starting from random nodes (without timestamps)
     * Similar to get_random_walks_and_times but returns only node IDs.
     *
     * @param max_walk_len Maximum length of each walk
     * @param walk_bias Random selection strategy for next edges
     * @param num_walks_total Number of total walks to generate
     * @param initial_edge_bias Optional different strategy for first edge
     * @param walk_direction Forward or backward in time
     * @return Vector of walks, each containing node IDs
     */
    [[nodiscard]] std::vector<std::vector<int>> get_random_walks(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        int num_walks_total,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time);

    /**
     * @brief Generate walks with fixed number of context windows
     *
     * @param max_walk_len Maximum length of each walk
     * @param walk_bias Random selection strategy for next edges
     * @param num_cw Target number of context windows (-1 to use num_walks_per_node)
     * @param num_walks_per_node Number of walks to generate per starting node
     * @param initial_edge_bias Optional different strategy for first edge
     * @param walk_direction Forward or backward in time
     * @param context_window_len Minimum walk length (-1 for default)
     * @param p_walk_success_threshold Minimum success rate threshold
     * @return Vector of walks with timestamps
     */
    [[nodiscard]] std::vector<std::vector<NodeWithTime>> get_random_walks_and_times_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD);

    /**
     * @brief Generate walks with fixed number of context windows (without timestamps)
     * Similar to get_random_walks_and_times_with_specific_number_of_contexts but returns only node IDs.
     *
     * @param max_walk_len Maximum length of each walk
     * @param walk_bias Random selection strategy for next edges
     * @param num_cw Target number of context windows (-1 to use num_walks_per_node)
     * @param num_walks_per_node Number of walks to generate per starting node
     * @param initial_edge_bias Optional different strategy for first edge
     * @param walk_direction Forward or backward in time
     * @param context_window_len Minimum walk length (-1 for default)
     * @param p_walk_success_threshold Minimum success rate threshold
     * @return Vector of walks with nodes only
     */
    [[nodiscard]] std::vector<std::vector<int>> get_random_walks_with_specific_number_of_contexts(
        int max_walk_len,
        const RandomPickerType* walk_bias,
        long num_cw=-1,
        int num_walks_per_node=-1,
        const RandomPickerType* initial_edge_bias=nullptr,
        WalkDirection walk_direction=WalkDirection::Forward_In_Time,
        int context_window_len=-1,
        float p_walk_success_threshold=DEFAULT_SUCCESS_THRESHOLD);

    /**
     * @brief Add multiple edges to the graph
     * @param edge_infos Vector of (source, target, timestamp) tuples
     */
    void add_multiple_edges(const std::vector<std::tuple<int, int, int64_t>>& edge_infos) const;

    /** @brief Get number of nodes in the graph */
    [[nodiscard]] size_t get_node_count() const;

    /** @brief Get number of edges in the graph */
    [[nodiscard]] size_t get_edge_count() const;

    /** @brief Get IDs of all nodes */
    [[nodiscard]] std::vector<int> get_node_ids() const;

    /** @brief Get all edges as (source, target, timestamp) tuples */
    [[nodiscard]] std::vector<std::tuple<int, int, int64_t>> get_edges() const;

    /** @brief Check if graph is directed */
    [[nodiscard]] bool get_is_directed() const;

    /** @brief Clear all edges and nodes from the graph */
    void clear();
};

#endif //TEMPORAL_RANDOM_WALK_CPU_H
