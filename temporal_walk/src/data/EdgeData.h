#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>

struct EdgeData {
private:
    static void create_alias_table(
        const std::vector<double>& weights,
        std::vector<double>& probs,
        std::vector<int>& alias);

public:

    // Core edge data
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    // Timestamp grouping
    std::vector<size_t> timestamp_group_offsets;     // Start of each timestamp group
    std::vector<int64_t> unique_timestamps;          // Corresponding unique timestamps

    // Global timestamp selection
    std::vector<double> forward_ts_prob;  // exp(tmax - T(e))/sum
    std::vector<int> forward_ts_alias;    // For forward walks - favors earlier timestamps
    std::vector<double> backward_ts_prob; // exp(T(e) - tmin)/sum
    std::vector<int> backward_ts_alias;   // For backward walks - favors later timestamps

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);

    std::vector<std::tuple<int, int, int64_t>> get_edges();

    // Group management
    void update_timestamp_groups();  // Call after sorting
    void update_temporal_weights();

    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const; // For backward walks
};

#endif //EDGEDATA_H
