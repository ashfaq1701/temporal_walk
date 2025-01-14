#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>

struct EdgeData {
    // Core edge data
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    // Timestamp grouping
    std::vector<size_t> group_offsets;     // Start of each timestamp group
    std::vector<int64_t> unique_timestamps; // Corresponding unique timestamps

    void reserve(size_t size);
    void clear();
    [[nodiscard]] size_t size() const;
    [[nodiscard]] bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);

    std::vector<std::tuple<int, int, int64_t>> get_edges();

    // Group management
    void update_timestamp_groups();  // Call after sorting
    [[nodiscard]] std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;
    [[nodiscard]] size_t get_timestamp_group_count() const;

    // Group lookup
    [[nodiscard]] size_t find_group_after_timestamp(int64_t timestamp) const;  // For forward walks
    [[nodiscard]] size_t find_group_before_timestamp(int64_t timestamp) const; // For backward walks
};

#endif //EDGEDATA_H
