#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>

struct EdgeData {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;
    std::vector<size_t> timestamp_group_offsets;  // Start of each timestamp group

    void reserve(size_t size);
    void clear();
    size_t size() const;
    bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);
    void update_timestamp_groups();  // Call after sorting
    std::pair<size_t, size_t> get_timestamp_group_range(size_t group_idx) const;
    size_t get_timestamp_group_count() const;
};

#endif //EDGEDATA_H
