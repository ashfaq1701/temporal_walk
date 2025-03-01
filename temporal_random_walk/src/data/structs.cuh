#ifndef STRUCTS_H
#define STRUCTS_H

#include "../cuda_common/macros.cuh"
#include <cstddef>
#include <cstdint>

#include "enums.h"
#include "common_vector.cuh"


struct SizeRange {
    size_t from;
    size_t to;

    HOST DEVICE SizeRange(): from(0), to(0) {}

    HOST DEVICE explicit SizeRange(size_t f, size_t t) : from(f), to(t) {}

    HOST DEVICE SizeRange& operator=(const SizeRange& other)
    {
        if (this != &other)
        {
            from = other.from;
            to = other.to;
        }
        return *this;
    }
};

template <typename T, typename U>
struct IndexValuePair {
    T index;
    U value;

    HOST DEVICE IndexValuePair() : index(), value() {}
    HOST DEVICE IndexValuePair(const T& idx, const U& val) : index(idx), value(val) {}

    // Optional: Add assignment operator for full compatibility
    HOST DEVICE IndexValuePair& operator=(const IndexValuePair& other) {
        if (this != &other) {
            index = other.index;
            value = other.value;
        }
        return *this;
    }
};

struct Edge {
    int u;
    int i;
    int64_t ts;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1) {}

    HOST DEVICE explicit Edge(int u, int i, int64_t ts) : u(u), i(i), ts(ts) {}

    HOST DEVICE Edge& operator=(const Edge& other) {
        if (this != &other) {
            u = other.u;
            i = other.i;
            ts = other.ts;
        }
        return *this;
    }
};

struct NodeWithTime {
    int node;
    int64_t timestamp;

    HOST DEVICE NodeWithTime(): node(-1), timestamp(-1) {}

    HOST DEVICE NodeWithTime(int node, int64_t timestamp): node(node), timestamp(timestamp) {}

    HOST DEVICE NodeWithTime& operator=(const NodeWithTime& other)
    {
        if (this != &other)
        {
            node = other.node;
            timestamp = other.timestamp;
        }

        return *this;
    }
};

template<GPUUsageMode GPUUsage>
struct WalkSet
{
    size_t num_walks;
    size_t max_len;

    std::vector<int> nodes;
    std::vector<int64_t> timestamps;
    std::vector<size_t> walk_lens;

    HOST DEVICE WalkSet(): num_walks(0), max_len(0), nodes({}), timestamps({}), walk_lens({}) {}

    HOST DEVICE explicit WalkSet(size_t num_walks, size_t max_len)
        : num_walks(num_walks), max_len(max_len), nodes({}), timestamps({}), walk_lens({})
    {
        nodes.resize(num_walks * max_len);
        timestamps.resize(num_walks * max_len);
        walk_lens.resize(num_walks);
    }

    HOST DEVICE void add_hop(int walk_number, int node, int64_t timestamp)
    {
        size_t offset = walk_number * max_len + walk_lens[walk_number];
        nodes[offset] = node;
        timestamps[offset] = timestamp;
        ++walk_lens[walk_number];
    }

    HOST DEVICE size_t get_walk_len(int walk_number)
    {
        return walk_lens[walk_number];
    }

    HOST DEVICE NodeWithTime get_walk_hop(int walk_number, int hop_number)
    {
        size_t walk_length = walk_lens[walk_number];
        if (hop_number < 0 || hop_number >= walk_length) {
            return NodeWithTime{-1, -1};  // Return invalid entry
        }

        // Compute offset safely
        size_t offset = walk_number * max_len + hop_number;
        return NodeWithTime{nodes[offset], timestamps[offset]};
    }

    HOST DEVICE void reverse_walk(int walk_number)
    {
        const size_t walk_length = walk_lens[walk_number];
        if (walk_length <= 1) return; // No need to reverse if walk is empty or has one hop

        const size_t start = walk_number * max_len;
        const size_t end = start + walk_length - 1;

        for (size_t i = 0; i < walk_length / 2; ++i) {
            // Swap nodes
            int temp_node = nodes[start + i];
            nodes[start + i] = nodes[end - i];
            nodes[end - i] = temp_node;

            // Swap timestamps
            int64_t temp_time = timestamps[start + i];
            timestamps[start + i] = timestamps[end - i];
            timestamps[end - i] = temp_time;
        }
    }
};

template <typename T>
struct DividedVector {
    std::vector<IndexValuePair<int, T>> elements;
    std::vector<T> group_offsets;
    size_t num_groups;

    // Constructor - divides input vector into n groups
    HOST DEVICE DividedVector(const std::vector<T>& input, int n)
        : num_groups(n)
    {
        const int total_size = static_cast<int>(input.size());
        const int base_size = total_size / n;
        const int remainder = total_size % n;

        // Reserve space for group offsets (n+1 offsets for n groups)
        group_offsets.reserve(n + 1);

        // Calculate and store group offsets
        size_t current_offset = 0;
        group_offsets.push_back(current_offset);

        for (int i = 0; i < n; i++) {
            const int group_size = base_size + (i < remainder ? 1 : 0);
            current_offset += group_size;
            group_offsets.push_back(current_offset);
        }

        // Allocate space for all elements
        elements.reserve(total_size);

        // Populate the elements array
        for (int i = 0; i < n; i++) {
            const size_t start_idx = group_offsets[i];
            const size_t end_idx = group_offsets[i + 1];

            for (size_t j = start_idx; j < end_idx; ++j) {
                elements.push_back(IndexValuePair<int, T>(j, input[j]));
            }
        }
    }

    // Get begin iterator for a specific group
    HOST DEVICE IndexValuePair<int, T>* group_begin(size_t group_idx) {
        if (group_idx >= num_groups) {
            return nullptr;
        }
        return elements.data() + group_offsets[group_idx];
    }

    // Get end iterator for a specific group
    HOST DEVICE IndexValuePair<int, T>* group_end(size_t group_idx) {
        if (group_idx >= num_groups) {
            return nullptr;
        }
        return elements.data() + group_offsets[group_idx + 1];
    }

    // Get size of a specific group
    HOST DEVICE [[nodiscard]] size_t group_size(size_t group_idx) const {
        if (group_idx >= num_groups) {
            return 0;
        }
        return group_offsets[group_idx + 1] - group_offsets[group_idx];
    }

    // Helper class for iterating over a group
    struct GroupIterator {
        DividedVector& divided_vector;
        size_t group_idx;

        HOST DEVICE GroupIterator(DividedVector& dv, size_t idx)
            : divided_vector(dv), group_idx(idx) {}

        HOST DEVICE IndexValuePair<int, T>* begin() const {
            return divided_vector.group_begin(group_idx);
        }

        HOST DEVICE IndexValuePair<int, T>* end() const {
            return divided_vector.group_end(group_idx);
        }

        HOST DEVICE [[nodiscard]] size_t size() const {
            return divided_vector.group_size(group_idx);
        }
    };

    // Get an iterator for a specific group
    HOST DEVICE GroupIterator get_group(size_t group_idx) {
        return GroupIterator(*this, group_idx);
    }
};

#endif // STRUCTS_H
