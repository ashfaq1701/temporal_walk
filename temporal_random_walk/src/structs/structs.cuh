#ifndef STRUCTS_H
#define STRUCTS_H

#include "../common/macros.cuh"
#include <cstddef>
#include <cstdint>

#include "enums.h"
#include "../common/data/common_vector.cuh"

struct SizeRange {
    size_t from;
    size_t to;

    HOST DEVICE SizeRange(): from(0), to(0) {}

    HOST DEVICE explicit SizeRange(size_t f, size_t t) : from(f), to(t) {}
};

struct Edge {
    int u;
    int i;
    int64_t ts;

    HOST DEVICE Edge(): u(-1), i(-1), ts(-1) {}

    HOST DEVICE explicit Edge(int u, int i, int ts) : u(u), i(i), ts(ts) {}
};

struct NodeWithTime {
    int node;
    int64_t timestamp;

    HOST DEVICE NodeWithTime(): node(-1), timestamp(-1) {}

    HOST DEVICE NodeWithTime(int node, int64_t timestamp): node(node), timestamp(timestamp) {}
};

template<GPUUsageMode GPUUsage>
struct WalkSet
{
    size_t num_walks;
    size_t max_len;

    CommonVector<int, GPUUsage> nodes;
    CommonVector<int64_t, GPUUsage> timestamps;
    CommonVector<size_t, GPUUsage> walk_lens;

    HOST DEVICE WalkSet(): num_walks(0), max_len(0), nodes(nullptr), timestamps(nullptr), walk_lens(nullptr) {}

    HOST DEVICE explicit WalkSet(size_t num_walks, size_t max_len)
        : num_walks(num_walks), max_len(max_len), nodes({}), timestamps({}), walk_lens({})
    {
        nodes.allocate(num_walks * max_len);
        timestamps.allocate(num_walks * max_len);
        walk_lens.allocate(num_walks);
    }

    HOST DEVICE void add_hop(int walk_number, int node, int64_t timestamp)
    {
        size_t offset = walk_number * max_len + walk_lens[walk_number];
        nodes[offset] = node;
        timestamps[offset] = timestamp;
        ++walk_lens[walk_number];
    }

    HOST DEVICE void get_walk_len(int walk_number)
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

#endif // STRUCTS_H
