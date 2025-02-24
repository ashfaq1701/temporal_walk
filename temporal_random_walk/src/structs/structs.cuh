#ifndef STRUCTS_H
#define STRUCTS_H

#include "../common/macros.cuh"
#include <cstdint>

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

#endif // STRUCTS_H
