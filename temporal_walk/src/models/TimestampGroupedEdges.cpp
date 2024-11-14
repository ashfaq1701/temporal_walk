#include "TimestampGroupedEdges.h"
#include "../utils/utils.h"

TimestampGroupedEdges::TimestampGroupedEdges(const int64_t ts) : timestamp(ts) {}

// Comparison operators with TimestampGroupedEdges
bool TimestampGroupedEdges::operator<(const TimestampGroupedEdges& other) const {
    return timestamp < other.timestamp;
}

bool TimestampGroupedEdges::operator==(const TimestampGroupedEdges& other) const {
    return timestamp == other.timestamp;
}

// Comparison operators with raw timestamp
bool TimestampGroupedEdges::operator<(int64_t ts) const {
    return timestamp < ts;
}

bool TimestampGroupedEdges::operator==(int64_t ts) const {
    return timestamp == ts;
}

int64_t TimestampGroupedEdges::get_timestamp() const {
    return timestamp;
}


void TimestampGroupedEdges::add_edge(const std::shared_ptr<TemporalEdge>& edge) {
    edges.push_back(edge);
}

TemporalEdge* TimestampGroupedEdges::select_random_edge() const {
    const int random_edge_idx = get_random_number(static_cast<int>(edges.size()));
    return edges[random_edge_idx].get();
}

bool TimestampGroupedEdges::empty() const {
    return edges.empty();
}

size_t TimestampGroupedEdges::size() const {
    return edges.size();
}
