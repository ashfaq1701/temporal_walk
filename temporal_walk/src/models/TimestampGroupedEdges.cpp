#include "TimestampGroupedEdges.h"
#include "TemporalEdge.h"
#include "../utils/utils.h"

TimestampGroupedEdges::TimestampGroupedEdges(const int64_t ts) : timestamp(ts) {}

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

bool TimestampGroupedEdgesComparator::operator()(const std::shared_ptr<TimestampGroupedEdges>& tge, int64_t timestamp) const {
    return tge->get_timestamp() < timestamp;
}

bool TimestampGroupedEdgesComparator::operator()(const std::shared_ptr<TimestampGroupedEdges>& lhs, const std::shared_ptr<TimestampGroupedEdges>& rhs) const {
    return lhs->get_timestamp() < rhs->get_timestamp();
}

bool TimestampGroupedEdgesComparator::operator()(int64_t timestamp, const std::shared_ptr<TimestampGroupedEdges>& tge) const {
    return timestamp < tge->get_timestamp();
}
