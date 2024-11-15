#ifndef TIMESTAMPGROUPEDEDGES_H
#define TIMESTAMPGROUPEDEDGES_H

#include <cstdint>
#include <vector>
#include <memory>

class TemporalEdge;

class TimestampGroupedEdges {
public:
    int64_t timestamp;
    std::vector<std::shared_ptr<TemporalEdge>> edges;

    explicit TimestampGroupedEdges(int64_t ts);

    [[nodiscard]] int64_t get_timestamp() const;

    void add_edge(const std::shared_ptr<TemporalEdge>& edge);

    [[nodiscard]] TemporalEdge* select_random_edge() const;

    [[nodiscard]] bool empty() const;

    [[nodiscard]] size_t size() const;
};

struct TimestampGroupedEdgesComparator {
    bool operator()(int64_t timestamp, const std::shared_ptr<TimestampGroupedEdges>& tge) const;
    bool operator()(const std::shared_ptr<TimestampGroupedEdges>& tge, int64_t timestamp) const;
    bool operator()(const std::shared_ptr<TimestampGroupedEdges>& lhs, const std::shared_ptr<TimestampGroupedEdges>& rhs) const;
};

#endif //TIMESTAMPGROUPEDEDGES_H
