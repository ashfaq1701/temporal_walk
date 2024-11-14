#ifndef TIMESTAMPGROUPEDEDGES_H
#define TIMESTAMPGROUPEDEDGES_H
#include <cstdint>
#include <vector>
#include "TemporalEdge.h"


class TimestampGroupedEdges {
    int64_t timestamp;
    std::vector<std::shared_ptr<TemporalEdge>> edges;

public:

    bool operator<(const TimestampGroupedEdges& other) const;
    bool operator==(const TimestampGroupedEdges& other) const;

    bool operator<(int64_t ts) const;
    bool operator==(int64_t ts) const;

    explicit TimestampGroupedEdges(int64_t ts);

    [[nodiscard]] int64_t get_timestamp() const;

    void add_edge(const std::shared_ptr<TemporalEdge>& edge);

    [[nodiscard]] TemporalEdge* select_random_edge() const;

    [[nodiscard]] bool empty() const;

    [[nodiscard]] size_t size() const;
};

#endif //TIMESTAMPGROUPEDEDGES_H
