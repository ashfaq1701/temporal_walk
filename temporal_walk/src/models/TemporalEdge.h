#ifndef TEMPORALEDGE_H
#define TEMPORALEDGE_H

#include <cstdint>
#include "Node.h"

class Node;

class TemporalEdge {
public:
    Node* u;
    Node* i;
    int64_t timestamp;

    TemporalEdge(Node* u, Node* i, int64_t ts);

    [[nodiscard]] Node* pick_random_endpoint() const;

    [[nodiscard]] Node* select_other_endpoint(const Node* node) const;
};

#endif //TEMPORALEDGE_H
