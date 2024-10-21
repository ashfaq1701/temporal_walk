#ifndef TEMPORALEDGE_H
#define TEMPORALEDGE_H

#include "Node.h"

class Node;

class TemporalEdge {
public:
    Node* u;
    Node* v;
    int64_t timestamp;

    TemporalEdge(Node* u, Node* v, int64_t ts);
};

#endif //TEMPORALEDGE_H
