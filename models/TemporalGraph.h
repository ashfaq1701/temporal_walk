#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include<unordered_map>
#include "Node.h"
#include "TemporalEdge.h"

class TemporalGraph {
public:
    std::unordered_map<int, std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<TemporalEdge>> edges;

    void addNode(int id);
    Node* getNode(int id);
    void addEdge(int id1, int id2, int64_t timestamp);
    [[nodiscard]] size_t get_node_count() const;
    [[nodiscard]] size_t get_edge_count() const;
};



#endif //TEMPORALGRAPH_H
