#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include<unordered_map>
#include "Node.h"
#include "TemporalEdge.h"

class RandomPicker;

class TemporalGraph {
public:
    std::unordered_map<int, std::shared_ptr<Node>> nodes;
    std::map<int64_t, std::vector<std::shared_ptr<TemporalEdge>>> edges;

    void add_node(int id);
    Node* get_node(int id);
    Node* get_or_create_node(int id);
    Node* get_random_node(RandomPicker* random_picker, bool begin_from_end);
    void add_edge(int id1, int id2, int64_t timestamp);
    void delete_edges_less_than_time(int64_t timestamp);
    [[nodiscard]] size_t get_node_count() const;
    [[nodiscard]] size_t get_edge_count() const;
    std::vector<int> get_node_ids();
};

#endif //TEMPORALGRAPH_H
