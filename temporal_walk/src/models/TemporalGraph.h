#ifndef TEMPORALGRAPH_H
#define TEMPORALGRAPH_H

#include <memory>
#include <cstddef>
#include<unordered_map>
#include "Node.h"
#include "TimestampGroupedEdges.h"

class RandomPicker;

class TemporalGraph {
public:
    std::unordered_map<int, std::shared_ptr<Node>> nodes;
    std::vector<std::shared_ptr<TimestampGroupedEdges>> edges;
    std::map<int64_t, std::shared_ptr<TimestampGroupedEdges>> edge_index;

    void add_node(int id);
    Node* get_node(int id);
    Node* get_or_create_node(int id);
    Node* get_random_node(RandomPicker* random_picker, bool begin_from_end);
    void add_edge(int id1, int id2, int64_t timestamp);
    std::vector<TemporalEdge*> get_edges() const;
    void sort_edges();
    void delete_edges_less_than_time(int64_t timestamp);
    [[nodiscard]] size_t get_node_count() const;
    [[nodiscard]] size_t get_edge_count() const;
    std::vector<int> get_node_ids();
};

#endif //TEMPORALGRAPH_H
