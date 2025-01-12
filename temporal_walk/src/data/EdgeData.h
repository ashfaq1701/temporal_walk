#ifndef EDGEDATA_H
#define EDGEDATA_H

#include <vector>
#include <cstdint>
#include <tuple>

struct EdgeData {
    std::vector<int> sources;
    std::vector<int> targets;
    std::vector<int64_t> timestamps;

    void reserve(size_t size);
    void clear();
    size_t size() const;
    bool empty() const;
    void resize(size_t new_size);
    void push_back(int src, int tgt, int64_t ts);
};

#endif //EDGEDATA_H
