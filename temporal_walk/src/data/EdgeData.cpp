#include "EdgeData.h"


void EdgeData::reserve(size_t size) {
    sources.reserve(size);
    targets.reserve(size);
    timestamps.reserve(size);
}

void EdgeData::clear() {
    sources.clear();
    targets.clear();
    timestamps.clear();
}

size_t EdgeData::size() const {
    return timestamps.size();
}

bool EdgeData::empty() const {
    return timestamps.empty();
}

void EdgeData::resize(size_t new_size) {
    sources.resize(new_size);
    targets.resize(new_size);
    timestamps.resize(new_size);
}

void EdgeData::push_back(int src, int tgt, int64_t ts) {
    sources.push_back(src);
    targets.push_back(tgt);
    timestamps.push_back(ts);
}