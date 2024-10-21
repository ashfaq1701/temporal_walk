#ifndef UTILS_H
#define UTILS_H
#include <random>
#include <__random/random_device.h>

template <typename T>
size_t countKeysLessThan(const std::map<int64_t, T>& inputMap, int64_t key) {
    auto it = inputMap.lower_bound(key);
    return std::distance(inputMap.begin(), it);
}


inline int get_random_number(int max_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(0, max_bound - 1);
    return dist(gen);
}

#endif //UTILS_H
