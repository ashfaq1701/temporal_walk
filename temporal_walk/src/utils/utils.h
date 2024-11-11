#ifndef UTILS_H
#define UTILS_H
#include <random>

template <typename T>
size_t countKeysLessThan(const std::map<int64_t, T>& inputMap, int64_t key) {
    auto it = inputMap.lower_bound(key);
    return std::distance(inputMap.begin(), it);
}

template <typename T>
size_t countKeysGreaterThan(const std::map<int64_t, T>& inputMap, int64_t key) {
    auto it = inputMap.upper_bound(key);
    return std::distance(it, inputMap.end());
}


inline int get_random_number(const int max_bound) {
    std::random_device rd;
    std::mt19937 gen(rd());

    std::uniform_int_distribution<> dist(0, max_bound - 1);
    return dist(gen);
}

inline bool get_random_boolean() {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_int_distribution<int> dist(0, 1);
    return dist(gen) == 1;
}

template <typename K, typename V>
void delete_items_less_than_key(std::map<K, V>& map_obj, const K& key) {
    const auto it = map_obj.lower_bound(key);
    map_obj.erase(map_obj.begin(), it);
}

#endif //UTILS_H
