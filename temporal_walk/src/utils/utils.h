#ifndef UTILS_H
#define UTILS_H
#include <random>
#include <algorithm>

template <typename T>
size_t count_keys_less_than(const std::map<int64_t, T>& inputMap, int64_t key) {
    auto it = inputMap.lower_bound(key);
    return std::distance(inputMap.begin(), it);
}

template <typename T>
size_t count_keys_greater_than(const std::map<int64_t, T>& inputMap, int64_t key) {
    auto it = inputMap.upper_bound(key);
    return std::distance(it, inputMap.end());
}

template <typename T, typename V, typename Comp>
size_t count_elements_less_than(const std::vector<T>& vec, const V& value, Comp comp) {
    auto it = std::lower_bound(vec.begin(), vec.end(), value, comp);
    return std::distance(vec.begin(), it);
}

template <typename T, typename V, typename Comp>
size_t count_elements_greater_than(const std::vector<T>& vec, const V& value, Comp comp) {
    auto it = std::upper_bound(vec.begin(), vec.end(), value, comp);
    return std::distance(it, vec.end());
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

template<typename T, typename V, typename Comp>
void delete_items_less_than(std::vector<T>& vec, const V& value, Comp comp) {
    auto it = std::lower_bound(vec.begin(), vec.end(), value, comp);
    vec.erase(vec.begin(), it);
}

#endif //UTILS_H
