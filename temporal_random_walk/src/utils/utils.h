#ifndef UTILS_H
#define UTILS_H

#include <random>
#include <algorithm>
#include <map>
#include "../data/structs.cuh"
#include "../cuda_common/types.cuh"

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

template<GPUUsageMode GPUUsage>
typename SelectVectorType<int, GPUUsage>::type repeat_elements(
    const typename SelectVectorType<int, GPUUsage>::type& arr,
    int times) {
    typename SelectVectorType<int, GPUUsage>::type repeated_items;
    repeated_items.reserve(arr.size() * times);

    for (const auto& item : arr) {
        for (int i = 0; i < times; ++i) {
            repeated_items.push_back(item);
        }
    }

    return repeated_items;
}

template <typename T, GPUUsageMode GPUUsage>
DividedVector<T, GPUUsage> divide_vector(
    const typename SelectVectorType<T, GPUUsage>::type& input,
    int n)
{
    return DividedVector<T, GPUUsage>(input, n);
}

template <GPUUsageMode GPUUsage>
typename SelectVectorType<int, GPUUsage>::type divide_number(int n, int i) {
    typename SelectVectorType<int, GPUUsage>::type parts(i);
    std::fill(parts.begin(), parts.end(), n / i);

    const int remainder = n % i;

    for (int j = 0; j < remainder; ++j) {
        ++parts[j];
    }

    return parts;
}

inline int pick_other_number(const std::tuple<int, int>& number, const int picked_number) {
    const int first = std::get<0>(number);
    const int second = std::get<1>(number);
    return (picked_number == first) ? second : first;
}

#endif //UTILS_H
