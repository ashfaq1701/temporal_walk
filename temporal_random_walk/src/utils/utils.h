#ifndef UTILS_H
#define UTILS_H
#include <random>
#include <algorithm>
#include <map>
#include <boost/math/distributions/beta.hpp>
#include "../data/structs.cuh"
#include "../data/common_vector.cuh"

thread_local static std::mt19937 thread_local_gen{std::random_device{}()};

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

inline float compute_beta_95th_percentile(size_t successes, size_t failures) {
    if (successes == 0 && failures == 0) {
        return 1.0f; // No walks processed yet, assume full success
    }

    boost::math::beta_distribution<float> beta_dist(1.0f + successes, 1.0f + failures);
    return boost::math::quantile(beta_dist, 0.95f); // 95th percentile
}

template<GPUUsageMode GPUUsage>
CommonVector<int, GPUUsage> repeat_elements(const CommonVector<int, GPUUsage>& arr, int times) {
    CommonVector<int, GPUUsage> repeated_items;
    repeated_items.allocate(arr.size() * times);

    for (const auto& item : arr) {
        for (int i = 0; i < times; ++i) {
            repeated_items.push_back(item);
        }
    }

    return repeated_items;
}

template <typename T, GPUUsageMode GPUUsage>
DividedVector<T, GPUUsage> divide_vector(
    const CommonVector<T, GPUUsage>& input,
    int n)
{
    return DividedVector<T, GPUUsage>(input, n);
}

template <GPUUsageMode GPUUsage>
CommonVector<int, GPUUsage> divide_number(int n, int i) {
    CommonVector<int, GPUUsage> parts(i);
    parts.fill(n / i);

    const int remainder = n % i;

    for (int j = 0; j < remainder; ++j) {
        ++parts[j];
    }

    return parts;
}

template <typename T, GPUUsageMode GPUUsage>
void shuffle_vector(CommonVector<T, GPUUsage>& vec) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec.begin(), vec.end(), rng);
}

template <typename T>
T generate_random_value(T start, T end) {
    std::uniform_real_distribution<T> dist(start, end);
    return dist(thread_local_gen);
}

inline int generate_random_int(const int start, const int end) {
    std::uniform_int_distribution<> dist(start, end);
    return dist(thread_local_gen);
}

inline int generate_random_number_bounded_by(const int max_bound) {
    return generate_random_int(0, max_bound - 1);
}

inline bool generate_random_boolean() {
    return generate_random_int(0, 1) == 1;
}

inline int pick_random_number(const int a, const int b) {
    return generate_random_boolean() ? a : b;
}

inline int pick_other_number(const std::tuple<int, int>& number, const int picked_number) {
    const int first = std::get<0>(number);
    const int second = std::get<1>(number);
    return (picked_number == first) ? second : first;
}

#endif //UTILS_H
