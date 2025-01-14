#ifndef UTILS_H
#define UTILS_H
#include <random>
#include <algorithm>
#include <map>
#include <boost/math/distributions/beta.hpp>

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

inline int get_random_number(const int max_bound) {
    std::uniform_int_distribution<> dist(0, max_bound - 1);
    return dist(thread_local_gen);
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

inline float compute_beta_95th_percentile(size_t successes, size_t failures) {
    if (successes == 0 && failures == 0) {
        return 1.0f; // No walks processed yet, assume full success
    }

    boost::math::beta_distribution<float> beta_dist(1.0f + successes, 1.0f + failures);
    return boost::math::quantile(beta_dist, 0.95f); // 95th percentile
}

inline std::vector<int> repeat_elements(const std::vector<int>& arr, int times) {
    std::vector<int> repeated_items;
    repeated_items.reserve(arr.size() * times);

    for (const auto& item : arr) {
        for (int i = 0; i < times; ++i) {
            repeated_items.push_back(item);
        }
    }

    return repeated_items;
}

inline std::vector<std::vector<int>> divide_vector(const std::vector<int>& input, int n) {
    std::vector<std::vector<int>> result(n);
    const int total_size = static_cast<int>(input.size());
    const int base_size = total_size / n;
    const int remainder = total_size % n;

    int start = 0;
    for (int i = 0; i < n; ++i) {
        const int current_size = base_size + (i < remainder ? 1 : 0);
        result[i].assign(input.begin() + start, input.begin() + start + current_size);
        start += current_size;
    }

    return result;
}

inline std::vector<int> divide_number(int n, int i) {
    std::vector<int> parts(i, n / i);
    const int remainder = n % i;

    for (int j = 0; j < remainder; ++j) {
        ++parts[j];
    }

    return parts;
}

template <typename T>
void shuffle_vector(std::vector<T>& vec) {
    std::random_device rd;
    std::mt19937 rng(rd());
    std::shuffle(vec.begin(), vec.end(), rng);
}

inline int pick_random_number(const int a, const int b) {
    return get_random_boolean() ? a : b;
}

inline int pick_other_number(const std::tuple<int, int>& number, int picked_number) {
    const int first = std::get<0>(number);
    const int second = std::get<1>(number);
    return (picked_number == first) ? second : first;
}

#endif //UTILS_H
