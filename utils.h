#ifndef UTILS_H
#define UTILS_H

template <typename T>
size_t countKeysLessThan(const std::map<int64_t, T>& inputMap, int64_t key) {
    auto it = inputMap.lower_bound(key);
    return std::distance(inputMap.begin(), it);
}

#endif //UTILS_H
