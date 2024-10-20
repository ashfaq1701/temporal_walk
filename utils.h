#ifndef UTILS_H
#define UTILS_H

template <typename T>
std::vector<int64_t> getKeysLessThan(const std::map<int64_t, T>& inputMap, int64_t key) {
    std::vector<int64_t> keysLessThan;
    auto it = inputMap.lower_bound(key);

    keysLessThan.reserve(std::distance(inputMap.begin(), it));

    for (auto iter = inputMap.begin(); iter != it; ++iter) {
        keysLessThan.push_back(iter->first);
    }

    return keysLessThan;
}

#endif //UTILS_H
