#ifndef UTILS_H
#define UTILS_H

#include <algorithm>
#include <map>
#include "../data/structs.cuh"
#include "../cuda_common/types.cuh"

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
