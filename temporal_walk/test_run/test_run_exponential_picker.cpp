#include <iostream>
#include "../src/random/ExponentialIndexRandomPicker.cuh"

constexpr int TOTAL_TIMESTEPS = 100000000;
constexpr GPUUsageMode GPU_USAGE_MODE = GPUUsageMode::ON_CPU;

int main()
{
    ExponentialIndexRandomPicker<GPU_USAGE_MODE> random_picker;

    std::cout << "Prioritizing end: " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        std::cout << random_picker.pick_random(0, TOTAL_TIMESTEPS, true) << std::endl;
    }

    std::cout << std::endl << std::endl;
    std::cout << "Prioritizing start: " << std::endl;
    std::cout << "---------------------------------------" << std::endl;
    for (int i = 0; i < 1000; i++)
    {
        std::cout << random_picker.pick_random(0, TOTAL_TIMESTEPS, false) << std::endl;
    }

    return 0;
}
