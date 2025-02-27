#ifndef INDEXBASEDRANDOMPICKER_H
#define INDEXBASEDRANDOMPICKER_H

#include "RandomPicker.h"

template<GPUUsageMode GPUUsage>
class IndexBasedRandomPicker : public RandomPicker<GPUUsage> {

public:
    ~IndexBasedRandomPicker() override = default;

    virtual int pick_random(int start, int end, bool prioritize_end)
    {
        #ifdef HAS_CUDA
        if (GPUUsage == GPUUsageMode::ON_GPU)
        {
            return this->pick_random_device(start, end, prioritize_end);
        }
        else
        #endif
        {
            return this->pick_random_host(start, end, prioritize_end);
        }
    }

    virtual HOST int pick_random_host(int start, int end, bool prioritize_end) { return -1; }

    #ifdef HAS_CUDA
    virtual DEVICE int pick_random_device(int start, int end, bool prioritize_end) { return -1; };
    #endif

    int get_picker_type() override
    {
        return INDEX_BASED_PICKER_TYPE;
    }
};

#endif //INDEXBASEDRANDOMPICKER_H
