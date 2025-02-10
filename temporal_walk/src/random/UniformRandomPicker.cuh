#ifndef UNIFORMRANDOMPICKER_H
#define UNIFORMRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"
#include "../utils/utils.h"

class UniformRandomPicker final : public IndexBasedRandomPicker {
public:
    bool use_gpu;

    explicit UniformRandomPicker(bool use_gpu);

    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //UNIFORMRANDOMPICKER_H
