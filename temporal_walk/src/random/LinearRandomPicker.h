#ifndef LINEARRANDOMPICKER_H
#define LINEARRANDOMPICKER_H

#include "IndexBasedRandomPicker.h"

class LinearRandomPicker final : public IndexBasedRandomPicker {
public:
    int pick_random(int start, int end, bool prioritize_end) override;
};

#endif //LINEARRANDOMPICKER_H
