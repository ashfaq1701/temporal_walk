#ifndef RANDOMPICKER_H
#define RANDOMPICKER_H

#include "../utils/utils.h"

constexpr int INDEX_BASED_PICKER_TYPE = 1;
constexpr int WEIGHT_BASED_PICKER_TYPE = 2;

class RandomPicker
{

public:
    virtual ~RandomPicker() = default;

    virtual int get_picker_type();
};

class InvalidRandomPicker final : public RandomPicker {};

#endif //RANDOMPICKER_H
