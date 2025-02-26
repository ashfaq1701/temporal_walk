#ifndef RANDOMPICKER_H
#define RANDOMPICKER_H

constexpr int INDEX_BASED_PICKER_TYPE = 1;
constexpr int WEIGHT_BASED_PICKER_TYPE = 2;

class RandomPicker
{

public:
    virtual ~RandomPicker() = default;

    virtual int get_picker_type() = 0;
};

#endif //RANDOMPICKER_H
