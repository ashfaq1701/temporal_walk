#ifndef RANDOMPICKER_H
#define RANDOMPICKER_H

#include "../utils/utils.h"

template<bool UseGPU>
class RandomPicker
{

public:
    virtual ~RandomPicker() = default;
};

#endif //RANDOMPICKER_H
