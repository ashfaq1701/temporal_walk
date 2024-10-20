#ifndef RANDOMPICKER_H
#define RANDOMPICKER_H



class RandomPicker {
public:
    virtual int pick_random(int start, int end) = 0;
    virtual ~RandomPicker() = default;
};



#endif //RANDOMPICKER_H
