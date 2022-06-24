#ifndef PIXEL
#define PIXEL
#include <iostream>

using namespace std;

class pixel{
    public:
    int x,y;

    pixel();
    pixel(int x, int y);
    pixel operator+(const pixel &p);
    pixel operator-(const pixel &p);
};

#endif