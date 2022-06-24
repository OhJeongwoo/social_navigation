#ifndef TRANSFORM_H
#define TRANSFORM_H
#include <iostream>
#include "point.h"
#include "pixel.h"

class Transform{
    public:
    double sx, sy, cx, cy;

    Transform();
    pixel xy2pixel(point p);
    point pixel2xy(pixel p);
};

#endif