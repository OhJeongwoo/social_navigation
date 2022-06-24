#include <iostream>
#include "Transform.h"

using namespace std;

Transform::Transform(){sx = 0.05;sy = -0.05;cx = -59.4;cy = 30.0;}

pixel Transform::xy2pixel(point p){
    int px = int((p.y-cy)/sy);
    int py = int((p.x-cx)/sx);
    return pixel(px, py);
}

point Transform::pixel2xy(pixel p){
    double x = p.y * sy + cy;
    double y = p.x * sx + cx;
    return point(x,y);
}
