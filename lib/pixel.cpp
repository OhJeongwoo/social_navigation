#include <iostream>
#include "pixel.h"

using namespace std;

pixel::pixel(): x(0), y(0) {}
pixel::pixel(int x, int y): x(x), y(y) {}
pixel pixel::operator+(const pixel &p){return pixel(x+p.x, y+p.y);}
pixel pixel::operator-(const pixel &p){return pixel(x-p.x, y-p.y);}