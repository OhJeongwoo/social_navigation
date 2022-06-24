#include <iostream>
#include "point.h"

using namespace std;

point::point(){this->x = 0; this->y = 0; this->z = 0;}
point::point(double x, double y){this->x = x; this->y = y; this->z = 0;}
point::point(double x, double y, double z){this->x = x; this->y = y; this->z = z;}

void point::print(){cout << "(" << x << ", " << y << ")" << endl;}

point point::operator+(const point &p){return point(x+p.x, y+p.y, z+p.z);}
point point::operator-(const point &p){return point(x-p.x, y-p.y, z-p.z);}
double point::operator*(const point &p){return x*p.x+y*p.y+z*p.z;}
point point::operator*(double alpha){return point(alpha*x, alpha*y);}
