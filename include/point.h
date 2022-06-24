#ifndef POINT
#define POINT
#include <iostream>

using namespace std;

class point{
    public:
    double x,y,z;

    point();
    point(double x, double y);
    point(double x, double y, double z);

    void print();

    point operator+(const point &p);
    point operator-(const point &p);
    double operator*(const point &p);
    point operator*(double alpha);
};

#endif