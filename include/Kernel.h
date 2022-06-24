#ifndef KERNEL_H
#define KERNEL_H
#include <iostream>
#include <vector>
#include "point.h"
#include "pixel.h"
#include "utils.h"


using namespace std;

class Kernel{
    public:
    // M*exp(-alpha * d^2)
    double M;
    double alpha;
    int half;
    int size;
    point center;
    vector<vector<double>> kernel;

    Kernel();
    Kernel(double M, double alpha, int half);
    double get(double d);
};

#endif