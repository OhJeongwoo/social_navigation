#include <iostream>
#include "Kernel.h"
#include "math.h"

using namespace std;

Kernel::Kernel() {}
Kernel::Kernel(double M, double alpha, int half){
    this->M = M;
    this->alpha = alpha;
    this->half = half;
    size = 2 * half + 1;
    
    for(int i = -half; i<= half; i++){
        vector<double> kernel_row;
        for(int j = -half; j<=half; j++) kernel_row.push_back(M*exp(-alpha*(i*i+j*j)));
        kernel.push_back(kernel_row);
    }
}

double Kernel::get(double d){return M*exp(-alpha*d*d);}