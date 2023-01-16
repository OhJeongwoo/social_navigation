#ifndef GPR_H
#define GPR_H
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <vector>
#include <queue>

#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

class GPR{
    public:
    // hyperparameter
    double l_ = 5.0;
    double sigma_f_ = 5.0;
    double sigma_y_ = 1.0;
    int iput_dim_;
    

    // dataset
    int n_data_; // # of data
    int n_train_;
    int n_test_;
    vector<vector<double>> iput_list_;
    vector<double> oput_list_;

    // Gaussian Process
    MatrixXd X_train_;
    MatrixXd X_test_;
    MatrixXd Y_train_;
    MatrixXd Y_test_;
    MatrixXd K11_;
    MatrixXd K11_inv_;
    double y_mean_;

    // etc
    const char delimiter_ = ' ';

    GPR();
    void load_dataset(string file_name);
    void make_dataset();
    void prebuild_GPR();
    void set_hyperparameter(double l, double f, double y);
    MatrixXd rbf_kernel(const MatrixXd& X, const MatrixXd& Y);
    MatrixXd inference(const MatrixXd& X);
};


#endif