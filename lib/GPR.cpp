#include <iostream>
#include "GPR.h"
#include "math.h"
#include <random>

using namespace std;
using namespace Eigen;

GPR::GPR(){
  l_ = 5.0;
  sigma_f_ = 5.0;
  sigma_y_ = 1.0;

  n_train_ = 100;
}

void GPR::load_dataset(string file_name){
    iput_list_.clear();
    oput_list_.clear();
    string in_line;
    ifstream in(file_name);
    int line_num = 0;
    while(getline(in, in_line)){
        stringstream ss(in_line);
        string token;
        vector<string> tokens;
        line_num ++;
        while(getline(ss,token,delimiter_)) tokens.push_back(token);
        if (line_num == 1) {
            n_data_ = stoi(tokens[0]);
            iput_dim_ = stoi(tokens[1]);
            cout << "# of data: " << n_data_ << endl;
            cout << "input dim: " << iput_dim_ << endl;
        }
        else if (line_num <= n_data_ + 1){
            vector<double> iput;
            double oput;
            for(int i = 0; i < iput_dim_; i++) iput.push_back(stod(tokens[i]));
            oput = stod(tokens[iput_dim_]);
            iput_list_.push_back(iput);
            oput_list_.push_back(oput);
        }    
    }
    cout << "complete to load dataset!" << endl;
}


void GPR::make_dataset(){
  mt19937 rng(time(0));
  int N = int(n_data_ * 0.9);
  n_test_ = n_data_ - N;
  uniform_int_distribution<uint32_t> dist(0, N - 1);
  X_train_ = MatrixXd::Zero(n_train_, iput_dim_);
  X_test_ = MatrixXd::Zero(n_test_, iput_dim_);
  Y_train_ = MatrixXd::Zero(n_train_, 1);
  Y_test_ = MatrixXd::Zero(n_test_, 1);
  
  double sum = 0.0;
  for(int i = 0; i < n_train_; i++){
    int idx = dist(rng);
    for(int j = 0; j < iput_dim_; j++) X_train_(i,j) = iput_list_[idx][j];
    Y_train_(i, 0) = oput_list_[idx];
    sum += oput_list_[idx];
  }
  y_mean_ = sum / n_train_;
  Y_train_ = Y_train_.array() - y_mean_;

  for(int i = 0; i < n_test_; i++){
    int idx = N + i;
    for(int j = 0; j < iput_dim_; j++) X_test_(i,j) = iput_list_[idx][j];
    Y_test_(i, 0) = oput_list_[idx];
  }
}


void GPR::prebuild_GPR(){
  K11_ = rbf_kernel(X_train_, X_train_) + sigma_y_ * sigma_y_ * MatrixXd::Identity(n_train_, n_train_);
  K11_inv_ = K11_.inverse();
}


void GPR::set_hyperparameter(double l, double f, double y){
    l_ = l;
    sigma_f_ = f;
    sigma_y_ = y;
}


MatrixXd GPR::rbf_kernel(const MatrixXd& X, const MatrixXd& Y) {
  int m = X.rows();
  int n = Y.rows();
  MatrixXd K(m, n);

  // Iterate over each pair of rows in X
  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      // Compute the squared Euclidean distance between the two rows
      VectorXd diff = X.row(i) - Y.row(j);
      double dist_sq = diff.dot(diff);

      // Compute the RBF kernel value with the given gamma and distance
      K(i, j) = sigma_f_ * sigma_f_ * std::exp(-1 / l_ / l_ * dist_sq);
    }
  }

  return K;
}


MatrixXd GPR::inference(const MatrixXd& X){
    MatrixXd K21 = rbf_kernel(X, X_train_);
    MatrixXd K12 = K21.transpose();
    MatrixXd K22 = rbf_kernel(X, X);
    MatrixXd mu = K21 * K11_inv_ * Y_train_;
    mu = mu.array() + y_mean_;
    return mu;
}