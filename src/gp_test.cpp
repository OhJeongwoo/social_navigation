#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <vector>
#include <time.h>
#include <queue>
#include <math.h>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/package.h>



#include <image_geometry/pinhole_camera_model.h> 

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "GPR.h"

using namespace std;
using namespace cv;

const double INF = 1e9;
const double EPS = 1e-6;
int main(int argc,char** argv){
  ros::init(argc, argv, "gp_test");
  cout << "start" << endl;
  GPR gp = GPR();
  // gp.load_dataset("/home/jeongwoooh/catkin_social/src/social_navigation/config/Qr.txt");
  gp.load_dataset("/home/jeongwoooh/catkin_social/src/social_navigation/config/Qc.txt");
  
  gp.make_dataset();
  gp.prebuild_GPR();
  
  double error = 0.0;
  vector<double> x;
  vector<double> y;
  clock_t begin = clock();
  for(int i = 0; i < gp.n_test_ - 10; i++){
    MatrixXd X = gp.X_test_.row(i);
    MatrixXd mu = gp.inference(X);
    error += (mu(0,0) - gp.Y_test_(i,0))*(mu(0,0) - gp.Y_test_(i,0));
    x.push_back(gp.Y_test_(i,0));
    y.push_back(mu(0,0));
  }
  clock_t end = clock();
  cout << "avg inference time: " << double(end-begin) / CLOCKS_PER_SEC / gp.n_test_ << endl;
  cout << sqrt(error / gp.n_test_) << endl;
  cout << "[";
  for(int i = 0; i < 100; i++){
    cout << x[i] << ", ";
  }
  cout << "]" << endl;
  cout << "[";
  for(int i = 0; i < 100; i++){
    cout << y[i] << ", ";
  }
  cout << "]" << endl;
}