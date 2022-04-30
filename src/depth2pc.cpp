#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <vector>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Image.h"
#include "social_navigation/InstanceArray.h"
#include "social_navigation/InstanceImage.h"
#include "social_navigation/PedestrianArray.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/Int32.h"

#include <pcl/point_cloud.h>
#include <pcl_conversions/pcl_conversions.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_types.h>
#include <image_geometry/pinhole_camera_model.h> 

#include "image_transport/image_transport.h"
#include "cv_bridge/cv_bridge.h"
//#include <opencv/cv.h>
//#include <opencv/highgui.h>

using namespace std;

const int W = 1080;
const int H = 720;

double depth(double x, double y, double z){
  return sqrt(x*x + y*y + z*z);
}


typedef pair<double,double> pdd;
typedef pair<pdd, geometry_msgs::Point> pxdep; // {pixel, point}


bool compare(const geometry_msgs::Point &p, const geometry_msgs::Point &q){
  return depth(p.x, p.y, p.z) < depth(q.x, q.y, q.z);
}

class Depth2PC{
  private:
  ros::NodeHandle nh_;
  ros::Publisher pub_ped_;
  ros::Publisher pub_img_;
  ros::Subscriber sub_;
  ros::Subscriber sub_img_;
  ros::Subscriber sub_sig_;
  
  pcl::PointCloud<pcl::PointXYZ> cloud_;
  vector<pxdep> depth_map_;
  vector<geometry_msgs::Point> peds_;
  
  double outlier_threshold_ = 1.0;
  int k_ = 10;

  double fx = 642.303466796875;
  double fy = 641.542236328125;
  double cx = 643.4588012695312;
  double cy = 374.88818359375;
  double tx = -0.05911977216601372;
  double ty = -2.3358141334028915e-05;
  double min_x = -10.0;
  double max_x = 10.0;
  double min_z = -1.0;
  double max_z = 15.0;
  double resolution = 0.02;
  int W_ = int((max_x-min_x) / resolution);
  int H_ = int((max_z-min_z) / resolution);
  
  int seq_ = 0;
  bool debug_ = true;

  public:
  Depth2PC(){
    // define topic communicator
    pub_ped_ = nh_.advertise<social_navigation::PedestrianArray>("/ped", 1000);
    pub_img_ = nh_.advertise<sensor_msgs::Image>("/local_map", 1000);
    sub_ = nh_.subscribe("/camera/depth/color/points", 1, &Depth2PC::callback, this);
    sub_sig_ = nh_.subscribe("/signal", 1, &Depth2PC::callback_signal, this);
    sub_img_ = nh_.subscribe("/instances", 1, &Depth2PC::callback_img, this);
  }

  void callback_signal(const std_msgs::Int32ConstPtr& msg){
    seq_ = msg->data;
    
    vector<pxdep> depth_map;
    for(const auto& p: cloud_){
      double x = p.x;
      double y = p.y;
      double z = p.z;

      if(isnan(x) || isnan(y) || isnan(z)) continue;

      // project point to image
      double u = (fx * x + cx * z + tx) / z;
      double v = (fy * y + cy * z + ty) / z; 
      
      if(u < 0 || u >= W || v < 0 || v >= H) continue;
      
      pxdep tmp;
      geometry_msgs::Point pt;
      pt.x = x;
      pt.y = y;
      pt.z = z;
      tmp.first = pdd(u,v);
      tmp.second = pt;
      depth_map.push_back(tmp);
    }
    depth_map_ = depth_map;
  }

  void callback(const sensor_msgs::PointCloud2ConstPtr& msg){
    pcl::fromROSMsg(*msg, cloud_);
    return;
  }

  void callback_img(const social_navigation::InstanceArrayConstPtr& msg){
    clock_t start = clock();
    if(msg->seq != seq_) cout << "[WARN] There is some problems in time synchronization!" << endl;
    
    vector<pxdep> depth_map = depth_map_;
    vector<social_navigation::InstanceImage> instances = msg->instances;
    int w = msg->width;
    int h = msg->height;
    social_navigation::PedestrianArray rt;
    vector<geometry_msgs::Point> peds;

    int K = (2*k_+1)*(2*k_+1);

    for(int i = 0; i < instances.size(); i++){
      int n = 0;
      double x = 0;
      double y = 0;
      double z = 0;
      
      vector<vector<int>> dp;
      vector<int> dp_sub;

      for(int py = 0; py<h;py++){
        int o =0;
        for(int px=0;px<=k_;px++) o += instances[i].data[py * w + px];
        dp_sub.push_back(o);
      }

      for(int px =0;px<w;px++){
        vector<int> dp_tmp;
        int o =0;
        for(int py=0;py<=k_;py++){
          o += dp_sub[py];  
        }
        dp_tmp.push_back(o);
        for(int py=1;py<h;py++){
          if(py>=k_+1) o -= dp_sub[py-k_-1];
          if(py+k_<h) o += dp_sub[py+k_];
          dp_tmp.push_back(o);
        }
        dp.push_back(dp_tmp);
        for(int py=0;py<h;py++){
          if(px-k_>=0) dp_sub[py] -= instances[i].data[py * w + px - k_];
          if(px+k_+1<w) dp_sub[py] += instances[i].data[py * w + px + k_ + 1];
        }
      }
      vector<geometry_msgs::Point> pv;
      for(pxdep p : depth_map){
        int px = p.first.first;
        int py = p.first.second;
        int idx = py * w + px;

        if(idx < 0 || idx >= w * h) continue;
        if(dp[px][py] == K) {
          pv.push_back(p.second);
          // x += p.second.x;
          // y += p.second.y;
          // z += p.second.z;
          n++;
        }
      }
      sort(pv.begin(), pv.end(), compare);
      if(n==0) continue;
      for(int t = int(2*n/10); t<int(8*n/10);t++){
        x += pv[t].x;
        y += pv[t].y;
        z += pv[t].z;
      }
      x /= (int(8*n/10) - int(2*n/10));
      y /= (int(8*n/10) - int(2*n/10));
      z /= (int(8*n/10) - int(2*n/10));
      // x /= n;
      // y /= n;
      // z /= n;
      if(isnan(x) || isnan(y) || isnan(z)) continue;
      geometry_msgs::Point ped;
      ped.x = x;
      ped.y = y;
      ped.z = z;
      peds.push_back(ped);
    }
    peds_ = peds;
    rt.pedestrians = peds;
    pub_ped_.publish(rt);
    
    if(debug_) draw_result();
    clock_t end = clock();
    cout << "elapsed time: " << double(end-start) / CLOCKS_PER_SEC << endl;

  }

  void draw_result(){
    cv::Mat image = cv::Mat::zeros(H_,W_,CV_8UC3);

    vector<pxdep> depth_map = depth_map_;
    for(pxdep p : depth_map){
      double x = p.second.x;
      double z = p.second.z;
      if(x<min_x || x >max_x || z < min_z || z > max_z) continue;
      int px = int((x - min_x) / resolution);
      int py = int((z - min_z) / resolution);
      image.at<cv::Vec3b>(py,px)[0] = 255;
      image.at<cv::Vec3b>(py,px)[1] = 255;
      image.at<cv::Vec3b>(py,px)[2] = 255;      
    }
    for(geometry_msgs::Point p : peds_){
      double x = p.x;
      double z = p.z;
      if(x<min_x || x >max_x || z < min_z || z > max_z) continue;
      int px = int((x - min_x) / resolution);
      int py = int((z - min_z) / resolution);
      cv::circle(image, cv::Point(px, py), 30, CV_RGB(255,0,0), -1);     
    }
    cv_bridge::CvImage img_bridge;
    sensor_msgs::Image img_msg; // >> message to be sent

    std_msgs::Header header; // empty header
    header.seq = 1; // user defined counter
    header.stamp = ros::Time::now(); // time
    img_bridge = cv_bridge::CvImage(header, sensor_msgs::image_encodings::RGB8, image);
    img_bridge.toImageMsg(img_msg); // from cv_bridge to sensor_msgs::Image
    pub_img_.publish(img_msg); //
    
  }

};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "depth2pc");
  Depth2PC depth2pc;
  ros::spin();
}

