#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <vector>
#include <time.h>
#include <queue>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/package.h>

#include <sensor_msgs/PointCloud2.h>
#include "sensor_msgs/Image.h"
#include "sensor_msgs/LaserScan.h"
#include "geometry_msgs/Point.h"
#include "gazebo_msgs/ModelStates.h"
#include "std_msgs/Int32.h"
#include "std_msgs/Float64MultiArray.h"
#include "std_msgs/Bool.h"
#include "social_navigation/GlobalPathRequest.h"
#include "social_navigation/GlobalPathResponse.h"

#include <image_geometry/pinhole_camera_model.h> 

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "RRT.h"

using namespace std;
using namespace cv;

const double INF = 1e9;
const double EPS = 1e-6;

class GlobalPathGenerator{
    private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_jackal_;
    ros::Subscriber sub_goal_;
    
    stringstream pkg_path_;
    string free_map_file_;
    string cost_map_file_;
    int img_w_;
    int img_h_;

    // jackal status
    double jx_; // jackal x
    double jy_; // jackal y
    double jt_; // jackal yaw
        
    // RRT
    point jackal_;
    point global_goal_;
    int n_path_;
    bool lock_;
    
    RRT rrt;

    public:
    GlobalPathGenerator(){
        // set random seed
        srand(time(NULL));
        
        // load cost map
        pkg_path_ << ros::package::getPath("social_navigation") << "/";
        cost_map_file_ = pkg_path_.str() + "config/costmap_301_1f.png";

        n_path_ = 5;
        lock_ = false;

        // load RRT module
        rrt = RRT(cost_map_file_);
        rrt.collision_threshold_ = 0.95;
        rrt.time_limit_ = 10.0;

        pub_ = nh_.advertise<social_navigation::GlobalPathResponse>("/global_path/response", 1000);
        sub_goal_ = nh_.subscribe("/global_path/request", 1, &GlobalPathGenerator::callback_goal, this);
        // sub_jackal_ = nh_.subscribe("/gazebo/model_states", 1, &GlobalPathGenerator::callback_jackal, this);
        
        sleep(1.0);
        cout << "initialize rrt" << endl;
    }

    void callback_goal(const social_navigation::GlobalPathRequest::ConstPtr& msg){
        cout << "callback goal" << endl;
        cout << msg->id << endl;
        if(msg->type == 2 && lock_) return;
        if(msg->type == 1){
            cout << "type 1 start" << endl;
            while(1){
                if(!lock_) break;
            }
            cout << "type 1 end" << endl;
        }
        lock_ = true;
        // execute service
        rrt.reset();
        point root = point(msg->root.x, msg->root.y);
        point goal = point(msg->goal.x, msg->goal.y);

        // vector<point> path = rrt.rrt_star(root, goal);
        // social_navigation::GlobalPathResponse rt;
        // vector<geometry_msgs::Point> pts;
        // double rt_dist = 1000.0;
        // int sz = path.size();
        // double d = 0.0;
        // if(sz>0) {
        //     d = dist(path[0], goal);
        //     geometry_msgs::Point pt;
        //     pt.x = goal.x;
        //     pt.y = goal.y;
        //     pt.z = 0.0;
        //     pts.push_back(pt);
        // }
        // for(int i = 0; i<sz-1;i++){
        //     geometry_msgs::Point pt;
        //     pt.x = path[i].x;
        //     pt.y = path[i].y;
        //     pt.z = d;
        //     pts.push_back(pt);
        //     d += dist(path[i], path[i+1]);
        // }
        // rt_dist = min(rt_dist, d);
        // rt.points = pts;
        // rt.n_points = pts.size();
        // rt.id = msg->id;
        // rt.type = msg->type;
        // rt.distance = rt_dist;
        // pub_.publish(rt);
        
        vector<vector<point>> paths = rrt.diverse_rrt(root, goal, msg->n_path);

        social_navigation::GlobalPathResponse rt;
        vector<geometry_msgs::Point> pts;
        double rt_dist = 1000.0;
        for(const vector<point>& path: paths){
            int sz = path.size();
            double d = 0.0;
            if(sz>0) {
                d = dist(path[0], goal);
                geometry_msgs::Point pt;
                pt.x = goal.x;
                pt.y = goal.y;
                pt.z = 0.0;
                pts.push_back(pt);
            }
            for(int i = 0; i<sz-1;i++){
                geometry_msgs::Point pt;
                pt.x = path[i].x;
                pt.y = path[i].y;
                pt.z = d;
                pts.push_back(pt);
                d += dist(path[i], path[i+1]);
            }
            rt_dist = min(rt_dist, d);
        }
        rt.points = pts;
        rt.n_points = pts.size();
        rt.id = msg->id;
        rt.type = msg->type;
        rt.distance = rt_dist;
        pub_.publish(rt);
        
        lock_ = false;
    }

};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "global_path_generator");
  GlobalPathGenerator global_planner = GlobalPathGenerator();
  ros::spin();
}