#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <vector>

#include <ros/ros.h>
#include <ros/time.h>
#include <ros/package.h>

#include "sensor_msgs/LaserScan.h"
#include "social_navigation/ClusterArray.h"
#include "geometry_msgs/Point.h"
#include "std_msgs/Int32.h"

#include "RRT.h"

using namespace std;

const int W = 1080;
const int H = 720;

class ClusteringModule{
  private:
  ros::NodeHandle nh_;
  ros::Publisher pub_;
  ros::Subscriber sub_;

  double distance_threshold_;
  int point_threshold_;



  public:
  ClusteringModule(){
    // define hyperparameter
    distance_threshold_ = 0.5;
    point_threshold_ = 5;

    // define topic communicator
    pub_ = nh_.advertise<social_navigation::ClusterArray>("/clusters", 1000);
    sub_ = nh_.subscribe("/scan", 1, &ClusteringModule::callback, this);

  }

  void callback(const sensor_msgs::LaserScan::ConstPtr& msg){
        social_navigation::ClusterArray rt;
        
        // subscribe lidar points
        vector<point> pts; 
        int n_points = msg->ranges.size();
        double w = msg->angle_min;
        double dw = msg->angle_increment;
        for(int k = 0; k < n_points; k++){
            if(isinf(msg->ranges[k])||isnan(msg->ranges[k])) continue;
            double a = w + dw * k;
            double x = msg->ranges[k] * cos(a);
            double y = msg->ranges[k] * sin(a);
            pts.push_back(point(x,y));
        }
        n_points = pts.size();
        
        // Euclidean clustering
        vector<vector<int>> adj;
        for(int i = 0; i < n_points; i++){
            vector<int> adj_row;
            for(int j = 0; j < n_points; j++){
                if(i==j) continue;
                if(dist(pts[i], pts[j]) < distance_threshold_) adj_row.push_back(j);
            }
        }

        vector<bool> visit(false, n_points);
        vector<social_navigation::Cluster> clusters;
        for(int i = 0; i < n_points; i++){
            if(visit[i]) continue;
            vector<geometry_msgs::Point> points;
            geometry_msgs::Point c;
            queue<int> q;
            q.push(i);
            while(1){
                if(q.empty()) break;
                int cur = q.front();
                q.pop();
                if(visit[cur]) continue;
                geometry_msgs::Point np;
                np.x = pts[cur].x;
                np.y = pts[cur].y;
                np.z = 0.0;
                points.push_back(np);
                c.x += pts[cur].x;
                c.y += pts[cur].y;
                visit[cur] = true;
                for(int nxt : adj[cur]){
                    if(visit[nxt]) continue;
                    q.push(nxt);
                }
            }
            if (points.size() < point_threshold_) continue;
            social_navigation::Cluster cluster;
            cluster.points = points;
            cluster.N = points.size();
            c.x /= points.size();
            c.y /= points.size();
            cluster.center = c;
            clusters.push_back(cluster);
        }
        rt.clusters = clusters;

        pub_.publish(rt);
    }
};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "euclidean_clustering");
  ClusteringModule clustering_module;
  ros::spin();
}

