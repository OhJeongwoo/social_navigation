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
#include "geometry_msgs/Point.h"
#include "std_msgs/Int32.h"
#include "social_navigation/RRT.h"

#include <image_geometry/pinhole_camera_model.h> 

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

typedef pair<double,double> point;
typedef pair<int, int> pixel;

struct node{
    point p;
    int parent = -1;
    vector<int> childs;
    bool is_path = false;
    node(): p(point(0,0)) {}
    node(point p): p(p) {}  
};

double get_dist(point a, point b){
    double dx = a.first - b.first;
    double dy = a.second - b.second;
    return sqrt(dx * dx + dy * dy);
}

int find_nearest(point p, vector<node>& nodes){
    if(nodes.size() == 0) return -1;
    int rt = 0;
    double min_dist = get_dist(p, nodes[0].p);
    for(int i = 1; i < nodes.size(); i++){
        double d = get_dist(p, nodes[i].p);
        if(d < min_dist){
            min_dist = d;
            rt = i;
        }
    }
    return rt;
}

point get_candidate(point p, point q, double t){
    // return (q - p) * t / ||q-p|| + p
    double d = get_dist(p, q);
    return point(p.first + (q.first - p.first) * t / d, p.second + (q.second - p.second) * t / d);
}


class RRT{
    private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Subscriber sub_;

    point root_;
    point goal_;
    int step_;
    int max_sample_;
    int lookahead_;
    double distance_threshold_;
    double tau_;

    double x_min_;
    double x_max_;
    double y_min_;
    double y_max_;

    double sx_;
    double sy_;
    double cx_;
    double cy_;
    int img_w_;
    int img_h_;
    vector<vector<bool>> collision_map_;
    stringstream pkg_path_;
    string collision_file_;
    string image_file_;
    Mat local_map_;
    const char delimiter =  ' ';

    clock_t init_time_;
    clock_t last_pub_time_;
    clock_t cur_time_;
    double control_interval_;

    bool draw_;

    public:
    RRT(){
        //define hyperparameter
        pkg_path_ << ros::package::getPath("social_navigation") << "/";
        collision_file_ = pkg_path_.str() + "config/collision_301_1f.txt";
        image_file_ = pkg_path_.str() + "config/free_space_301_1f.png";
        local_map_ = imread(image_file_, CV_8UC1);
        load_collision_map();
        
        sy_ = -2.517*0.02;
        sx_ = 2.494*0.02;
        cy_ = 30.199;
        cx_ = -59.361;
        root_ = point(35, 30);
        goal_ = point(12, 88);

        step_ = 10;
        max_sample_ = 10000;
        distance_threshold_ = 1.0;
        tau_ = 0.2;
        lookahead_ = 10;

        draw_ = true;
        cout << "complete to initialize" << endl;

        pub_ = nh_.advertise<geometry_msgs::Point>("/local_goal", 1000);
        sub_ = nh_.subscribe("/rrt", 1, &RRT::callback, this);
    }

    void callback(const social_navigation::RRT::ConstPtr& msg){
        root_ =  point(msg->root.x, msg->root.y);
        goal_ = point(msg->goal.x, msg->goal.y);

        cout << "root: " << root_.first << ", " << root_.second << endl;
        cout << "goal: " << goal_.first << ", " << goal_.second << endl;
        point local_goal = rrt();
        geometry_msgs::Point rt;
        rt.x = local_goal.first;
        rt.y = local_goal.second;
        rt.z = 0.0;
        pub_.publish(rt);
    }

    void load_collision_map(){
        string in_line;
        ifstream in(collision_file_);
        int line_num = 0;
        int N = -1;
        while(getline(in, in_line)){
            stringstream ss(in_line);
            string token;
            vector<string> tokens;
            line_num ++;
            while(getline(ss,token,delimiter)) tokens.push_back(token);
            if (line_num == 1) {
                img_w_ = stoi(tokens[0]);
                img_h_ = stoi(tokens[1]);
                N = stoi(tokens[2]);
                collision_map_ = vector<vector<bool>>(img_w_, vector<bool>(img_h_, true));
            }
            else if (line_num <= N + 1){
                int px = stoi(tokens[0]);
                int py = stoi(tokens[1]);
                collision_map_[px][py] = false;
            }    
        }
    }

    pixel get_pixel(point p){
        int px = int((p.second-cy_)/sy_);
        int py = int((p.first-cx_)/sx_);
        return pixel(px, py);
    }

    point interpolate(point a, point b, double lambda){
        return point(a.first * lambda + b.first * (1 - lambda), a.second * lambda + b.second * (1 - lambda));
    }

    bool is_collision(point a, point b){
        for(int i = 0; i <= step_; i++){
            pixel p = get_pixel(interpolate(a, b, 1.0 * i / step_));
            if(p.first < 0 || p.first >= img_w_ || p.second < 0 || p.second >= img_h_) return true;
            if(collision_map_[p.first][p.second]) return true;
        }
        return false;
    }

    point rrt(){
        clock_t start = clock();
        vector<node> pts;
        point goal = goal_;
        point root = root_;
        x_min_ = min<double>(goal.first, root.first) - 5.0;
        x_max_ = max<double>(goal.first, root.first) + 5.0;
        y_min_ = min<double>(goal.second, root.second) - 5.0;
        y_max_ = max<double>(goal.second, root.second) + 5.0;
        pts.push_back(node(root));
        int n_sample = 1;
        int nearest_index = 0;
        double dist = get_dist(root, goal);
        int i = 0;
        while(true){
            i ++;
            if(i>10000) break;
            double x = x_min_ + (x_max_ - x_min_) * ((double)rand() / (double) RAND_MAX);
            double y = y_min_ + (y_max_ - y_min_) * ((double)rand() / (double) RAND_MAX);
            point p = point(x,y);
            int idx = find_nearest(p, pts);
            p = get_candidate(pts[idx].p, p, tau_);
            if(is_collision(pts[idx].p, p)) continue;
            node nnode = node(p);
            nnode.parent = idx;
            pts.push_back(nnode);
            pts[idx].childs.push_back(n_sample);
            if(dist > get_dist(p, goal)){
                dist = get_dist(p, goal);
                nearest_index = n_sample;
            }
            n_sample ++;
            if(dist < distance_threshold_ || n_sample > max_sample_) break;
        }
        vector<int> path;
        while(1){
            path.push_back(nearest_index);
            pts[nearest_index].is_path = true;
            if(pts[nearest_index].parent == -1) break;
            nearest_index = pts[nearest_index].parent;
        }
        clock_t end = clock();
        cout << (double)(end-start) / CLOCKS_PER_SEC << endl;
        cout << "drawing" << endl;
        if(draw_){
            draw(pts, goal);
        }
        if(path.size() < lookahead_) return pts[path[0]].p;
        return pts[path[path.size() - lookahead_]].p;
    }

    void draw(const vector<node>& pts, point goal){
        // local_map_ = Mat(img_h_, img_w_, CV_8UC3, Scalar(255,255,255));
        // local_map_ = Mat(img_h_, img_w_, CV_8UC3, Scalar(255,255,255));
        local_map_ = imread(image_file_, CV_8UC1);
        string local_map_save_path = pkg_path_.str() + "test.png";
        

        for(const node& p: pts){
            if(p.parent == -1) continue;
            pixel cur = get_pixel(p.p);
            pixel par = get_pixel(pts[p.parent].p);
            if(p.is_path) cv::line(local_map_, Point(cur.second, cur.first), Point(par.second, par.first), Scalar(0), 5);
            else cv::line(local_map_, Point(cur.second, cur.first), Point(par.second, par.first), Scalar(140), 2);
        }
        
        pixel root = get_pixel(pts[0].p);
        cv::circle(local_map_,  Point(root.second, root.first), 10.0, Scalar(80), -1);

        pixel pgoal = get_pixel(goal);
        cv::circle(local_map_,  Point(pgoal.second, pgoal.first), 10.0, Scalar(40), -1);

        cv::imwrite(local_map_save_path, local_map_);
    }



    void loop(){
        while(1){
            if(double(clock() - last_pub_time_) / CLOCKS_PER_SEC < control_interval_){
                continue;
            }
            rrt();
            last_pub_time_ = clock();
        }
    }

};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "rrt");
  RRT rrt = RRT();
  ros::spin();
}

