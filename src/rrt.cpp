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
#include "social_navigation/RRT.h"
#include "social_navigation/RRTresponse.h"

#include <image_geometry/pinhole_camera_model.h> 

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

const double INF = 1e9;

typedef pair<double,double> point;
typedef pair<int, int> pixel;

struct node{
    point p;
    int parent = -1;
    vector<int> childs;
    bool is_path = false;
    double cost = 0.0;
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

vector<int> find_near(point p, vector<node>& nodes, double d){
    vector<int> rt;
    for(int i = 0; i < nodes.size(); i++){
        if(d > get_dist(p, nodes[i].p)) rt.push_back(i);
    }
    return rt;
}

int find_constraint_nearest(point p, vector<node>& nodes, double min_d){
    if(nodes.size() == 0) return -1;
    int rt = -1;
    double min_dist = INF;
    for(int i = 0; i < nodes.size(); i++){
        double d = get_dist(p, nodes[i].p);
        if(d < min_d) continue;
        if(d < min_dist){
            min_dist = d;
            rt = i;
        }
    }
    if(rt == -1) cout << "error" << endl;
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
    ros::Subscriber sub_lidar_;
    ros::Subscriber sub_jackal_;

    point root_;
    point goal_;
    int step_;
    int max_sample_;
    double distance_threshold_;
    double tau_;
    double coeff_tau_;
    vector<node> local_path_;
    int path_length_;
    double lambda_;
    double collision_threshold_;
    bool stop_;

    double x_min_;
    double x_max_;
    double y_min_;
    double y_max_;

    double jx_;
    double jy_;
    double jt_;

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
    string costmap_file_;
    Mat local_map_;
    Mat cost_map_;
    Mat local_cost_map_;
    int skip_angle_;
    int kernel_block_;
    double kernel_step_;
    int kernel_half_;
    int kernel_size_;
    vector<vector<int>> kernel_;
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
        costmap_file_ = pkg_path_.str() + "config/costmap_301_1f.png";
        local_map_ = imread(image_file_, CV_8UC1);
        cost_map_ = imread(costmap_file_, CV_8UC1);
        img_w_ = cost_map_.rows;
        img_h_ = cost_map_.cols;
        local_cost_map_ = Mat(img_w_, img_h_, CV_8UC1, Scalar(0));
        
        kernel_block_ = 7;
        kernel_step_ = 0.05;
        kernel_half_ = 20;
        kernel_size_ = 2 * kernel_half_ + 1;
        build_kernel();
        
        // sy_ = -2.517*0.02;
        // sx_ = 2.494*0.02;
        // cy_ = 30.199;
        // cx_ = -59.361;
        sy_ = -0.05;
        sx_ = 0.05;
        cy_ = 30.0;
        cx_ = -59.4;
        root_ = point(35, 30);
        goal_ = point(12, 88);

        step_ = 10;
        max_sample_ = 10000;
        distance_threshold_ = 1.0;
        tau_ = 0.2;
        coeff_tau_ = 5.0;
        path_length_ = 50;
        lambda_ = 0.3;
        collision_threshold_ = 0.8;
        stop_ = false;

        draw_ = true;
        cout << "complete to initialize" << endl;

        pub_ = nh_.advertise<social_navigation::RRTresponse>("/local_goal", 1000);
        sub_lidar_ = nh_.subscribe("/front/scan", 1, &RRT::callback_lidar, this);
        sub_jackal_ = nh_.subscribe("/gazebo/model_states", 1, &RRT::callback_jackal, this);
        sleep(1.0);
        sub_ = nh_.subscribe("/rrt", 1, &RRT::callback, this);
    }

    void build_kernel(){
        for(int i = 0; i < kernel_size_; i++){
            cout << i<<endl;
            vector<int> kernel_row;
            for(int j = 0; j < kernel_size_; j++){
                int d = abs(i - kernel_half_) + abs(j - kernel_half_);
                if(d <= kernel_block_) kernel_row.push_back(255);
                else{
                    double x = 1 - (d - kernel_block_) * kernel_step_;
                    if(x<0) kernel_row.push_back(0);
                    else kernel_row.push_back(int(x*255));
                }
            }
            kernel_.push_back(kernel_row);
        }
    }

    void callback(const social_navigation::RRT::ConstPtr& msg){
        root_ =  point(msg->root.x, msg->root.y);
        goal_ = point(msg->goal.x, msg->goal.y);

        cout << "root: " << root_.first << ", " << root_.second << endl;
        cout << "goal: " << goal_.first << ", " << goal_.second << endl;
        cout << get_cost(root_) << endl;
        
        vector<point> path = rrt(msg->option);
        cout << path.size() << endl;
        social_navigation::RRTresponse rt;
        vector<geometry_msgs::Point> rt_data;
        for(point p : path) {
            geometry_msgs::Point x;
            x.x = p.first;
            x.y = p.second;
            x.z = 0.0;
            rt_data.push_back(x);
        }
        rt.path = rt_data;
        rt.stop = stop_;
        pub_.publish(rt);
    }


    void callback_lidar(const sensor_msgs::LaserScan::ConstPtr& msg){
        Mat local_cost_map = Mat(img_w_, img_h_, CV_8UC1, Scalar(0));
        double w = msg->angle_min;
        double dw = msg->angle_increment;
        int n = msg->ranges.size();
        for(int k = 0; k < n; k++){
            if(isinf(msg->ranges[k])||isnan(msg->ranges[k])) continue;
            double a = jt_ + w + dw * k;
            double x = jx_ + msg->ranges[k] * cos(a);
            double y = jy_ + msg->ranges[k] * sin(a);
            pixel p = get_pixel(point(x,y));
            for(int i = -kernel_half_; i <= kernel_half_; i++){
                for(int j = -kernel_half_; j <= kernel_half_; j++){
                    int px = p.first + i;
                    int py = p.second + j;
                    local_cost_map.at<uchar>(px,py) = max<uchar>(local_cost_map.at<uchar>(px,py), kernel_[i+kernel_half_][j+kernel_half_]);
                }
            }
        }
        local_cost_map_ = local_cost_map;
    }

    void callback_jackal(const gazebo_msgs::ModelStates::ConstPtr& msg){
        int idx = -1;
        for(int i = 0; i < msg->name.size(); i++){
            if(msg->name[i].compare("jackal") == 0) idx = i;
        }
        if(idx == -1) {
            cout << "Jackal does not exist!!" << endl;
            return;
        }
        geometry_msgs::Pose pos = msg->pose[idx];
        geometry_msgs::Quaternion q = pos.orientation;
        double qx = 1 - 2 * (q.y * q.y + q.z * q.z);
        double qy = 2 * (q.w * q.z + q.x * q.y);
        jx_ = pos.position.x;
        jy_ = pos.position.y;
        jt_ = atan2(qy, qx);
        return;
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

    double get_cost(point p){
        int px = int((p.second-cy_)/sy_);
        int py = int((p.first-cx_)/sx_);
        if(px < 0 || px >= img_w_ || py < 0 || py >= img_h_) return INF;
        return double(max(cost_map_.at<uchar>(px, py), local_cost_map_.at<uchar>(px,py))) / 255.0;
    }

    point interpolate(point a, point b, double lambda){
        return point(a.first * lambda + b.first * (1 - lambda), a.second * lambda + b.second * (1 - lambda));
    }

    double get_edge_cost(point a, point b){
        double d = get_dist(a, b);
        double c = 0.0;
        for(int i = 0; i <= step_; i++) c += get_cost(interpolate(a, b, 1.0 * i / step_));
        c /= step_ + 1;
        return (c + lambda_) * d;
    }

    bool is_collision(point a, point b){
        // for(int i = 0; i <= step_; i++){
        //     pixel p = get_pixel(interpolate(a, b, 1.0 * i / step_));
        //     if(p.first < 0 || p.first >= img_w_ || p.second < 0 || p.second >= img_h_) return true;
        //     if(collision_map_[p.first][p.second]) return true;
        // }
        // return false;
        double c = 0.0;
        for(int i = 0; i <= step_; i++) c += get_cost(interpolate(a, b, 1.0 * i / step_));
        c /= step_ + 1;
        // cout << c << endl;
        if(c > collision_threshold_) return true;
        return false;
    }

    vector<node> init_rrt(point root, bool option){
        // if option is true, remove fixed local path
        vector<node> rt;
        if(local_path_.size() == 0 || option){
            node r;
            r.p = root;
            r.parent = -1;
            rt.push_back(r);
            return rt;
        }
        int idx = find_constraint_nearest(root, local_path_, tau_);
        node r;
        r.p = root;
        r.parent = -1;
        rt.push_back(r);
        double d = 0.0;
        stop_ = false;
        for(int i = idx; i < local_path_.size(); i++){
            node r;
            r.p = local_path_[i].p;
            r.parent = i - idx;
            // d += get_dist(rt[i-idx].p, r.p);
            d += get_edge_cost(rt[i-idx].p, r.p);
            if(is_collision(rt[i-idx].p, r.p)) stop_ = true;
            r.cost = d;
            rt.push_back(r);
        }
        if(stop_) cout << "stop!!" << endl;
        return rt;
    }

    vector<point> rrt(bool option){
        clock_t start = clock();
        point goal = goal_;
        point root = root_;
        vector<node> pts = init_rrt(root, option);
        x_min_ = min<double>(goal.first, root.first) - 5.0;
        x_max_ = max<double>(goal.first, root.first) + 5.0;
        y_min_ = min<double>(goal.second, root.second) - 5.0;
        y_max_ = max<double>(goal.second, root.second) + 5.0;
        // pts.push_back(node(root));
        int n_sample = pts.size();
        // int nearest_index = 0;
        // double dist = get_dist(root, goal);
        int nearest_index = find_nearest(goal, pts);
        double dist = get_dist(pts[nearest_index].p, goal);
        int tot_step = 0;
        while(true){
            if(tot_step > max_sample_ * 2) break;
            tot_step ++;
            double x = x_min_ + (x_max_ - x_min_) * ((double)rand() / (double) RAND_MAX);
            double y = y_min_ + (y_max_ - y_min_) * ((double)rand() / (double) RAND_MAX);
            point p = point(x,y);
            int idx = find_nearest(p, pts);
            p = get_candidate(pts[idx].p, p, tau_);
            if(is_collision(pts[idx].p, p)) continue;
            vector<int> near_idx = find_near(p, pts, coeff_tau_ * tau_);
            node nnode = node(p);
            nnode.parent = idx;
            // nnode.cost = pts[idx].cost + get_dist(pts[idx].p, p);
            nnode.cost = pts[idx].cost + get_edge_cost(pts[idx].p, p);
            pts.push_back(nnode);
            pts[idx].childs.push_back(n_sample);
            for(int i = 0; i < near_idx.size(); i++){
                int n_idx = near_idx[i];
                if(n_idx == idx) continue;
                // if(pts[n_idx].cost > pts[n_sample].cost + get_dist(pts[n_idx].p, pts[n_sample].p)){
                if(pts[n_idx].cost > pts[n_sample].cost + get_edge_cost(pts[n_idx].p, pts[n_sample].p)){
                    int par = pts[n_idx].parent;
                    if(par != -1){
                        vector<int> n_child;
                        for(int k : pts[par].childs){
                            if(k == n_idx) continue;
                            n_child.push_back(k);
                        }
                        pts[par].childs = n_child;
                    }
                    pts[n_idx].parent = n_sample;
                    // double delta_cost = pts[n_sample].cost + get_dist(pts[n_idx].p, pts[n_sample].p) - pts[n_idx].cost;
                    double delta_cost = pts[n_sample].cost + get_edge_cost(pts[n_idx].p, pts[n_sample].p) - pts[n_idx].cost;
                    queue<int> q;
                    q.push(n_idx);
                    while(!q.empty()){
                        int k = q.front();
                        q.pop();
                        pts[k].cost += delta_cost;
                        for(int c: pts[k].childs) q.push(c);
                    }
                    pts[n_sample].childs.push_back(n_idx);
                }
            }
            if(dist > get_dist(p, goal)){
                dist = get_dist(p, goal);
                nearest_index = n_sample;
            }
            n_sample ++;
            if(dist < distance_threshold_ || n_sample > max_sample_) break;
        }
        vector<int> path;
        vector<point> rt;
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
        if(path.size() < path_length_) {
            vector<node> ntree;
            for(int i = path.size() - 1; i >= 0 ; i --) ntree.push_back(node(pts[path[i]].p));
            local_path_ = ntree;
        }
        else{
            vector<node> ntree;
            for(int i = path.size() - 1; i > path.size() - path_length_ ; i --) ntree.push_back(node(pts[path[i]].p));
            local_path_ = ntree;
        }
        for(int i = 0; i < path.size(); i++) rt.push_back(pts[path[path.size()-1-i]].p);
        return rt;
        // if(path.size() < lookahead_) return pts[path[0]].p;
        // return pts[path[path.size() - lookahead_]].p;
    }

    void draw(const vector<node>& pts, point goal){
        // local_map_ = Mat(img_h_, img_w_, CV_8UC3, Scalar(255,255,255));
        // local_map_ = Mat(img_h_, img_w_, CV_8UC3, Scalar(255,255,255));
        local_map_ = imread(costmap_file_, CV_8UC1);
        string local_map_save_path = pkg_path_.str() + "test.png";
        
        for(int i = 0; i < img_w_; i++){
            for(int j =0; j < img_h_; j++) local_map_.at<uchar>(i, j) = max(local_map_.at<uchar>(i, j), local_cost_map_.at<uchar>(i, j));
        }

        for(const node& p: pts){
            if(p.parent == -1) continue;
            pixel cur = get_pixel(p.p);
            pixel par = get_pixel(pts[p.parent].p);
            if(p.is_path) cv::line(local_map_, Point(cur.second, cur.first), Point(par.second, par.first), Scalar(255), 5);
            else cv::line(local_map_, Point(cur.second, cur.first), Point(par.second, par.first), Scalar(180), 2);
        }
        
        pixel root = get_pixel(pts[0].p);
        cv::circle(local_map_,  Point(root.second, root.first), 10.0, Scalar(280), -1);

        pixel pgoal = get_pixel(goal);
        cv::circle(local_map_,  Point(pgoal.second, pgoal.first), 10.0, Scalar(240), -1);

        cv::imwrite(local_map_save_path, local_map_);
        // cv::imwrite(local_cost_map_save_path, local_cost_map_);
    }


};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "rrt");
  RRT rrt = RRT();
  ros::spin();
}