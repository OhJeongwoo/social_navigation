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
#include "social_navigation/TrajectoryPredict.h"
#include "social_navigation/Request.h"

#include <image_geometry/pinhole_camera_model.h> 

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

#include "RRT.h"

using namespace std;
using namespace cv;

const double INF = 1e9;
const double EPS = 1e-6;

class GlobalPlanner{
    private:
    ros::NodeHandle nh_;
    ros::Publisher pub_;
    ros::Publisher pub_flag_;
    ros::Subscriber sub_;
    ros::Subscriber sub_flag_;
    ros::Subscriber sub_signal_;
    ros::Subscriber sub_lidar_;
    ros::Subscriber sub_jackal_;
    ros::ServiceClient srv_pedestrian_;

    stringstream pkg_path_;
    string free_map_file_;
    string cost_map_file_;
    int img_w_;
    int img_h_;

    // jackal status
    double jx_; // jackal x
    double jy_; // jackal y
    double jt_; // jackal yaw
    
    // lidar info
    vector<point> lidar_points_;

    // RRT
    point local_goal_;
    point global_goal_;
    int n_cand_; // 5
    int lookahead_; // 25
    
    // MCTS
    double V_;
    double dt_;
    vector<point> actions_; //
    vector<Tnode> tree_; //
    int n_actions_; // 17
    int visit_threshold_; // 5
    int max_depth_; // 20
    double gamma_; // 0.95
    double alpha_visit_;
    double distance_threshold_;
    double time_limit_;

    vector<double> time_array_; // 0.0 0.5 ... 10.0
    bool ignore_;

    RRT rrt;

    public:
    GlobalPlanner(){
        // set random seed
        srand(time(NULL));
        
        // load cost map
        pkg_path_ << ros::package::getPath("social_navigation") << "/";
        cost_map_file_ = pkg_path_.str() + "config/costmap_301_1f.png";

        // load RRT module
        rrt = RRT(cost_map_file_);

        // set hyperparameter
        n_cand_ = 5;
        lookahead_ = 25; // local_goal_dist / tau
        V_ = 1.0;
        dt_ = 0.5;
        double sv = 0.5 * V_ * dt_;
        double fv = 1.0 * V_ * dt_;
        actions_.push_back(point(0.0, 0.0));
        for(int i = 0; i < 8; i++) actions_.push_back(point(sv * cos(2.0 * M_PI * i / 8), sv * sin(2.0 * M_PI * i / 8)));
        for(int i = 0; i < 8; i++) actions_.push_back(point(fv * cos(2.0 * M_PI * i / 8), fv * sin(2.0 * M_PI * i / 8)));
        n_actions_ = actions_.size();
        visit_threshold_ = 5;
        max_depth_ = 20;
        gamma_ = 0.95;
        alpha_visit_ = 1.0;
        distance_threshold_ = 1.0;
        time_limit_ = 1.0;
        ignore_ = false;

        for(int i = 0; i < max_depth_ + 1; i++) time_array_.push_back(dt_ * i);

        pub_ = nh_.advertise<geometry_msgs::Point>("/local_goal", 1000);
        sub_flag_ = nh_.subscribe("/request", 1, &GlobalPlanner::callback_request, this);
        sub_signal_ = nh_.subscribe("/flag", 1, &GlobalPlanner::callback_flag, this);
        sub_lidar_ = nh_.subscribe("/front/scan", 1, &GlobalPlanner::callback_lidar, this);
        sub_jackal_ = nh_.subscribe("/gazebo/model_states", 1, &GlobalPlanner::callback_jackal, this);
        srv_pedestrian_ = nh_.serviceClient<social_navigation::TrajectoryPredict>("/trajectory_predict");
    }

    void callback_flag(const std_msgs::Bool::ConstPtr& msg){
        ignore_ = true;
        clock_t init_time = clock();
        while(1){
            if(double(clock() - init_time) / CLOCKS_PER_SEC > 4.0) break;
        }
        ignore_ = false;
        return;
    }

    void callback_request(const social_navigation::Request::ConstPtr& msg){
        clock_t init_time = clock();
        cout << "start request" << endl;
        if(ignore_ && !msg->reset) {
            cout << "ignore" << endl;
            return;
        }
        global_goal_ = point(msg->goal.x, msg->goal.y);
        if(msg->reset){
            local_goal_ = point(msg->jackal.x, msg->jackal.y);
        }
        cout << "start mcts" << endl;
        mcts();
        cout << "end mcts" << endl;
        clock_t end_time = clock();
        cout << "elapsed time: " << double(end_time-init_time)/CLOCKS_PER_SEC << endl;
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
    }

    void callback_lidar(const sensor_msgs::LaserScan::ConstPtr& msg){
        double jt = jt_;
        double jx = jx_;
        double jy = jy_;
        vector<point> pts; 
        int n_points = msg->ranges.size();
        double w = msg->angle_min;
        double dw = msg->angle_increment;
        for(int k = 0; k < n_points; k++){
            if(isinf(msg->ranges[k])||isnan(msg->ranges[k])) continue;
            double a = jt + w + dw * k;
            double x = jx + msg->ranges[k] * cos(a);
            double y = jy + msg->ranges[k] * sin(a);
            pts.push_back(point(x,y));
        }
        lidar_points_ = pts;

    }

    int rsample(int idx){
        double sum = 0.0;
        for(int i : tree_[idx].childs){
            sum += tree_[i].weight;
        }
        double x = 1.0 * rand() / RAND_MAX;
        double y = 0.0;
        int rt = -1;
        for(int i = 0; i < tree_[idx].childs.size(); i++){
            y += tree_[tree_[idx].childs[i]].weight / sum;
            if(x < y) {
                rt = i;
                break;
            }
        }
        if(rt == -1) rt = tree_[idx].childs.size() - 1;
        return rt;
    }

    int softmax_sample(const vector<double>& w){
        int N = w.size();
        double s = 0.0;
        for(int i = 0; i < N; i++) s += exp(w[i]);
        double x = 1.0 * rand() / RAND_MAX;
        double y = 0.0;
        for(int i = 0; i < N; i++){
            y += exp(w[i]) / s;
            if(x < y) return i;
        }
        return N-1;
    }

    void test_rrt(){
        clock_t init_time = clock();
        cout << "start loop" << endl;
        // rrt.rrt(point(-29.38, -5.60), point(28.73, 15.92));
        // rrt.carrt(point(-29.38, -5.60), point(-21.05, -1.17));
        // rrt.diverse_rrt(point(-29.38, -5.60), point(-21.05, -1.17), 10);
        rrt.diverse_rrt(point(12.38, -3.60), point(28.73, 15.92), 5);
        // rrt.diverse_rrt(point(-29.38, -5.60), point(15.19, -7.38), 1);
        // rrt.diverse_rrt(point(0.38, -5.60), point(28.73, 15.92), 3);
        
        cout << double(clock() - init_time) / CLOCKS_PER_SEC << endl;
        cout << "end rrt" << endl;
    }

    void mcts(){
        // fetch current status (lidar point clouds, jackal position)
        // fetch global pedestrian trajectory (time: 0.0 sec ~ 2.0 sec, 0.1 sec interval)
        vector<point> pts = lidar_points_;
        point jackal = point(jx_, jy_);

        // call service
        social_navigation::TrajectoryPredict pedestrians;
        pedestrians.request.times = time_array_;
        

        if(!srv_pedestrian_.call(pedestrians)){
            cout << "error" << endl;
        }
        

        // generate cost map
        // remove point clouds which are near to pedestrians
        vector<point> static_points;
        vector<vector<point>> ped_goals;
        int T = pedestrians.response.times.size();
        int P = pedestrians.response.velocity.size();
        for(int t = 0; t < T; t++){
            vector<point> ped_goal;
            for(int i = 0; i < P; i++){
                geometry_msgs::Point q = pedestrians.response.trajectories[i].trajectory[t];
                ped_goal.push_back(point(q.x, q.y));
                if(t==0) cout << q.x << " " << q.y << endl;
            }
            ped_goals.push_back(ped_goal);
        }
        cout << "processing lidar" << endl;
        cout << pts.size() << endl;
        for(point p : pts){
            bool check = true;
            for(int i = 0; i < P; i++) {
                if(dist(p, ped_goals[0][i]) < 0.5) {
                    check = false;
                    break;
                }
            }
            if(check) static_points.push_back(p);
        }
        rrt.reset();
        rrt.set_pedestrians(ped_goals[0]);
        rrt.make_local_map(static_points);


        // generate local goal candidate
        vector<point> candidates;
        // vector<vector<point>> paths = rrt.diverse_rrt(local_goal_, global_goal_, n_cand_);
        vector<vector<point>> paths = rrt.diverse_rrt(jackal, global_goal_, n_cand_);
        for(const vector<point>& path : paths) candidates.push_back(path[max<int>(path.size() - lookahead_, 0)]);


        // generate tree root nodes
        int sz = 0;
        for(int i = 0; i < n_cand_; i++){
            Tnode nnode = Tnode();
            nnode.jackal = jackal;
            nnode.goal = candidates[i];
            nnode.peds = ped_goals[0];
            nnode.reward = rrt.get_state_reward(jackal, jackal, nnode.goal, ped_goals[0]);
            nnode.value = nnode.reward;
            nnode.weight = 0.0;
            nnode.n_visit = 0;
            nnode.is_leaf = false;
            nnode.parent = -1;
            nnode.depth = 0;
            tree_.push_back(nnode);
            sz++;
        }
        for(int i = 0; i < n_cand_; i++){
            for(int j = 0; j < n_actions_; j++){
                Tnode nnode = Tnode();
                nnode.jackal = jackal + actions_[j];
                nnode.goal = tree_[i].goal;
                nnode.peds = get_next_pedestrians(jackal, ped_goals[0], ped_goals[1], pedestrians.response.velocity);
                nnode.reward = rrt.get_state_reward(nnode.jackal, jackal, nnode.goal, nnode.peds);
                nnode.value = nnode.reward;
                nnode.weight = exp(nnode.value + alpha_visit_);
                nnode.n_visit = 0;
                nnode.is_leaf = true;
                nnode.parent = i;
                nnode.depth = 1;
                tree_[i].childs.push_back(sz);
                tree_.push_back(nnode);
                sz++;
            }
        }


        clock_t  init_time = clock();
        int steps = 0;
        int depth_maximum = 0;
        while(1){
            if(double(clock() - init_time) / CLOCKS_PER_SEC > time_limit_) break;
            steps++;
            // select candidate uniformly
            int goal_index = rand() % n_cand_;
            
            // traverse until leaf node
            int cur_idx = goal_index;
            while(1){
                if(tree_[cur_idx].is_leaf) break;
                cur_idx = tree_[cur_idx].childs[rsample(cur_idx)];
            }

            // if the visit count is over threshold, expand tree and select leaf node
            if(tree_[cur_idx].n_visit >= visit_threshold_){
                for(int i = 0; i < n_actions_; i++){
                    Tnode nnode = Tnode();
                    nnode.jackal = tree_[cur_idx].jackal + actions_[i];
                    nnode.goal = tree_[cur_idx].goal;
                    nnode.depth = tree_[cur_idx].depth + 1;
                    nnode.peds = get_next_pedestrians(jackal, ped_goals[nnode.depth - 1], ped_goals[nnode.depth], pedestrians.response.velocity);
                    nnode.reward = rrt.get_state_reward(nnode.jackal, tree_[cur_idx].jackal, nnode.goal, nnode.peds);
                    nnode.value = nnode.reward;
                    nnode.weight = exp(nnode.value + alpha_visit_);
                    nnode.n_visit = 0;
                    nnode.is_leaf = true;
                    nnode.parent = cur_idx;
                    tree_[cur_idx].childs.push_back(sz);
                    tree_.push_back(nnode);
                    sz++;
                }
                tree_[cur_idx].is_leaf = false;
                cur_idx = tree_[cur_idx].childs[rsample(cur_idx)];
            }

            double tot_value = -tree_[cur_idx].reward;
            double discounted_factor = gamma_;
            vector<point> peds = tree_[cur_idx].peds;
            point robot = tree_[cur_idx].jackal;
            point goal = tree_[cur_idx].goal;
            depth_maximum = max<int>(tree_[cur_idx].depth, depth_maximum);
            for(int i = tree_[cur_idx].depth; i < max_depth_; i++){
                // calculate cost by each action
                vector<double> value_list;
                peds = get_next_pedestrians(robot, peds, ped_goals[i], pedestrians.response.velocity);
                for(int j = 0; j < n_actions_; j++) value_list.push_back(rrt.get_state_reward(robot + actions_[j], robot, goal, peds));
                // sample action
                int a_idx = softmax_sample(value_list);
                // execute action and add cost
                tot_value += value_list[a_idx] * discounted_factor;
                discounted_factor *= gamma_;
                robot = robot + actions_[a_idx];
                if(dist(robot, goal) < distance_threshold_) break;
            }
            if(dist(robot, goal) > distance_threshold_) tot_value += -0.2 * dist(robot, goal) * discounted_factor;

            // update leaf node info
            tree_[cur_idx].value = (tree_[cur_idx].value * tree_[cur_idx].n_visit + tot_value) / (tree_[cur_idx].n_visit + 1);
            tree_[cur_idx].n_visit ++;
            tree_[cur_idx].weight = exp(tree_[cur_idx].value + alpha_visit_ / tree_[cur_idx].n_visit);

            // update tree
            while(1){
                cur_idx = tree_[cur_idx].parent;
                if(cur_idx == -1) break;
                double n_value = 0.0;
                double w_sum = 0.0;
                for(int i = 0; i < n_actions_; i++){
                    int idx = tree_[cur_idx].childs[i];
                    w_sum += tree_[idx].weight;
                    n_value += tree_[idx].value * tree_[idx].weight;
                }
                n_value /= w_sum;
                tree_[cur_idx].value = tree_[cur_idx].reward + gamma_ * n_value;
                tree_[cur_idx].n_visit ++;
                tree_[cur_idx].weight = exp(tree_[cur_idx].value + alpha_visit_ / tree_[cur_idx].n_visit);
            }
        }
        cout << "max depth: " << depth_maximum << endl;

        // select the best candidate and publish
        double best_value = tree_[0].value - dist(tree_[0].goal, local_goal_);
        double best_cand = 0;
        for(int i = 1; i < n_cand_; i++){
            if(best_value < tree_[i].value - dist(tree_[i].goal, local_goal_)) {
                best_value = tree_[i].value - dist(tree_[i].goal, local_goal_);
                best_cand = i;
            }
        }
        for(int i = 0; i < n_cand_; i++){
            cout << i << "(" << tree_[i].n_visit << "): " << tree_[i].value << endl;
        }
        cout << "###########################################" << endl;
        cout << "best(" << best_cand << "): " << best_value << endl;
        local_goal_ = tree_[best_cand].goal;
        jackal = paths[best_cand][max<int>(paths[best_cand].size() - 5, 0)];
        local_goal_.print();
        rrt.draw_mcts_result(paths, best_cand, global_goal_, ped_goals);
        
        geometry_msgs::Point rt;
        rt.x = local_goal_.x;
        rt.y = local_goal_.y;
        pub_.publish(rt);
        tree_.clear();
    }
        

};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "global_planner");
  GlobalPlanner mcts = GlobalPlanner();
  ros::spin();
}