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
#include "social_navigation/PathArray.h"
#include "social_navigation/Request.h"
#include "social_navigation/GlobalPathRequest.h"
#include "social_navigation/GlobalPathResponse.h"
#include "social_navigation/GlobalPlannerRequest.h"
#include "social_navigation/GlobalPlannerResponse.h"

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
    ros::Publisher pub_; // global_planner/response
    ros::Publisher pub_path_; // global_path/request
    ros::Subscriber sub_; // global_planner/request
    ros::Subscriber sub_goal_; // global_goal
    ros::Subscriber sub_lidar_; // scan
    ros::Subscriber sub_jackal_; // jackal or /gazebo/model_states
    ros::Subscriber sub_path_; // global_path/response
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
    double lookahead_distance_;
    
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
    vector<geometry_msgs::Point> global_path_;
    int record_num_;
    bool has_local_goal_;
    int age_;
    double cost_cut_threshold_;
    bool mcts_mode_;
    bool const_vel_mode_;
    bool carrt_mode_;
    bool mpc_mode_;
    
    double lambda_;

    RRT rrt;

    public:
    GlobalPlanner(){
        // set random seed
        srand(time(NULL));
        
        mcts_mode_ = false;
        const_vel_mode_ = false;
        carrt_mode_ = true;
        mpc_mode_ = false;

        // load cost map
        pkg_path_ << ros::package::getPath("social_navigation") << "/";
        cost_map_file_ = pkg_path_.str() + "config/costmap_301_1f.png";

        // load RRT module
        rrt = RRT(cost_map_file_);

        // set hyperparameter
        if(mcts_mode_) n_cand_ = 5;
        else n_cand_ = 1;
        lookahead_distance_ = 5.0;
        V_ = 1.5;
        dt_ = 0.2;
        double sv = 0.5 * V_ * dt_;
        double fv = 1.0 * V_ * dt_;
        actions_.push_back(point(0.0, 0.0));
        int angle = 4;
        for(int i = 0; i < angle; i++) actions_.push_back(point(sv * cos(2.0 * M_PI * i / angle), sv * sin(2.0 * M_PI * i / angle)));
        for(int i = 0; i < angle; i++) actions_.push_back(point(fv * cos(2.0 * M_PI * i / angle), fv * sin(2.0 * M_PI * i / angle)));
        n_actions_ = actions_.size();
        visit_threshold_ = 5;
        max_depth_ = 20;
        gamma_ = 0.99;
        alpha_visit_ = 1.0;
        distance_threshold_ = 1.0;
        time_limit_ = 0.2;
        rrt.time_limit_ = time_limit_ / n_cand_;
        ignore_ = false;
        record_num_ = 0;
        has_local_goal_ = false;
        age_ = 0;
        cost_cut_threshold_ = 8.0;
        lambda_ = 0.2;

        for(int i = 0; i < max_depth_ + 1; i++) time_array_.push_back(dt_ * i);

        pub_ = nh_.advertise<social_navigation::GlobalPlannerResponse>("/global_planner/response", 1000);
        pub_path_ = nh_.advertise<social_navigation::GlobalPathRequest>("/global_path/request", 1000);
        sub_ = nh_.subscribe("/global_planner/request", 1, &GlobalPlanner::callback_request, this);
        sub_goal_ = nh_.subscribe("/global_goal", 1, &GlobalPlanner::callback_goal, this);
        sub_path_ = nh_.subscribe("/global_path/response", 1, &GlobalPlanner::callback_path, this);
        sub_lidar_ = nh_.subscribe("/front/scan", 1, &GlobalPlanner::callback_lidar, this);
        sub_jackal_ = nh_.subscribe("/gazebo/model_states", 1, &GlobalPlanner::callback_jackal, this);
        srv_pedestrian_ = nh_.serviceClient<social_navigation::TrajectoryPredict>("/trajectory_predict");
        
        cout << "initialize completed" << endl;
        if(mpc_mode_) cout << "MPC Mode" << endl;
        else{
            if(mcts_mode_) {
                if(const_vel_mode_) cout << "MCTS-CV Mode" << endl;
                else cout << "SAN-MCTS Mode" << endl;
            }
            else{
                if(carrt_mode_) cout << "CARRT-REP Mode" << endl;
                else cout << "RRT-REP Mode" << endl;
            }
        }
    }


    void callback_goal(const geometry_msgs::Point::ConstPtr& msg){
        global_goal_ = point(msg->x, msg->y);
        social_navigation::GlobalPathRequest req;
        req.id = 0;
        req.type = 1;
        req.n_path = 5;
        geometry_msgs::Point req_root;
        req_root.x = jx_;
        req_root.y = jy_;
        req.root = req_root;
        geometry_msgs::Point req_goal;
        req_goal.x = global_goal_.x;
        req_goal.y = global_goal_.y;
        req.goal = req_goal;
        pub_path_.publish(req);
    }

    void callback_path(const social_navigation::GlobalPathResponse::ConstPtr& msg){
        global_path_ = msg->points;
    }

    void callback_request(const social_navigation::GlobalPlannerRequest::ConstPtr& msg){
        if(msg->seq==0) {
            has_local_goal_ = false;
            age_ = 0;
        }
        if(mpc_mode_) mpc(msg->id, msg->seq);
        else mcts(msg->id, msg->seq);
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

    void mcts(int id, int seq){
        record_num_ ++;
        // fetch current status (lidar point clouds, jackal position)
        // fetch global pedestrian trajectory (time: 0.0 sec ~ 2.0 sec, 0.1 sec interval)
        vector<point> pts = lidar_points_;
        point jackal = point(jx_, jy_);
        point goal = global_goal_;
        vector<geometry_msgs::Point> global_path = global_path_;

        // call service
        social_navigation::TrajectoryPredict pedestrians;
        pedestrians.request.times = time_array_;
        
        bool success = true;
        if(!srv_pedestrian_.call(pedestrians)){
            cout << "error" << endl;
            success = false;
        }
        

        // generate cost map
        // remove point clouds which are near to pedestrians
        vector<point> static_points;
        vector<vector<point>> ped_goals;
        int T = pedestrians.response.times.size();
        int P = pedestrians.response.velocity.size();
        if(const_vel_mode_){
            vector<double> ped_vel;
            for(int i = 0; i < P; i++) ped_vel.push_back(1.5);
            pedestrians.response.velocity = ped_vel;
            for(int t = 0; t < T; t++){
                vector<point> ped_goal;
                for(int i = 0; i < P; i++){
                    geometry_msgs::Point q0 = pedestrians.response.trajectories[i].trajectory[0];
                    geometry_msgs::Point q1 = pedestrians.response.trajectories[i].trajectory[1];
                    point dir = normalize(point(q1.x, q1.y) - point(q0.x, q0.y));
                    double x = q0.x + 1.5 * dt_ * dir.x;
                    double y = q0.y + 1.5 * dt_ * dir.y;
                    ped_goal.push_back(point(x,y));
                }
                ped_goals.push_back(ped_goal);
            }
        }
        else{
            for(int t = 0; t < T; t++){
                vector<point> ped_goal;
                for(int i = 0; i < P; i++){
                    geometry_msgs::Point q = pedestrians.response.trajectories[i].trajectory[t];
                    ped_goal.push_back(point(q.x, q.y));
                }
                ped_goals.push_back(ped_goal);
            }
        }
        
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
        vector<vector<point>> paths;
        if(n_cand_ == 1){
            vector<point> path;
            cout << "replanning mode" << endl;
            if(carrt_mode_) cout << "carrt mode" << endl;
            else cout << "rrt star mode" << endl;
            if(carrt_mode_) path = rrt.carrt(jackal, goal);
            else path = rrt.rrt_star(jackal, goal);
            int sz = path.size();
            bool check = true;
            point candidate;
            for(int i = sz-1; i --; i>=0){
                if(dist(jackal, path[i]) > lookahead_distance_){
                    candidate = path[i];
                    check = false;
                    break;
                }
            }
            if(check) candidate = path[0];
            social_navigation::GlobalPlannerResponse rt;
            rt.id = id;
            rt.seq = seq;
            geometry_msgs::Point rt_local_goal;
            rt_local_goal.x = candidate.x;
            rt_local_goal.y = candidate.y;
            rt.local_goal = rt_local_goal;
            pub_.publish(rt);
            return;
        }

        // jeongowo 220721
        // if(seq > 1) paths = rrt.diverse_rrt(jackal, goal, n_cand_);
        // else paths = rrt.diverse_rrt(jackal, goal, n_cand_ - 1);
        // // for(const vector<point>& path : paths) candidates.push_back(path[max<int>(path.size() - lookahead_, 0)]);
        // for(const vector<point>& path : paths) {
        //     int sz = path.size();
        //     bool check = true;
        //     for(int i = sz-1; i --; i>=0){
        //         if(dist(jackal, path[i]) > lookahead_distance_){
        //             candidates.push_back(path[i]);
        //             check = false;
        //             break;
        //         }
        //     }
        //     if(check) candidates.push_back(path[0]);
        // }
        // if(seq==0) candidates.push_back(local_goal_);

        //jeongwoo 220722
        if(has_local_goal_) paths = rrt.diverse_rrt(jackal, goal, n_cand_ - 1);
        else paths = rrt.diverse_rrt(jackal, goal, n_cand_);
        for(const vector<point>& path : paths) {
            int sz = path.size();
            bool check = true;
            for(int i = sz-1; i --; i>=0){
                if(dist(jackal, path[i]) > lookahead_distance_){
                    candidates.push_back(path[i]);
                    check = false;
                    break;
                }
            }
            if(check) candidates.push_back(path[0]);
        }
        if(has_local_goal_) candidates.push_back(local_goal_);
        

        // generate tree root nodes
        int sz = 0;
        for(int i = 0; i < n_cand_; i++){
            Tnode nnode = Tnode();
            nnode.jackal = jackal;
            nnode.goal = candidates[i];
            nnode.peds = ped_goals[0];
            pb simul_result = rrt.get_state_reward(jackal, jackal, nnode.goal, ped_goals[0]);
            point return_value = simul_result.first;
            bool done = simul_result.second;
            
            nnode.reward = return_value.x;
            nnode.cost = return_value.y;
            nnode.value = nnode.reward;
            nnode.cvalue = nnode.cost;
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
                nnode.peds = get_next_pedestrians(jackal, ped_goals[0], ped_goals[1], pedestrians.response.velocity, const_vel_mode_);
                pb simul_result = rrt.get_state_reward(nnode.jackal, jackal, nnode.goal, nnode.peds);
                point return_value = simul_result.first;
                bool done = simul_result.second;
                nnode.reward = return_value.x;
                nnode.cost = return_value.y;
                nnode.value = nnode.reward;
                nnode.cvalue = nnode.cost;
                nnode.weight = exp(nnode.value - lambda_ * nnode.cost + alpha_visit_);
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
                    nnode.peds = get_next_pedestrians(jackal, ped_goals[nnode.depth - 1], ped_goals[nnode.depth], pedestrians.response.velocity, const_vel_mode_);
                    pb simul_result = rrt.get_state_reward(nnode.jackal, tree_[cur_idx].jackal, nnode.goal, nnode.peds);
                    point return_value = simul_result.first;
                    bool done = simul_result.second;
                    nnode.reward = return_value.x;
                    nnode.cost = return_value.y;
                    nnode.value = nnode.reward;
                    nnode.cvalue = nnode.cost;
                    nnode.weight = exp(nnode.value - lambda_ * nnode.cost + alpha_visit_);
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

            double tot_value = tree_[cur_idx].reward;
            double tot_cost = tree_[cur_idx].cost;
            double discounted_factor = gamma_;
            vector<point> peds = tree_[cur_idx].peds;
            point robot = tree_[cur_idx].jackal;
            point goal = tree_[cur_idx].goal;
            depth_maximum = max<int>(tree_[cur_idx].depth, depth_maximum);
            int cur_depth = tree_[cur_idx].depth;
            bool success = false;
            for(int i = tree_[cur_idx].depth; i < max_depth_; i++){
                // calculate cost by each action
                vector<double> sample_list;
                vector<double> value_list;
                vector<double> cost_list;
                vector<double> done_list;
                peds = get_next_pedestrians(robot, peds, ped_goals[i], pedestrians.response.velocity, const_vel_mode_);
                for(int j = 0; j < n_actions_; j++){
                    pb simul_result = rrt.get_state_reward(robot + actions_[j], robot, goal, peds);
                    point return_value = simul_result.first;
                    bool done = simul_result.second;
                    sample_list.push_back(return_value.x - lambda_ * return_value.y);
                    value_list.push_back(return_value.x);
                    cost_list.push_back(return_value.y);
                    done_list.push_back(done);
                }
                // sample action
                int a_idx = softmax_sample(sample_list);
                // execute action and add cost
                tot_value += value_list[a_idx] * discounted_factor;
                tot_cost += cost_list[a_idx] * discounted_factor;
                discounted_factor *= gamma_;
                robot = robot + actions_[a_idx];
                cur_depth ++;
                if(done_list[a_idx]) break;
                if(dist(robot, goal) < distance_threshold_) {
                    success = true;
                    break;
                }
            }
            if(dist(robot, goal) > distance_threshold_) tot_value += -1.0 * dist(robot, goal) * discounted_factor;
            if(!success && cur_depth < max_depth_) tot_cost += rrt.MAX_COST_ * discounted_factor;
            // update leaf node info
            tree_[cur_idx].value = (tree_[cur_idx].value * tree_[cur_idx].n_visit + tot_value) / (tree_[cur_idx].n_visit + 1);
            tree_[cur_idx].cvalue = (tree_[cur_idx].cvalue * tree_[cur_idx].n_visit + tot_cost) / (tree_[cur_idx].n_visit + 1);
            tree_[cur_idx].n_visit ++;
            // int par_idx = tree_[cur_idx].parent;
            // tree_[cur_idx].weight = exp(tree_[cur_idx].value - tree_[cur_idx].cvalue * lambda_ + alpha_visit_ * sqrt(log(tree_[par_idx].n_visit + n_actions_ + 1) / (tree_[cur_idx].n_visit + 1)));
            tree_[cur_idx].weight = exp(tree_[cur_idx].value - tree_[cur_idx].cvalue * lambda_ + alpha_visit_ / tree_[cur_idx].n_visit);

            // update tree
            while(1){
                cur_idx = tree_[cur_idx].parent;
                if(cur_idx == -1) break;
                // int par_idx = tree_[cur_idx].parent;
                double n_value = 0.0;
                double n_cost = 0.0;
                double w_sum = 0.0;
                for(int i = 0; i < n_actions_; i++){
                    int idx = tree_[cur_idx].childs[i];
                    w_sum += tree_[idx].weight;
                    n_value += tree_[idx].value * tree_[idx].weight;
                    n_cost += tree_[idx].cvalue * tree_[idx].weight;
                }
                n_value /= w_sum;
                n_cost /= w_sum;
                tree_[cur_idx].value = tree_[cur_idx].reward + gamma_ * n_value;
                tree_[cur_idx].cvalue = tree_[cur_idx].cost + gamma_ * n_cost;
                tree_[cur_idx].n_visit ++;
                tree_[cur_idx].weight = exp(tree_[cur_idx].value - lambda_ * tree_[cur_idx].cvalue + alpha_visit_ / tree_[cur_idx].n_visit);
                

                // double max_value = -1000.0;
                // double max_cost = 0.0;
                // for(int i = 0; i < n_actions_; i++){
                //     int idx = tree_[cur_idx].childs[i];
                //     if(tree_[idx].value > max_value){
                //         max_value = tree_[idx].value;
                //         max_cost = tree_[idx].cost;
                //     }
                // }
                // tree_[cur_idx].value = tree_[cur_idx].reward + gamma_ * max_value;
                // tree_[cur_idx].cvalue = tree_[cur_idx].cost + gamma_ * max_cost;
                // tree_[cur_idx].n_visit ++;
                // if (par_idx != -1) tree_[cur_idx].weight = exp(tree_[cur_idx].value - lambda_ * tree_[cur_idx].cvalue + alpha_visit_ * sqrt(log(tree_[par_idx].n_visit + n_actions_ + 1) / (tree_[cur_idx].n_visit + 1)));
            }
        }

        // select the best candidate and publish
        double best_value = -INF;
        int best_cand = -1;
        vector<vector<double>> scores;
        for(int i = 0; i < n_cand_; i++){
            bool is_dangerous = false;
            vector<double> score;
            double tree_value = tree_[i].value;
            double tree_cost = tree_[i].cvalue;
            double local_value = 0.0;
            double global_value = 0.0;
            if (tree_cost > cost_cut_threshold_) is_dangerous = true;
            if (has_local_goal_) {
                local_value = -0.2 * dist(tree_[i].goal, local_goal_);
                if(dist(tree_[i].goal, jackal) < 1.0) is_dangerous = true;
            }
            if (global_path.size() > 0){
                double min_d = INF;
                double remain_d;
                for(int j = 0; j < global_path.size(); j++){
                    if(min_d > dist(tree_[i].goal, point(global_path[j].x, global_path[j].y))){
                        min_d = dist(tree_[i].goal, point(global_path[j].x, global_path[j].y));
                        remain_d = min_d + global_path[j].z;
                    }
                }
                global_value = -0.1 * remain_d;
            }
            double value = tree_value + local_value + global_value;
            score.push_back(tree_value);
            score.push_back(tree_cost);
            score.push_back(local_value);
            score.push_back(global_value);
            scores.push_back(score);
            if(is_dangerous) continue;
            if(best_cand < 0 || best_value < value){
                best_value = value;
                best_cand = i;
            }
        }
        
        bool estop = false;
        if(best_cand == -1) {
            estop = true;
            has_local_goal_ = false;
            age_ = 0;
        }
        else if(has_local_goal_ && best_cand == n_cand_ - 1){
            age_++;
            if(age_ > 20) has_local_goal_ = false;
        }
        else {
            local_goal_ = tree_[best_cand].goal;
            has_local_goal_ = true;
            age_ = 0;
        }
        // rrt.draw_global_result(record_num_, jackal, goal, best_cand, candidates, ped_goals, global_path, scores);
        
        social_navigation::GlobalPlannerResponse rt;
        rt.id = id;
        rt.seq = seq;
        geometry_msgs::Point rt_local_goal;
        rt_local_goal.x = local_goal_.x;
        rt_local_goal.y = local_goal_.y;
        rt.local_goal = rt_local_goal;
        rt.estop = estop;
        pub_.publish(rt);

        social_navigation::GlobalPathRequest req;
        req.id = id;
        req.type = 1;
        req.n_path = 5;
        geometry_msgs::Point req_root;
        req_root.x = jackal.x;
        req_root.y = jackal.y;
        req.root = req_root;
        geometry_msgs::Point req_goal;
        req_goal.x = goal.x;
        req_goal.y = goal.y;
        req.goal = req_goal;
        pub_path_.publish(req);

        tree_.clear();
    }
        
    void mpc(int id, int seq){
        cout << "mpc mode" << endl;
        clock_t  start_time = clock();
        record_num_ ++;
        // fetch current status (lidar point clouds, jackal position)
        // fetch global pedestrian trajectory (time: 0.0 sec ~ 2.0 sec, 0.1 sec interval)
        vector<point> pts = lidar_points_;
        point jackal = point(jx_, jy_);
        point goal = global_goal_;
        vector<geometry_msgs::Point> global_path = global_path_;

        // call service
        social_navigation::TrajectoryPredict pedestrians;
        pedestrians.request.times = time_array_;
        
        bool success = true;
        if(!srv_pedestrian_.call(pedestrians)){
            cout << "error" << endl;
            success = false;
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
            }
            ped_goals.push_back(ped_goal);
        }
        
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
        vector<vector<point>> paths;

        paths = rrt.diverse_rrt(jackal, goal, n_cand_);
        for(const vector<point>& path : paths) {
            int sz = path.size();
            bool check = true;
            for(int i = sz-1; i --; i>=0){
                if(dist(jackal, path[i]) > lookahead_distance_){
                    candidates.push_back(path[i]);
                    check = false;
                    break;
                }
            }
            if(check) candidates.push_back(path[0]);
        }
        

        // generate tree root nodes
        vector<double> values;
        for(int i = 0; i < n_cand_; i++) values.push_back(-1000);
        vector<double> auxilary_values;
        for(int i = 0; i < n_cand_; i++){
            double val = 0.0;
            point cur_goal = candidates[i];
            if (has_local_goal_) {
                val += -0.2 * dist(cur_goal, local_goal_);
            }
            if (global_path.size() > 0){
                double min_d = INF;
                double remain_d;
                for(int j = 0; j < global_path.size(); j++){
                    if(min_d > dist(cur_goal, point(global_path[j].x, global_path[j].y))){
                        min_d = dist(cur_goal, point(global_path[j].x, global_path[j].y));
                        remain_d = min_d + global_path[j].z;
                    }
                }
                val += -0.1 * remain_d;
            }
            auxilary_values.push_back(val);
        }
        
        int best_cand = -1;
        double best_value = -1000;
        clock_t  init_time = clock();
        while(1){
            if(double(clock() - init_time) / CLOCKS_PER_SEC > time_limit_) break;
            // select candidate uniformly
            int goal_index = rand() % n_cand_;
            
            double tot_value = 0;
            double tot_cost = 0;
            double discounted_factor = 1.0;
            vector<point> peds = ped_goals[0];
            point robot = jackal;
            point goal = candidates[goal_index];
            int cur_depth = 1;
            bool success = false;
            for(int i = cur_depth; i < max_depth_; i++){
                // calculate cost by each action
                peds = get_next_pedestrians(robot, peds, ped_goals[i], pedestrians.response.velocity, const_vel_mode_);
                double x = V_ * 2.0 * (rand() / RAND_MAX - 0.5);
                double y = V_ * 2.0 * (rand() / RAND_MAX - 0.5);
                point action = point(x,y);
                pb simul_result = rrt.get_state_reward(robot + action, robot, goal, peds);
                point return_value = simul_result.first;
                double reward = return_value.x;
                double cost = return_value.y;
                bool done = simul_result.second;
                
                // execute action and add cost
                tot_value += reward * discounted_factor;
                tot_cost += cost * discounted_factor;
                discounted_factor *= gamma_;
                robot = robot + action;
                cur_depth ++;
                if(done) break;
                if(dist(robot, goal) < distance_threshold_) {
                    success = true;
                    break;
                }
            }
            if(dist(robot, goal) > distance_threshold_) tot_value += -1.0 * dist(robot, goal) * discounted_factor;
            if(!success && cur_depth < max_depth_) tot_cost += rrt.MAX_COST_ * discounted_factor;
            if(tot_cost > cost_cut_threshold_) continue;
            if(values[goal_index] < tot_value) {
                values[goal_index] = tot_value;
                if(best_cand == -1 || best_value < tot_value + auxilary_values[goal_index]){
                    best_value = tot_value + auxilary_values[goal_index];
                    best_cand = goal_index;
                }
            }
        }

        bool estop = false;
        if(best_cand == -1) {
            estop = true;
            has_local_goal_ = false;
        }
        else {
            has_local_goal_ = true;
            local_goal_ = candidates[best_cand];
        }
        cout << double(clock() - start_time) / CLOCKS_PER_SEC << endl; 

        social_navigation::GlobalPlannerResponse rt;
        rt.id = id;
        rt.seq = seq;
        geometry_msgs::Point rt_local_goal;
        rt_local_goal.x = local_goal_.x;
        rt_local_goal.y = local_goal_.y;
        rt.local_goal = rt_local_goal;
        rt.estop = estop;
        pub_.publish(rt);

        social_navigation::GlobalPathRequest req;
        req.id = id;
        req.type = 1;
        req.n_path = 5;
        geometry_msgs::Point req_root;
        req_root.x = jackal.x;
        req_root.y = jackal.y;
        req.root = req_root;
        geometry_msgs::Point req_goal;
        req_goal.x = goal.x;
        req_goal.y = goal.y;
        req.goal = req_goal;
        pub_path_.publish(req);

        tree_.clear();
    }

};
 
int main(int argc,char** argv){
  ros::init(argc, argv, "mcts_global_planner");
  GlobalPlanner mcts = GlobalPlanner();
  ros::spin();
}