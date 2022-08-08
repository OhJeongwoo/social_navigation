#ifndef RRT_H
#define RRT_H
#include <iostream>
#include <fstream>
#include <iomanip>
#include <string.h>
#include <sstream>
#include <vector>
#include <queue>
#include "Kernel.h"
#include "Transform.h"
#include "geometry_msgs/Point.h"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

class RRT{
    public:
    string cost_map_file_;
    Mat cost_map_;
    Mat local_map_;
    Mat path_map_;
    vector<point> pedestrians_;
    point root_;
    point goal_;
    Transform transform_;
    int img_w_;
    int img_h_;

    // hyperparamters
    int max_step_; // maximum number of rrt sampling trial
    int max_samples_;
    int step_; // step size for ege cost estimation
    double tau_; // tree expansion length
    double lambda_; // default coefficient for cost
    double coeff_tau_; // distance for rewiring
    double collision_threshold_; // collision cost threshold
    double goal_threshold_;
    double time_limit_;
    double lambda_ped_;
    double lambda_static_;
    double lambda_dist_;
    double MAX_COST_;

    double slack_cost_;

    Kernel kernel_;
    Kernel kernel_path_;
    Kernel kernel_ped_;
    double M_;
    double M_path_;
    double M_ped_;
    double alpha_;
    double alpha_path_;
    double alpha_ped_;
    double cost_path_;
    int kernel_half_;
    int kernel_path_half_;
    int kernel_ped_half_;
    int kernel_size_;
    int kernel_path_size_;
    int kernel_ped_size_;

    RRT();
    RRT(string cost_map_file);

    void initialize();
    void reset();
    void set_pedestrians(const vector<point>& pedestrians);
    void update_path_cost_map(const vector<point>& path);
    void make_local_map(const vector<point>& points);
    void drawing(const vector<node>& tree);
    void draw_diverse_path(const vector<vector<point>>& trees, int best_tree);
    void draw_diverse_result(const vector<vector<point>>& trees);
    void draw_mcts_result(const vector<vector<point>>& trees, int best_tree, point global_goal, vector<vector<point>>& peds);
    void draw_global_result(int seq, point jackal, point goal, int best_cand, const vector<point>& candidates, vector<vector<point>>& peds, vector<geometry_msgs::Point>& global_path, vector<vector<double>>& scores);
    bool is_collision(point p, point q);
    double get_cost(point p);
    double get_collision_cost(point p);
    double get_edge_cost(point p, point q);
    pb get_state_reward(point robot, point prev_robot, point goal, const vector<point>& peds);
    vector<point> rrt(point root, point goal);
    vector<point> rrt_star(point root, point goal);
    vector<point> carrt(point root, point goal);
    vector<vector<point>> diverse_rrt(point root, point goal, int K);
};


#endif