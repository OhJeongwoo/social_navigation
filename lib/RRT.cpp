#include <iostream>
#include "RRT.h"
#include "utils.h"

using namespace std;
using namespace cv;

RRT::RRT(){}

RRT::RRT(string cost_map_file){
    cost_map_file_ = cost_map_file;
    cost_map_ = imread(cost_map_file, CV_8UC1);
    initialize();
}


void RRT::initialize(){
    img_w_ = cost_map_.rows;
    img_h_ = cost_map_.cols;
    local_map_ = Mat(img_w_, img_h_, CV_8UC1, Scalar(0));
    path_map_ = Mat(img_w_, img_h_, CV_8UC1, Scalar(0));
    M_ = 200;
    M_path_ = 200;
    M_ped_ = 1.0;
    alpha_ = 0.05;
    alpha_path_ = 0.01;
    alpha_ped_ = 0.05;
    kernel_size_ = 30;
    kernel_path_half_ = 100;
    kernel_ped_half_ = 30;
    kernel_half_ = 30;
    kernel_ = Kernel(M_, alpha_, kernel_half_);
    kernel_path_ = Kernel(M_path_, alpha_path_, kernel_path_half_);
    kernel_ped_ = Kernel(M_ped_, alpha_ped_, kernel_ped_half_);
    cost_path_ = 3.0;
    lambda_ped_ = 10.0;
    lambda_static_ = 10.0;
    lambda_dist_ = 0.1;

    max_step_ = 20000; // maximum number of rrt sampling trial
    max_samples_ = 10000;
    step_ = 10; // step size for ege cost estimation
    tau_ = 0.2; // tree expansion length
    lambda_ = 0.01;
    coeff_tau_ = 2.0; // distance for rewiring
    collision_threshold_ = 0.85; // collision cost threshold
    goal_threshold_ = 1.0;
    slack_cost_ = 0.1;
    time_limit_ = 0.2;
}


void RRT::reset(){
    path_map_ = Mat(img_w_, img_h_, CV_8UC1, Scalar(0));
    cost_map_ = imread(cost_map_file_, CV_8UC1);
    // Mat background = cost_map_;
    // Mat background2 = local_map_;
    // string save_path = "/home/jeongwoooh/catkin_social/src/social_navigation/test_costmap.png";
    // string save_path2 = "/home/jeongwoooh/catkin_social/src/social_navigation/test_localmap.png";
    // cout << save_path << endl;
    // cout << save_path2 << endl;
    
    // cv::imwrite(save_path, background);
    // cv::imwrite(save_path2, background2);
}


void RRT::set_pedestrians(const vector<point>& pedestrians){pedestrians_ = pedestrians;}


void RRT::update_path_cost_map(const vector<point>& path){
    for(const point& p : path){
        pixel q = transform_.xy2pixel(p);
        for(int i = -kernel_path_half_; i <= kernel_path_half_;  i++){
            for(int j = -kernel_path_half_; j <= kernel_path_half_;  j++){
                int ix = kernel_path_half_ + i;
                int iy = kernel_path_half_ + j;
                int nx = q.x + i;
                int ny = q.y + j;
                path_map_.at<uchar>(nx, ny) = max<uchar>(path_map_.at<uchar>(nx, ny), uchar(kernel_path_.kernel[ix][iy]));
            }    
        }
    }
}


void RRT::make_local_map(const vector<point>& points){
    for(const point& p : points){
        pixel q = transform_.xy2pixel(p);
        for(int i = -kernel_half_; i <= kernel_half_;  i++){
            for(int j = -kernel_half_; j <= kernel_half_;  j++){
                int ix = kernel_half_ + i;
                int iy = kernel_half_ + j;
                int nx = q.x + i;
                int ny = q.y + j;
                local_map_.at<uchar>(nx, ny) = max<uchar>(local_map_.at<uchar>(nx, ny), uchar(kernel_.kernel[ix][iy]));
            }    
        }
    }
}


void RRT::drawing(const vector<node>& tree){
    Mat background = cost_map_;
    string save_path = "/home/jeongwoooh/catkin_social/src/social_navigation/test.png";
    // cout << save_path << endl;
    
    // for(int i = 0; i < img_w_; i++){
    //     for(int j =0; j < img_h_; j++) local_map_.at<uchar>(i, j) = max(local_map_.at<uchar>(i, j), local_cost_map_.at<uchar>(i, j));
    // }

    for(const node& p: tree){
        if(p.parent == -1) continue;
        pixel cur = transform_.xy2pixel(p.p);
        pixel par = transform_.xy2pixel(tree[p.parent].p);
        if(p.is_path) cv::line(background, Point(cur.y, cur.x), Point(par.y, par.x), Scalar(255), 5);
        else cv::line(background, Point(cur.y, cur.x), Point(par.y, par.x), Scalar(180), 2);
    }
    
    pixel root = transform_.xy2pixel(tree[0].p);
    cv::circle(background,  Point(root.y, root.x), 10.0, Scalar(280), -1);

    // pixel pgoal = transform_.xy2pixel(goal);
    // cv::circle(local_map_,  Point(pgoal.second, pgoal.first), 10.0, Scalar(240), -1);

    cv::imwrite(save_path, background);
}


void RRT::draw_diverse_path(const vector<vector<point>>& trees, int best_tree){
    Mat background = cost_map_;
    string save_path = "/home/jeongwoooh/catkin_social/src/social_navigation/diverse_test.png";
    // cout << save_path << endl;
    pixel root = transform_.xy2pixel(root_);
    cv::circle(background,  Point(root.y, root.x), 10.0, Scalar(0), -1);
    int idx = 0;
    for(const vector<point>& tree: trees){
        if(idx == best_tree){
            pixel root = transform_.xy2pixel(tree[max<int>(tree.size() - 25, 0)]);
            cv::circle(background,  Point(root.y, root.x), 10.0, Scalar(0), -1);
        }
        else{
            pixel root = transform_.xy2pixel(tree[max<int>(tree.size() - 25, 0)]);
            cv::circle(background,  Point(root.y, root.x), 10.0, Scalar(200), -1);
        }
        //for(int i = 1; i < min<int>(tree.size(), 25); i++){
        //    pixel cur = transform_.xy2pixel(tree[tree.size()-i]);
        //    pixel par = transform_.xy2pixel(tree[tree.size()-i-1]);
        //    if(idx == best_tree) cv::line(background, Point(cur.y, cur.x), Point(par.y, par.x), Scalar(20), 3);
        //    else cv::line(background, Point(cur.y, cur.x), Point(par.y, par.x), Scalar(255), 1);
        //}
        idx ++;
    }
    
    cv::imwrite(save_path, background);
}

void RRT::draw_mcts_result(const vector<vector<point>>& trees, int best_tree, point global_goal, vector<vector<point>>& peds){
    Mat background = cost_map_;
    string save_path = "/home/jeongwoooh/catkin_social/src/social_navigation/diverse_test.png";
    // cout << save_path << endl;
    pixel root = transform_.xy2pixel(root_);
    cv::circle(background,  Point(root.y, root.x), 10.0, Scalar(0), -1);

    pixel goal = transform_.xy2pixel(global_goal);
    cv::circle(background,  Point(goal.y, goal.x), 10.0, Scalar(255), -1);
    int idx = 0;
    for(const vector<point>& tree: trees){
        if(idx == best_tree){
            pixel root = transform_.xy2pixel(tree[max<int>(tree.size() - 25, 0)]);
            cv::circle(background,  Point(root.y, root.x), 5.0, Scalar(0), -1);
        }
        else{
            pixel root = transform_.xy2pixel(tree[max<int>(tree.size() - 25, 0)]);
            cv::circle(background,  Point(root.y, root.x), 5.0, Scalar(200), -1);
        }
        
        idx ++;
    }
    int T = peds.size();
    for(int t=0;t<T;t++){
        int P = peds[t].size();
        for(int i =0;i<P;i++){
            pixel p = transform_.xy2pixel(peds[t][i]);
            cv::circle(background,  Point(p.y, p.x), 10.0*(t+1)/T, Scalar(150), -1);
        }
        cout << P << endl;
    }
    
    cv::imwrite(save_path, background);
}


bool RRT::is_collision(point p, point q){
    for(int i = 0; i <= step_; i++){
        if(get_collision_cost(interpolate(p,q,1.0*i/step_)) > collision_threshold_) return true; 
    }
    return false;
}


double RRT::get_cost(point p){
    pixel q = transform_.xy2pixel(p);
    double rt = cost_map_.at<uchar>(q.x, q.y);
    rt = max<uchar>(rt, local_map_.at<uchar>(q.x, q.y));
    rt /= 255.0;
    rt += cost_path_ * path_map_.at<uchar>(q.x, q.y) / 255.0;
    double ped_cost = 0.0;
    for(int i = 0; i < pedestrians_.size(); i++) ped_cost = max<double>(ped_cost, kernel_ped_.get(dist(pedestrians_[i], p)));
    rt += ped_cost;
    return rt;
}


double RRT::get_collision_cost(point p){
    pixel q = transform_.xy2pixel(p);
    double rt = cost_map_.at<uchar>(q.x, q.y);
    rt = max<uchar>(rt, local_map_.at<uchar>(q.x, q.y));
    rt /= 255.0;
    return rt;
}


double RRT::get_edge_cost(point p, point q){
    double d = dist(p,q);
    double c = 0.0;
    for(int i = 0; i <= step_; i++) c += get_cost(interpolate(p,q,1.0*i/step_));
    c /= step_ + 1;
    return (c + lambda_) * d;
}


double RRT::get_state_cost(point robot, point prev_robot, point goal, const vector<point>& peds){
    pixel p = transform_.xy2pixel(robot);
    double static_cost = cost_map_.at<uchar>(p.x, p.y);
    static_cost = max<uchar>(static_cost, local_map_.at<uchar>(p.x, p.y));
    static_cost /= 255.0;
    double ped_cost = 0.0;
    for(point ped : peds){
        double d = dist(robot, ped);
        ped_cost = max<double>(ped_cost, M_ped_ * exp(- alpha_ped_ * d * d));
        // if(d < 3.0)cout << "d: " << d << ", cost: " << ped_cost << endl;
    }
    double delta = dist(robot, goal) - dist(prev_robot, goal);
    return lambda_static_ * static_cost + lambda_ped_ * ped_cost + lambda_dist_ * delta + slack_cost_;
}


vector<point> RRT::rrt(point root, point goal){
    int max_step = 500;
    int max_samples = 200;
    double x_min = min<double>(goal.x, root.x) - 5.0;
    double x_max = max<double>(goal.x, root.x) + 5.0;
    double y_min = min<double>(goal.y, root.y) - 5.0;
    double y_max = max<double>(goal.y, root.y) + 5.0;
    
    vector<node> tree;
    tree.push_back(node(root));
    int step = 0;
    int sz = 1;
    double goal_dist = dist(root, goal);
    int leaf_index = 0;
    while(1){
        step += 1;
        if(step > max_step) break;

        // sample random point
        double x = x_min + (x_max - x_min) * ((double)rand() / (double) RAND_MAX);
        double y = y_min + (y_max - y_min) * ((double)rand() / (double) RAND_MAX);
        point p  = point(x,y);
        
        // find nearest
        int nearest_idx = find_nearest(tree, p, false);

        // get candidate
        p = get_candidate(tree[nearest_idx].p, p, tau_);

        // check collision
        if (is_collision(tree[nearest_idx].p, p)) continue;

        // append candidate
        node nnode = node(p);
        nnode.parent = nearest_idx;
        nnode.cost = tree[nearest_idx].cost + get_edge_cost(tree[nearest_idx].p, p);
        tree[nearest_idx].childs.push_back(sz);
        tree.push_back(nnode);
        
        if(goal_dist > dist(p, goal)){
            goal_dist = dist(p, goal);
            leaf_index = sz;
        }
        sz ++;
        if(sz > max_samples || goal_dist < goal_threshold_) break;
    }
    
    vector<point> path;
    while(leaf_index != -1){
        path.push_back(tree[leaf_index].p);
        tree[leaf_index].is_path = true;
        leaf_index = tree[leaf_index].parent;
    }
    
    return path;
}


vector<point> RRT::rrt_star(point root, point goal){
    root_ = root;
    goal_ = goal;
    double x_min = min<double>(goal_.x, root_.x) - 10.0;
    double x_max = max<double>(goal_.x, root_.x) + 10.0;
    double y_min = min<double>(goal_.y, root_.y) - 10.0;
    double y_max = max<double>(goal_.y, root_.y) + 10.0;
    
    vector<node> tree;
    tree.push_back(node(root_));
    int step = 0;
    int sz = 1;
    double goal_dist = dist(root_, goal_);
    int leaf_index = 0;
    while(1){
        step += 1;
        if(step > max_step_) break;

        // sample random point
        double x = x_min + (x_max - x_min) * ((double)rand() / (double) RAND_MAX);
        double y = y_min + (y_max - y_min) * ((double)rand() / (double) RAND_MAX);
        point p  = point(x,y);
        
        // find nearest
        int nearest_idx = find_nearest(tree, p, false);

        // get candidate
        p = get_candidate(tree[nearest_idx].p, p, tau_);

        // check collision
        if (is_collision(tree[nearest_idx].p, p)) continue;

        // append candidate
        node nnode = node(p);
        nnode.parent = nearest_idx;
        nnode.cost = tree[nearest_idx].cost + get_edge_cost(tree[nearest_idx].p, p);
        tree[nearest_idx].childs.push_back(sz);
        tree.push_back(nnode);
        

        // get near nodes for rewiring
        vector<int> near_idx = get_near(tree, p, coeff_tau_ * tau_, false);
        for(int idx : near_idx){
            if(idx == nearest_idx) continue;
            if(idx == sz) continue;
            if(tree[idx].cost < tree[sz].cost + get_edge_cost(tree[idx].p, tree[sz].p)) continue;
            int par = tree[idx].parent;
            if(par!=-1){
                vector<int> n_child;
                for(int k : tree[par].childs){
                    if(k == idx) continue;
                    n_child.push_back(k);
                }
                tree[par].childs = n_child;
            }
            tree[idx].parent = sz;
            double delta_cost = tree[sz].cost + get_edge_cost(tree[idx].p, tree[sz].p) - tree[idx].cost;
            queue<int> q;
            q.push(idx);
            while(!q.empty()){
                int k = q.front();
                q.pop();
                tree[k].cost += delta_cost;
                for(int c: tree[k].childs) q.push(c);
            }
            tree[sz].childs.push_back(idx);
        }
        if(goal_dist > dist(p, goal_)){
            goal_dist = dist(p, goal_);
            leaf_index = sz;
        }
        sz ++;
        if(sz > max_samples_ || goal_dist < goal_threshold_) break;
    }

    vector<point> path;
    while(leaf_index != -1){
        path.push_back(tree[leaf_index].p);
        tree[leaf_index].is_path = true;
        leaf_index = tree[leaf_index].parent;
    }
    
    return path;
}


vector<point> RRT::carrt(point root, point goal){
    clock_t init_time = clock();
    root_ = root;
    goal_ = goal;
    double x_min = min<double>(goal_.x, root_.x) - 10.0;
    double x_max = max<double>(goal_.x, root_.x) + 10.0;
    double y_min = min<double>(goal_.y, root_.y) - 10.0;
    double y_max = max<double>(goal_.y, root_.y) + 10.0;

    vector<node> tree;
    tree.push_back(node(root_));
    tree[0].in_tree = true;
    int step = 0;
    int sz = 1;
    double goal_dist = dist(root_, goal_);
    int leaf_index = 0;
    double rrt_sum = 0.0;
    while(1){
        step += 1;
        if(step > max_step_) break;
        if(double(clock()-init_time)/CLOCKS_PER_SEC > time_limit_) break;

        // sample random point
        double x = x_min + (x_max - x_min) * ((double)rand() / (double) RAND_MAX);
        double y = y_min + (y_max - y_min) * ((double)rand() / (double) RAND_MAX);
        point x_new  = point(x,y);
        
        // find nearest
        int nearest_idx = find_nearest(tree, x_new, true);
        node x_nearest = tree[nearest_idx];

        vector<point> ce_path;
        if(dist(x_nearest.p, x_new) < tau_) x_new = get_candidate(x_nearest.p, x_new, tau_);
        else{
            ce_path = rrt(x_nearest.p, x_new);
            reverse(ce_path.begin(), ce_path.end());
            if(ce_path.size() == 1) continue;
            x_new = ce_path[1];
        }

        // check collision
        if (is_collision(x_nearest.p, x_new)) continue;
        
        // get near 
        vector<int> x_near = get_near(tree, x_new, coeff_tau_ * tau_, false);
        
        // get parent
        int min_index = nearest_idx;
        double min_cost = x_nearest.cost + get_edge_cost(x_nearest.p, x_new);
        for(const int& idx : x_near){
            double cur_cost = tree[idx].cost + get_edge_cost(tree[idx].p, x_new);
            if(cur_cost < min_cost){
                min_cost = cur_cost;
                min_index = idx;
            }
        }

        // push back x_new with connecting x_min, the index of x_new is sz
        node nnode = node(x_new);
        nnode.parent = min_index;
        nnode.cost = min_cost;
        tree.push_back(nnode);
        tree[min_index].childs.push_back(sz);
        if(goal_dist > dist(goal_, tree[sz].p)){
            leaf_index = sz;
            goal_dist = dist(goal_, tree[sz].p);
        }
        if(!tree[min_index].in_tree){
            int cur_index = min_index;
            while(!tree[cur_index].in_tree && cur_index != -1){
                tree[cur_index].in_tree = true;
                if(goal_dist > dist(goal_, tree[cur_index].p)){
                    leaf_index = cur_index;
                    goal_dist = dist(goal_, tree[cur_index].p);
                }
                cur_index = tree[cur_index].parent;
            }
        }
        
        // rewiring
        for(int i = 0; i < x_near.size(); i++){
            int idx = x_near[i];
            if (idx == min_index) continue;
            if (min_cost + get_edge_cost(tree[idx].p, x_new) > tree[idx].cost) continue;
            tree[idx].in_tree = true;
            if(goal_dist > dist(goal_, tree[idx].p)){
                leaf_index = idx;
                goal_dist = dist(goal_, tree[idx].p);
            }
            double delta = min_cost + get_edge_cost(tree[idx].p, x_new) - tree[idx].cost;
            int par_idx = tree[idx].parent;
            if(par_idx != -1){
                vector<int> n_child;
                for(int c : tree[par_idx].childs){
                    if(c == idx) continue;
                    n_child.push_back(c);
                }
                tree[par_idx].childs = n_child;
            }
            tree[idx].parent = sz;
            tree[sz].childs.push_back(idx);
            queue<int> q;
            q.push(idx);
            while(1){
                if(q.empty()) break;
                int cur = q.front();
                q.pop();
                tree[cur].cost += delta;
                for(int c: tree[cur].childs) q.push(c);
            }
        }
        sz++;
        
        if(ce_path.size() > 2){
            for(int i = 2; i < ce_path.size(); i++){
                //plan single tree
                node y_nearest = tree[sz-1];
                int y_nearest_idx = sz-1;
                point y_new = ce_path[i];
                vector<int> y_near = get_near(tree, y_new, coeff_tau_ * tau_, false);
                double y_min_cost = y_nearest.cost + get_edge_cost(y_nearest.p, y_new);
                int y_min_idx = y_nearest_idx;
                for(int j = 0; j < y_near.size(); j++){
                    int cur_idx = y_near[j];
                    if(cur_idx == y_nearest_idx) continue;
                    if(is_collision(tree[cur_idx].p, y_new)) continue;
                    double cur_cost = tree[cur_idx].cost + get_edge_cost(tree[cur_idx].p, y_new);
                    if(cur_cost > y_min_cost) continue;
                    y_min_cost = cur_cost;
                    y_min_idx = cur_idx;
                }
                node nnode = node(y_new);
                nnode.parent = y_min_idx;
                nnode.cost = y_min_cost;
                nnode.in_tree = false;
                tree.push_back(nnode);
                tree[y_min_idx].childs.push_back(sz);
                
                for(int j = 0; j < y_near.size(); j++){
                    int idx = y_near[j];
                    if(idx == y_min_idx) continue;
                    if(is_collision(tree[idx].p, y_new)) continue;
                    double delta = y_min_cost + get_edge_cost(y_new, tree[idx].p) - tree[idx].cost;
                    if(delta > 0) continue;
                    int par_idx = tree[idx].parent;
                    if(par_idx != -1){
                        vector<int> n_child;
                        for(int c : tree[par_idx].childs){
                            if(c == idx) continue;
                            n_child.push_back(c);
                        }
                        tree[par_idx].childs = n_child;
                    }
                    tree[idx].parent = sz;
                    tree[sz].childs.push_back(idx);
                    queue<int> q;
                    q.push(idx);
                    while(1){
                        if(q.empty()) break;
                        int cur = q.front();
                        q.pop();
                        tree[cur].cost += delta;
                        for(int c: tree[cur].childs) q.push(c);
                    }
                }
                sz++;
            }
        }

        if(sz > max_samples_ || goal_dist < goal_threshold_) break;
    }

    vector<point> path;
    while(leaf_index != -1){
        path.push_back(tree[leaf_index].p);
        tree[leaf_index].is_path = true;
        leaf_index = tree[leaf_index].parent;
    }
    // drawing(tree);
    clock_t end_time = clock();
    // cout << "elapsed time: " << double(end_time - init_time) / CLOCKS_PER_SEC << endl;
    return path;
}


vector<vector<point>> RRT::diverse_rrt(point root, point goal, int K){
    vector<vector<point>> rt;
    for(int k = 0; k < K; k++){
        vector<point> path = carrt(root, goal);
        update_path_cost_map(path);
        rt.push_back(path);
    }
    // draw_diverse_path(rt,-1);
    return rt;
}

