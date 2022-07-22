#include "utils.h"
#include <iostream>
#include "point.h"
#include "pixel.h"

#include "cv_bridge/cv_bridge.h"
#include "opencv2/opencv.hpp"

using namespace std;
using namespace cv;

using namespace std;

double V = 1.0;
double alpha_goal_ = 1.0;
double alpha_robot_ = 0.1;
double dt = 0.2;


double norm(point p){return sqrt(p.x*p.x + p.y*p.y);}
double dist(point p, point q){return norm(p-q);}
point normalize(point p){return p * (1/norm(p));}
point interpolate(point p, point q, double alpha){return p*alpha+q*(1-alpha);}


int find_nearest(const vector<node>& tree, point p, bool option){
    int sz = tree.size();
    if(sz == 0) return -1;
    int rt = 0;
    double d = dist(tree[0].p, p);
    for(int i = 1;i<sz;i++){
        if(!tree[i].in_tree && option) continue;
        double cd = dist(tree[i].p, p);
        if(cd < d){
            rt = i;
            d = cd;
        }
    }
    return rt;
}


point get_candidate(point p, point q, double d){
    return p + (q-p)*(d/dist(p,q));
}


vector<int> get_near(const vector<node>& tree, point p, double d, bool option){
    vector<int> rt;
    int sz = tree.size();
    for(int i = 0; i < sz; i++){
        if(!tree[i].in_tree && option) continue;
        if(dist(tree[i].p, p) < d) rt.push_back(i);
    }
    return rt;
}


vector<point> get_next_pedestrians(point robot, vector<point> peds, vector<point> goal, vector<double> velocity){
    if(peds.size() != goal.size()) cout << "ERROR! number of pedestrians are different." << endl;

    vector<point> rt;
    int n = peds.size();
    for(int i = 0; i < n; i++){
        point vg = goal[i] - peds[i];
        vg = vg * (alpha_goal_ / norm(vg));
        point vr = peds[i] - robot;
        vr = vr * (alpha_robot_ / max(norm(vr), 1.0));
        point v = normalize(vg + vr) * velocity[i] * dt;
        rt.push_back(peds[i] + v);
    }
    return rt;
}

