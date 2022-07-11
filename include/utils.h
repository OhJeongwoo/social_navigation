#ifndef UTILS_H
#define UTILS_H
#include <iostream>
#include <vector>
#include <set>

#include "point.h"
#include "pixel.h"

using namespace std;

struct node{
    point p;
    int parent = -1;
    vector<int> childs;
    bool is_path = false;
    double cost = 0.0;
    bool in_tree = true;

    node(): p(point(0,0)) {}
    node(point p): p(p) {}  
};

struct Tnode{
    point jackal;
    point goal;
    vector<point> peds;
    double value;
    double reward;
    double weight;
    int n_visit;
    bool is_leaf;
    int parent;
    vector<int> childs;
    int depth;

    Tnode(): value(0.0), reward(0.0), weight(0.0), n_visit(0), is_leaf(true), parent(-1) {}
};

double norm(point p);
double dist(point p, point q);
point normalize(point p);
point interpolate(point p, point q, double alpha);
int find_nearest(const vector<node>& tree, point p, bool option);
point get_candidate(point p, point q, double d);
vector<int> get_near(const vector<node>& tree, point p, double d, bool option);
vector<point> get_next_pedestrians(point robot, vector<point> peds, vector<point> goal, vector<double> velocity);

#endif