import os

import numpy as np
from matplotlib.pyplot import get
from lxml import etree
from lxml.etree import Element
import math
import random
from geometry_msgs.msg import Pose, Point, Quaternion

from collections import namedtuple

WAIT = 0
INIT = 1
MOVE = 2

EPS = 1e-6
acc_kp = 1.0
acc_kd = 1.0
str_kp = 1.0
str_kd = 1.0

action_list = [[0.0, -1], [0.0, -0.5], [0.0, 0.0], [0.0, 0.5], [0.0, 1.0]
             , [0.5, -1], [0.5, -0.5], [0.5, 0.0], [0.5, 0.5], [0.5, 1.0]
             , [1.0, -1], [1.0, -0.5], [1.0, 0.0], [1.0, 0.5], [1.0, 1.0]]
STOP = 2

def check_path(path_name):
    if not os.path.exists(path_name):
        os.mkdir(path_name)

def y2q(yaw):
  cy = math.cos(0.5 * yaw)
  sy = math.sin(0.5 * yaw)
  return Quaternion(0.0, 0.0, sy, cy)

def L2dist(A, B):
    return math.sqrt((A.x - B.x) ** 2 + (A.y - B.y) ** 2)

def norm_2d(p):
    return (p.x ** 2 + p.y ** 2) ** 0.5

def get_length(A, B):
    return math.sqrt((A[0] - B[0]) ** 2 + (A[1] - B[1]) ** 2)

def arr2str(arr):
    rt = ""
    for x in arr:
        rt += str(x) + " "
    return rt

def transform_wall_info(map_dict):
    vertices = map_dict["vertices"]
    material = map_dict["material"]
    thickness = map_dict["thickness"]
    height = map_dict["height"]
    
    rt = []
    for vertex_set in vertices:
        N = len(vertex_set)
        for seq in range(N):
            wall_info = {}
            A = vertex_set[seq]
            B = vertex_set[(seq+1)%N]
            wall_info['center'] = [(A[0]+B[0])/2, (A[1]+B[1])/2, height/2]
            wall_info['A'] = vertex_set[seq]
            wall_info['B'] = vertex_set[(seq+1)%N]
            wall_info['yaw'] = math.atan2(B[1]-A[1], B[0]-A[0])
            wall_info['length'] = get_length(A,B)
            wall_info['thickness'] = thickness
            wall_info['height'] = height
            wall_info['material'] = material
            wall_info['name'] = 'wall_' + str(seq).zfill(3)
            rt.append(wall_info)
    return rt

def transform_actor_info(act_dict):
    init_position = act_dict['init_position']
    target_weight = act_dict['target_weight']
    obstacle_weight = act_dict['obstacle_weight']
    animation_factor = act_dict['animation_factor']
    N = len(init_position)
    rt = []
    for seq in range(N):
        actor_info = {}
        actor_info['name'] = 'actor_' + str(seq).zfill(3)
        actor_info['x'] = init_position[seq][0]
        actor_info['y'] = init_position[seq][1]
        actor_info['z'] = 1.25
        actor_info['target_weight'] = target_weight
        actor_info['obstacle_weight'] = obstacle_weight
        actor_info['animation_factor'] = animation_factor
        rt.append(actor_info)
    return rt

def get_element(name, value):
    rt = Element(name)
    rt.text = value
    return rt

def get_wall(wall_info):
    wall = Element("link", name=wall_info['name'])
    wall_collision = Element("collision", name=wall_info['name'] + '_collision')
    wall_visual = Element("visual", name=wall_info['name'] + '_visual')

    geometry_collision = Element("geometry")
    box_collision = Element("box")
    box_collision.append(get_element("size", str(wall_info['length'])+ " " + str(wall_info['thickness']) + " " + str(wall_info['height'])))
    geometry_collision.append(box_collision)

    geometry_visual = Element("geometry")
    box_visual = Element("box")
    box_visual.append(get_element("size", str(wall_info['length'])+ " " + str(wall_info['thickness']) + " " + str(wall_info['height'])))
    geometry_visual.append(box_visual)

    material = Element("material")
    script = Element("script")
    script.append(get_element("uri", "file://media/materials/scripts/gazebo.material"))
    script.append(get_element("name", wall_info['material']))
    material.append(script)
    material.append(get_element("ambient", "1 1 1 1"))

    zero_pose_collision = Element("pose", frame='')
    zero_pose_collision.text = "0 0 0 0 0 0"

    zero_pose_visual = Element("pose", frame='')
    zero_pose_visual.text = "0 0 0 0 0 0"

    meta = Element("meta")
    meta.append(get_element("layer", str(0)))

    wall_collision.append(geometry_collision)
    wall_collision.append(zero_pose_collision)
    wall_collision.append(get_element("max_contacts", str(10)))

    wall_visual.append(geometry_visual)
    wall_visual.append(zero_pose_visual)
    wall_visual.append(material)
    wall_visual.append(meta)


    wall.append(wall_collision)
    wall.append(wall_visual)

    pose = Element("pose", frame='')
    pose.text = str(wall_info['center'][0]) + " " + str(wall_info['center'][1]) + " " + str(wall_info['center'][2]) + " 0 0 " + str(wall_info['yaw'])
    wall.append(pose)

    wall.append(get_element('self_collide', str(0)))
    wall.append(get_element('enable_wind', str(0)))
    wall.append(get_element('kinematic', str(0)))

    return wall

def get_actor(actor_info):
    actor = Element("actor", name=actor_info['name'])

    actor.append(get_element("pose", arr2str([actor_info['x'], actor_info['y'], actor_info['z'], 0, 0, 0])))
    
    skin = Element("skin")
    skin.append(get_element("filename", "walk.dae"))
    skin.append(get_element("scale", "1.0"))
    actor.append(skin)

    animation = Element("animation", name="walking")
    animation.append(get_element("filename", "walk.dae"))
    animation.append(get_element("scale", "1.0"))
    animation.append(get_element("interpolate_x", "true"))
    actor.append(animation)

    plugin = Element("plugin", name=actor_info['name'], filename="libHumanPlugin.so")
    plugin.append(get_element("model_name", actor_info['name']))
    plugin.append(get_element("x", str(actor_info['x'])))
    plugin.append(get_element("y", str(actor_info['y'])))
    plugin.append(get_element("target_weight", str(actor_info['target_weight'])))
    plugin.append(get_element("obstacle_weight", str(actor_info['obstacle_weight'])))
    plugin.append(get_element("animation_factor", str(actor_info['animation_factor'])))

    ignore_obstacles = Element("ignore_obstacles")
    ignore_obstacles.append(get_element("model", "ground_plane"))
    plugin.append(ignore_obstacles)

    actor.append(plugin)

    return actor
    

def interpolate(A,B,alpha):
    return [A[0] * (1-alpha) + B[0] * alpha, A[1] * (1-alpha) + B[1] * alpha]

def transform_coordinate(x, y, ct, st):
    return x * ct + y * st, -x * st + y * ct

def collision_cost(d):
    return 1.0 / (1.0 + math.exp(10 * (d-0.6)))


def purepursuit(pose, goal, v, prv_acc_err =None, prv_str_err=None):
    siny_cosp = 2 * (pose.orientation.w * pose.orientation.z + pose.orientation.x * pose.orientation.y)
    cosy_cosp = 1 - 2 * (pose.orientation.y * pose.orientation.y + pose.orientation.z * pose.orientation.z)


    yaw = np.arctan2(siny_cosp, cosy_cosp)

    dt = 0.1
    dy = goal.y - pose.position.y
    dx = goal.x - pose.position.x
    theta = np.arctan2(dy, dx)
    cur_err = theta - yaw
    while(cur_err > np.pi):
        cur_err -= 2 * np.pi
    while(cur_err < -np.pi):
        cur_err += 2 * np.pi
    
    if prv_str_err is not None:
        if cur_err - prv_str_err > 0.01:
            str_val = str_kp * cur_err
        else:
            str_val = str_kp * cur_err + str_kd * (cur_err - prv_str_err) / dt
        # print("sc: %.3f, scd: %.3f" %(cur_err, (cur_err - prv_str_err) / dt))
    else:
        str_val = str_kp * cur_err
    
    target_str = np.clip(str_val, -1.0, 1.0)
    target_vel = np.clip(v / (1 + 4.0 * abs(str_val)), 0.0, 1.0)

    cmd = [target_vel, target_str]

    return cmd

def get_similar_action(a):
    min_d = get_length(action_list[0], a)
    rt = 0
    for i, act in enumerate(action_list):
        d = get_length(act, a)
        if min_d > d:
            min_d = d
            rt = i
    return [rt]


def make_group(N, max_n=3):
    rt = []
    while N > 0:
        n = min(random.randint(1, max_n), N)
        rt.append(n)
        N -= n
    return rt


def set_relative_pose(N, grid=0.5):
    rt = []
    for _ in range(N):
        while True:
            x = random.uniform(-grid,grid)
            y = random.uniform(-grid,grid)
            if len(rt) == 0:
                break
            q = Point(x,y,0.0)
            dist_list = []
            for p in rt:
                dist_list.append(L2dist(p,q))
            if min(dist_list) > 0.3:
                break
        rt.append(Point(x,y,0.0))
    return rt


def pedestrian_controller(peds, goals, jackal=None):
    N = len(peds)
    rt = {}
    rt_v = {}
    c1 = 2.0 # for goal
    c2 = 0.2 # for social force
    actor_name_ = peds.keys()
    for name in actor_name_:
        ped = peds[name]
        g_id = ped['group']
        ppos = ped['pos']
        rpos = ped['rpos']
        goal = goals[g_id]
        if goal is None:
            continue
        g = Point(rpos.x + goal.x, rpos.y + goal.y, 0.0)
        r = Point(g.x - ppos.x, g.y - ppos.y, 0.0)
        v = (r.x ** 2 + r.y ** 2) ** 0.5
        if v < 0.1:
            continue
        flag = False
        if v < 1.0:
            flag = True
        r = Point(r.x / v * c1, r.y / v * c1, 0.0)
        v = max(min(1.0, v), 0.3)
        f = Point(0.0, 0.0, 0.0)
        for q_name in actor_name_:
            q = peds[q_name]
            if q['group'] == g_id:
                continue
            d = L2dist(ppos, q['pos'])
            if d > 3.0:
                continue
            c = 3.0 - d
            u = Point(ppos.x - q['pos'].x, ppos.y - q['pos'].y, 0.0)
            nu = norm_2d(u)
            if nu < 0.1:
                continue
            f.x += u.x / nu * c * c2
            f.y += u.y / nu * c * c2
        d = L2dist(ppos, jackal)
        if d < 3.0:
            c = 3.0 - d
            u = Point(ppos.x - jackal.x, ppos.y - jackal.y, 0.0)
            nu = norm_2d(u)
            if nu > 0.01:
                f.x += 3.0 * u.x / nu * c * c2
                f.y += 3.0 * u.y / nu * c * c2
        if flag and (f.x > 0.1 or f.y > 0.1):
            v = 0.2
            f.x = 0.0
            f.y = 0.0
        # dir = Point(r.x + f.x, r.y + f.y, 0.0)
        # if norm_2d(dir) < 0.01:
        #     continue
        # dir = Point(ppos.x + v * dir.x / norm_2d(dir), ppos.y + v * dir.y / norm_2d(dir), 0.0)
        dir = Point(r.x + f.x, r.y + f.y, 0.0)
        dir_norm = norm_2d(dir)
        dir = Point(dir.x / dir_norm, dir.y / dir_norm, 0.0)
        rt_v[name] = v
        rt[name] = Point(ppos.x + dir.x * v, ppos.y + dir.y * v, 0.0)
        # dir = Point(g.x + f.x, g.y + f.y, 0.0)
        # rt[name] = dir
        # rt_v[name] = min(L2dist(dir, ppos), 1.0)
    return rt, rt_v
    # for ped in peds:
    #     g_id = ped['group']
    #     g_goal = goal[g_id]
    #     p_pos = ped['pos']
    #     goal = Point(ped['rpos'].x + g_goal.x, ped['rpos'].y + g_goal.y, 0.0)
    #     g_v = Point(goal.x - p_pos.x, goal.y - p_pos.y, 0.0)
    #     vel = (g_v.x ** 2 + g_v.y ** 2) ** 0.5
    #     g_v = Point(g_v.x / vel, g_v.y / vel, 0.0)
    #     s_v = Point(0.0, 0.0, 0.0)
    #     for q in peds:
    #         if q['group'] == g_id:
    #             continue
    #         d = L2dist(p_pos, q['pos'])


