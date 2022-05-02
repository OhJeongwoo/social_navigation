import rospkg
import pickle
import json
import numpy as np
import matplotlib.pyplot as plt
import random
from PIL import Image


traj_file = rospkg.RosPack().get_path("social_navigation") + "/config/sample_path_full.pkl"
save_file = rospkg.RosPack().get_path("social_navigation") + "/config/ped_traj_candidate.json"
free_file = rospkg.RosPack().get_path("social_navigation") + "/config/collision_301_1f.png"

im = Image.open(free_file)

with open(traj_file, 'rb') as file:
    data = pickle.load(file)

sy_ = -0.05
sx_ = 0.05
cy_ = 30.0
cx_ = -59.4

def pixel2xy(p):
    x = p[0] * sx_ + cx_
    y = p[1] * sy_ + cy_
    return [x,y]

def xy2pixel(p):
    px = int((p[0]-cx_)/sx_)
    py = int((p[1]-cy_)/sy_)

def is_valid(p):
    px,py = xy2pixel(p)
    if im.getpixel((px,py)) == 255:
        return true
    return false

def transform_traj(traj):
    rt = []
    for p in traj:
        rt.append(pixel2xy(p))
    return np.array(rt)

waypoints_ = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                    [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
n_waypoints_ = len(waypoints_)

st_cnt = [0 for i in range(n_waypoints_)]
en_cnt = [0 for i in range(n_waypoints_)]
adj_mat = [[0 for i in range(n_waypoints_)] for j in range(n_waypoints_)]

rt = []
rx = []
ry = []
rc = []
for traj in data:
    cand = {}
    t = traj.shape[0]
    check = True
    for i in range(t):
        if im.getpixel((traj[i][0], traj[i][1])) == 0:
            check = False
            break
    if not check:
        continue
    transformed_traj = transform_traj(traj)
    if t > 100:
        continue
    color = random.randint(0,10)
    for i in range(t):
        rx.append(transformed_traj[i][0])
        ry.append(transformed_traj[i][1])
        rc.append(color)
    x = transformed_traj[0][0]
    y = transformed_traj[0][1]
    cand['time'] = (t-1) * 0.4
    cand['interval'] = 0.4
    cand['waypoints'] = transformed_traj.tolist()
    min_d = 10
    min_idx = -1
    for i, p in enumerate(waypoints_):
        if min_d > ((x - p[0]) ** 2 + (y - p[1]) ** 2) ** 0.5:
            min_d = ((x - p[0]) ** 2 + (y - p[1]) ** 2) ** 0.5
            min_idx = i
    cand['start'] = min_idx
    st_cnt[min_idx] += 1

    x = transformed_traj[t-1][0]
    y = transformed_traj[t-1][1]
    
    min_d = 10
    min_idx = -1
    for i, p in enumerate(waypoints_):
        if min_d > ((x - p[0]) ** 2 + (y - p[1]) ** 2) ** 0.5:
            min_d = ((x - p[0]) ** 2 + (y - p[1]) ** 2) ** 0.5
            min_idx = i
    cand['end'] = min_idx
    en_cnt[min_idx] += 1
    adj_mat[cand['start']][cand['end']] += 1
    rt.append(cand)
with open(save_file, 'w') as jf:
    json.dump(rt,jf,indent=4)


print(st_cnt)
print(en_cnt)
print(np.array(adj_mat))

plt.scatter(rx, ry, c=rc, s=0.2)
plt.show()