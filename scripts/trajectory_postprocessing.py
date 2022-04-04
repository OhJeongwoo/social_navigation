import json
import joblib
import numpy as np



traj_dict = joblib.load('subgoal_dat.gz')
N = len(traj_dict.keys())

cx = 18.56
cy = 27.20
tx = 1.02031 + 1.0
ty = 25.2413 - 2.5

save_dict = []
for seq in range(N):
    L = len(traj_dict[seq]['im_world'])
    traj = {}
    traj['interval'] = 1.0
    traj['time'] = (L-1) * traj['interval']
    waypoints = []
    for i in range(L):
        print(traj_dict[seq]['im_world'][i])
        waypoints.append([cx-traj_dict[seq]['im_world'][i][0]+tx, traj_dict[seq]['im_world'][i][1]-cy+ty])
    traj['waypoints'] = waypoints
    save_dict.append(traj)

with open('traj.json', 'w') as jf:
    json.dump(save_dict, jf, indent=4)