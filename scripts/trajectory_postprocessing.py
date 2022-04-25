import json
import joblib
import numpy as np



traj_dict = joblib.load('subgoal_dat.gz')
N = len(traj_dict.keys())


#cx = 18.56
#cy = 27.20
#tx = 1.02031 + 1.0
#ty = 25.2413 - 2.5

cx = -59.361
tx = 2.494
cy = 30.199
ty = -2.517

save_dict = []
for seq in range(0, N):
    L = len(traj_dict[seq]['im_world'])
    traj = {}
    traj['interval'] = 1.0
    traj['time'] = (L-1) * traj['interval']
    waypoints = []
    for i in range(L):
        print(traj_dict[seq]['im_world'][i])
        #waypoints.append([cx-traj_dict[seq]['im_world'][i][0]+tx, traj_dict[seq]['im_world'][i][1]-cy+ty])
        waypoints.append([traj_dict[seq]['im_world'][i][0] *tx + cx,traj_dict[seq]['im_world'][i][1] * ty + cy])
        #waypoints.append([5.0, 0.0])
        print(waypoints[-1])
    traj['waypoints'] = waypoints
    save_dict.append(traj)

with open('traj.json', 'w') as jf:
    json.dump(save_dict, jf, indent=4)
