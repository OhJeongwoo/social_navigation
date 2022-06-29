import json
import joblib
import numpy as np



traj_dict = joblib.load('goal_mcts_example.gz')
N = len(traj_dict.keys())
import ipdb;ipdb.set_trace()
cx = -59.361
tx = 2.494
cy = 30.199
ty = -2.517

save_dict = []
for seq in range(N):
    traj = {}
    
    traj['spawn'] = [traj_dict[seq]['im_world'][0][0]*tx+cx, traj_dict[seq]['im_world'][0][1]*ty+cy]
    traj['goal'] = [traj_dict[seq]['im_world'][1][0]*tx+cx, traj_dict[seq]['im_world'][1][1]*ty+cy]
    print(traj)
    save_dict.append(traj)

with open('goal_mcts_example.json', 'w') as jf:
    json.dump(save_dict, jf, indent=4)
