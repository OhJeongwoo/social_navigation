import json
import joblib
import numpy as np



traj_dict = joblib.load('goal.gz')
N = len(traj_dict.keys())

cx = 18.56
cy = 27.20
tx = 1.02031 + 1.0
ty = 25.2413 - 2.5

save_dict = []
for seq in range(N):
    traj = {}
    traj['spawn'] = [cx-traj_dict[seq]['im_world'][0][0]+tx, traj_dict[seq]['im_world'][0][1]-cy+ty]
    traj['goal'] = [cx-traj_dict[seq]['im_world'][1][0]+tx, traj_dict[seq]['im_world'][1][1]-cy+ty]
    save_dict.append(traj)

with open('goal.json', 'w') as jf:
    json.dump(save_dict, jf, indent=4)