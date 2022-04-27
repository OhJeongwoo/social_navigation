import json
import numpy as np

with open("ped_traj_sample.json", 'r') as jf:
    data = json.load(jf)

n_traj = len(data)

lookahead = 8
traj_data = []
for i in range(n_traj):
    print("the length of %d-th traj: %d" %(i, len(data[i])))
    traj_data.append(data[i][:8])

traj_data = np.array(traj_data)
print(traj_data)
print(traj_data.shape)