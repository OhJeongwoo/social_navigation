import pandas as pd
import yaml
import argparse
import torch
from model import YNet
import numpy as np
import cv2
import json
import pickle
import time

CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATA_FILE_PATH = 'data/ped_traj_sample.json'
IMAGE_FILE_PATH = '../../config/free_space_301_1f.png'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 20  # K_e
NUM_TRAJ = 1  # K_a

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]
print(params)

## load starting data
with open(DATA_FILE_PATH, 'r') as jf:
    data = json.load(jf)
n_traj = len(data)
traj_data = []
for i in range(n_traj):
    # print("the length of %d-th traj: %d" %(i, len(data[i])))
    if len(data[i])<(OBS_LEN*2):
        break
    traj_data.append(data[i][:(OBS_LEN*2):2])
traj_data = np.array(traj_data)
print(traj_data.shape)
checkpoint = [[-28.44, 3.90], [-29.38, -5.60], [-21.05, -1.17], [-20.57, -7.67], [-13.99, -1.62],
                           [-6.55, -2.20], [8.14, -2.49], [15.19, -7.38], [29.93, -7.62], [28.73, 15.92]]
checkpoint = np.array(checkpoint)
print(checkpoint.shape)

## convert to pixel
sy_ = -0.05
sx_ = 0.05
cy_ = 30.0
cx_ = -59.4
traj_data[:,:,0] = (traj_data[:,:,0] - cx_)/sx_*params["resize"]
traj_data[:,:,1] = (traj_data[:,:,1] - cy_)/sy_*params["resize"]
checkpoint[:,0] = (checkpoint[:,0] - cx_)/sx_*params["resize"]
checkpoint[:,1] = (checkpoint[:,1] - cy_)/sy_*params["resize"]
print(checkpoint)

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params, image_path=IMAGE_FILE_PATH)
model.load(f'pretrained_models/{experiment_name}_weights.pt')

trajs = []
ends = []
for i_traj in range(traj_data.shape[0]):
    time_start = time.time()
    time_count = 0
    for default_i in range(NUM_GOALS):
        traj = traj_data[i_traj:i_traj+1]
        ended = False
        while not ended:
            waypoint_samples, future_samples = model.predict(traj[:,-OBS_LEN:,:], params,
                           num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None)
            time_count += 1
            # if time_count>50:
            #     break
            for i in range(NUM_GOALS):
                for chk in checkpoint:
                    way = future_samples.cpu().detach().numpy()[i,0,-1,:]
                    if np.linalg.norm(chk-way) < 0.6/sx_*params["resize"]:
                        traj = np.concatenate([traj, future_samples.cpu().detach().numpy()[i, :, :, :]], axis=1)
                        ends.append(chk)
                        ended = True
                        break
                if ended:
                    break
            if ended:
                break
            traj = np.concatenate([traj,future_samples.cpu().detach().numpy()[default_i,:,:,:]],axis=1)
        trajs.append(np.reshape(traj,(-1,2))/params["resize"])
        print("--", default_i)
    print("-----", i_traj)
    # print("time :", (time.time() - time_start) / time_count)
    
image = cv2.imread(IMAGE_FILE_PATH)
for i in trajs[-1]:
    image = cv2.circle(image, (int(i[0]),int(i[1])), 2, (0,0,255), -1)
cv2.imshow('image',image)
cv2.waitKey(1000000)

with open('sample_path_full.pkl', 'wb') as file:
    pickle.dump(trajs, file)

with open('sample_path_ends.pkl', 'wb') as file:
    pickle.dump(ends, file)