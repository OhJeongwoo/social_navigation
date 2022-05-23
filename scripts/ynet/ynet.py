import yaml
import torch
from model import YNet
import numpy as np
import rospy
import bisect
from geometry_msgs.msg import Point
from social_navigation.msg import Trajectory, TrajectoryArray
from social_navigation.srv import TrajectoryPredict, TrajectoryPredictResponse

def interpolate(new_t, ts, traj):
    # new_t : (ros)time
    # ts : (ros)time[]
    # datas : geometry_msgs/Point[]
    # return : geometry_msgs/Point
    if len(ts)<2:
        rospy.logerr("can not interpolate traj with length %d", len(ts))
        return Point()
    bii = bisect.bisect(ts, new_t, lo=1, hi=len(ts) - 1)
    alpha = (ts[bii] - new_t).to_sec() / (ts[bii] - ts[bii - 1]).to_sec()
    new_point = Point()
    new_point.x = traj[bii - 1].x * alpha + traj[bii].x * (1 - alpha)
    new_point.y = traj[bii - 1].y * alpha + traj[bii].y * (1 - alpha)
    new_point.z = traj[bii - 1].z * alpha + traj[bii].z * (1 - alpha)
    return new_point


def interpolates(new_ts, traj):
    # new_ts : (ros)time[]
    # datas : social_navigation/Trajectory
    # return : social_navigation/Trajectory
    new_traj = Trajectory()
    new_traj.trajectory = [interpolate(new_t, traj.times, traj.trajectory) for new_t in new_ts]
    new_traj.pedestrian_id = traj.pedestrian_id
    new_traj.times = new_ts
    return new_traj


def interpolate_to_torch(new_t, ts, traj):
    # return : (2,)
    new_point = interpolate(new_t, ts, traj)
    return torch.tensor([new_point.x, new_point.y])


def interpolates_to_torch(new_ts, traj):
    # return : (num_timestep, 2)
    new_traj = [interpolate_to_torch(new_t, traj.times, traj.trajectory) for new_t in new_ts]
    return torch.stack(new_traj, dim=0)


trajs = []  # social_navigation/Trajectory[]
CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATA_FILE_PATH = 'data/ped_traj_sample.json'
IMAGE_FILE_PATH = '../../config/free_space_301_1f.png'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 3  # K_e
NUM_TRAJ = 1  # K_a

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params, image_path=IMAGE_FILE_PATH)
model.load(f'pretrained_models/{experiment_name}_weights.pt')


def trajectory_predict(req):
    # res : TrajectoryPredictResponse()
    # res.trajectories : Trajectory[]
    global trajs
    new_ts = req.time_sequence
    trajectories = [interpolates(new_ts, traj) for traj in trajs]
    res = TrajectoryPredictResponse(trajectories)
    return res

def torch_to_trajs(torch_trajs, t0, dt):
    # torch_trajs : (num_ped, PRED_LEN, 2)
    trajs = []
    for torch_traj in torch_trajs:
        traj = Trajectory()
        for torch_point in torch_traj:
            point = Point()
            point.x = torch_point[0]
            point.y = torch_point[1]
            traj.trajectory.append(point)
        traj.times = [t0 + dt * (i+1) for i in range(PRED_LEN)]
        traj.pedestrian_id = -1 # TODO
        trajs.append(traj)
    return trajs

def callback(msg):
    global trajs
    prev_trajs = msg.trajectories
    t0 = rospy.Time.now()
    dt = rospy.Duration.from_sec(0.4)
    input_ts = [t0 + dt * (i + 1 - OBS_LEN) for i in range(OBS_LEN)]
    input_trajs = [interpolates_to_torch(input_ts, prev_traj) for prev_traj in prev_trajs]
    input_trajs = torch.stack(input_trajs, dim=0)
    # TODO: transform
    _, future_trajs = model.predict(input_trajs, params,
                                    num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None)
    future_trajs = future_trajs[0,:,:]
    # TODO: transform
    trajs = torch_to_trajs(future_trajs, t0, dt)

if __name__ == "__main__":
    rospy.init_node("ynet")
    rospy.Service('trajectory_predict', TrajectoryPredict, trajectory_predict)
    rospy.Subscriber('trajectories', TrajectoryArray, callback)
    rospy.spin()