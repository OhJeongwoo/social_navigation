import yaml
import torch
from model import YNet
import rospy
import bisect
import math
import tf2_ros
from geometry_msgs.msg import PointStamped, Point
from zed_interfaces.msg import ObjectsStamped, Object
from social_navigation.msg import Trajectory
from social_navigation.srv import TrajectoryPredict, TrajectoryPredictResponse
import cv2

CONFIG_FILE_PATH = 'config/sdd_trajnet.yaml'  # yaml config file containing all the hyperparameters
DATA_FILE_PATH = 'data/ped_traj_sample.json'
IMAGE_FILE_PATH = '../../config/free_space_301_1f.png'
OBS_LEN = 8  # in timesteps
PRED_LEN = 12  # in timesteps
NUM_GOALS = 3  # K_e
NUM_TRAJ = 1  # K_a

sy_ = -0.05
sx_ = 0.05
cy_ = 30.0
cx_ = -59.4

with open(CONFIG_FILE_PATH) as file:
    params = yaml.load(file, Loader=yaml.FullLoader)
experiment_name = CONFIG_FILE_PATH.split('.yaml')[0].split('config/')[1]

model = YNet(obs_len=OBS_LEN, pred_len=PRED_LEN, params=params, image_path=IMAGE_FILE_PATH)
model.load(f'pretrained_models/{experiment_name}_weights.pt')

IS_PREDICTED = 1 # 1 if predicted current PAST_TRAJS, -1 if predicting
PAST_TRAJS = [] # social_navigation/Trajectory[]
PREDICTED_TRAJS = []  # social_navigation/Trajectory[]

def interpolate(new_t, ts, traj):
    # new_t : (ros)time
    # ts : (ros)time[]
    # trajs : geometry_msgs/Point[]
    # return : geometry_msgs/Point
    assert len(ts)==len(traj)
    if len(ts)<2:
        rospy.logwarn("can not interpolate traj with length %d", len(ts))
        new_point = Point()
        if len(ts)>0:
            new_point.x = traj[0].x
            new_point.y = traj[0].y
            new_point.z = traj[0].z
        return new_point
    bii = bisect.bisect(ts, new_t, lo=1, hi=len(ts) - 1)
    alpha = (ts[bii] - new_t).to_sec() / (ts[bii] - ts[bii - 1]).to_sec()
    if alpha > 6.0:
        alpha = 6.0
    elif alpha < -5.0:
        alpha = -5.0
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

def speed(traj, ts):
    dx = traj[-1].x - traj[0].x
    dy = traj[-1].y - traj[0].y
    dt = (ts[-1] - ts[0]).to_sec()
    v = math.hypot(dx, dy)/dt
    return min(v, 1.0)

def trajectory_predict(req):
    # res : TrajectoryPredictResponse()
    # res.trajectories : Trajectory[]
    global PREDICTED_TRAJS
    trajs = PREDICTED_TRAJS.copy()
    new_ts = [rospy.Time.now() + rospy.Duration.from_sec(t) for t in req.times]
    trajectories = [interpolates(new_ts, traj) for traj in trajs]
    res = TrajectoryPredictResponse()
    res.times = req.times
    res.velocity = [speed(traj.trajectory, new_ts) for traj in trajectories]
    res.trajectories = trajectories

    # image = cv2.imread(IMAGE_FILE_PATH, cv2.IMREAD_COLOR)
    # for traj in trajectories:
    #     for point in traj.trajectory:
    #         x = point.x / sx_ + 1278
    #         y = point.y / sy_ + 642
    #         cv2.circle(image, (int(x), int(y)), 2, (0, 0, 255), -1)
    # cv2.imwrite('predicted.png', image)
    return res


def is_tracked(objects, pedestrian_id):
    for object in objects:
        if object.label != "Person":
            continue
        if object.label_id == pedestrian_id and object.tracking_state != 0:
            return True
    return False


def callback(msg):
    # msg : ObjectsStamped
    global PAST_TRAJS
    past_trajs = PAST_TRAJS.copy()
    # remove untracked trajectories
    past_trajs = [traj for traj in past_trajs if is_tracked(msg.objects, traj.pedestrian_id)]
    frame_id = msg.header.frame_id
    for object in msg.objects:
        if object.label != "Person":
            print("not Person")
            continue
        point = Point()
        point.x = object.position[0]
        point.y = object.position[1]
        point.z = object.position[2]
        if frame_id[:3] == "zed":
            try:
                transform = tfBuffer.lookup_transform('map', frame_id, msg.header.stamp)
            except (tf2_ros.LookupException, tf2_ros.ConnectivityException, tf2_ros.ExtrapolationException):
                print("[ynet] tf lookup exception")
                return 0
            x0 = transform.transform.translation.x
            y0 = transform.transform.translation.y
            z0 = transform.transform.translation.z
            qz = transform.transform.rotation.z
            qw = transform.transform.rotation.w
            yaw = math.atan2(2*qw*qz, 1-2*qz*qz)

            x0 += point.x * math.cos(yaw) - point.y * math.sin(yaw)
            y0 += point.x * math.sin(yaw) + point.y * math.cos(yaw)
            z0 += point.z
            point.x, point.y, point.z = x0, y0, z0
        is_new = True
        for traj in past_trajs:
            if traj.pedestrian_id == object.label_id and object.tracking_state != 0:
                is_new = False
                last_point = traj.trajectory[-1]
                last_time = traj.times[-1]
                if math.hypot(last_point.x - point.x, last_point.y - point.y) > \
                        5.0 * (msg.header.stamp - last_time).to_sec():
                    traj.times.clear()
                    traj.trajectory.clear()
                traj.times.append(msg.header.stamp)
                traj.trajectory.append(point)
                break
        if is_new:
            traj = Trajectory()
            traj.pedestrian_id = object.label_id
            traj.times = [msg.header.stamp]
            traj.trajectory = [point]
            past_trajs.append(traj)
    PAST_TRAJS = past_trajs
    global IS_PREDICTED
    IS_PREDICTED = 0


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


def loop():
    rospy.sleep(0.0)
    while rospy.Time.now().to_sec() <= 0.4*OBS_LEN:
        rospy.sleep(0.4)
    while not rospy.is_shutdown():
        rospy.sleep(0.01)
        global IS_PREDICTED
        if IS_PREDICTED != 0:
            continue
        else:
            IS_PREDICTED = -1

        global PAST_TRAJS
        past_trajs = PAST_TRAJS.copy()
        if len(past_trajs) == 0:
            continue
        t0 = rospy.Time.now()
        dt = rospy.Duration.from_sec(0.4)
        input_ts = [t0 + dt * (i + 1 - OBS_LEN) for i in range(OBS_LEN)]
        input_trajs = [interpolates_to_torch(input_ts, past_traj) for past_traj in past_trajs]
        input_trajs = torch.stack(input_trajs, dim=0)
        input_trajs[:, :, 0] = (input_trajs[:, :, 0] - cx_) / sx_ * params["resize"]
        input_trajs[:, :, 1] = (input_trajs[:, :, 1] - cy_) / sy_ * params["resize"]
        _, future_trajs = model.predict(input_trajs, params,
                                        num_goals=NUM_GOALS, num_traj=NUM_TRAJ, device=None)
        future_trajs = future_trajs[0, :, :, :]
        future_trajs[:, :, 0] = future_trajs[:, :, 0] * sx_ / params["resize"] + cx_
        future_trajs[:, :, 1] = future_trajs[:, :, 1] * sy_ / params["resize"] + cy_

        global PREDICTED_TRAJS
        PREDICTED_TRAJS = torch_to_trajs(future_trajs, t0, dt)

        if IS_PREDICTED == -1:
            IS_PREDICTED = 1

if __name__ == "__main__":
    rospy.init_node("ynet")
    rospy.Service('trajectory_predict', TrajectoryPredict, trajectory_predict)
    rospy.Subscriber('/zed2i/zed_node/obj_det/objects', ObjectsStamped, callback, queue_size=2)
    tfBuffer = tf2_ros.Buffer()
    tfListener = tf2_ros.TransformListener(tfBuffer)
    loop()
    rospy.spin()
