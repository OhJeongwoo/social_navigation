import torch
import rospy
from geometry_msgs.msg import Point
from social_navigation.msg import Trajectory, TrajectoryArray

PRED_LEN = 12  # in timesteps

def torch_to_trajs(torch_trajs, t0, dt):
    # torch_trajs : (num_ped, PRED_LEN, 2)
    trajs = TrajectoryArray()
    for torch_traj in torch_trajs:
        traj = Trajectory()
        for torch_point in torch_traj:
            point = Point()
            point.x = torch_point[0]
            point.y = torch_point[1]
            traj.trajectory.append(point)
        traj.times = [t0 + dt * (i+1-torch_trajs.size(dim=1)) for i in range(torch_trajs.size(dim=1))]
        traj.pedestrian_id = -1 # TODO
        trajs.trajectories.append(traj)
    return trajs

if __name__ == "__main__":
    rospy.init_node("ynet_sub_test")
    pub = rospy.Publisher('trajectories', TrajectoryArray, queue_size=2)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        trajs = torch.ones((4,8,2))
        t0 = rospy.Time.now()
        dt = rospy.Duration.from_sec(0.4)
        t0 = t0 - dt * 15
        trajs = torch_to_trajs(trajs, t0, dt)
        pub.publish(trajs)
        rate.sleep()