import torch
import rospy
from geometry_msgs.msg import Point
from social_navigation.srv import TrajectoryPredict, TrajectoryPredictRequest

PRED_LEN = 12  # in timesteps

if __name__ == "__main__":
    rospy.init_node("ynet_srv_test")
    rate = rospy.Rate(10)
    while not rospy.is_shutdown():
        rospy.wait_for_service('trajectory_predict')
        t0 = rospy.Time.now()
        tp = TrajectoryPredictRequest()
        tp.times = [0.0 + 0.4 * i for i in range(45)]

        try:
            sp = rospy.ServiceProxy('trajectory_predict', TrajectoryPredict)
            res = sp(tp)
            print("Success", (rospy.Time.now()-t0).to_sec())
            try:
                print(res.trajectories)
            except:
                print("exception")
        except rospy.ServiceException as e:
            print("Service call failed: %s" % e)
        rate.sleep()