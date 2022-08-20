import rospy
import math
from geometry_msgs.msg import PoseWithCovarianceStamped
from zed_interfaces.srv import set_pose, reset_odometry

def callback(msg):
    # msg : PoseWithCovarianceStamped
    x = msg.pose.pose.position.x
    y = msg.pose.pose.position.y
    z = msg.pose.pose.position.z
    qz = msg.pose.pose.orientation.z
    qw = msg.pose.pose.orientation.w

    R = 0
    P = 0
    Y = math.atan2(2*qw*qz, 1-2*qz*qz)

    rospy.wait_for_service('set_pose')
    try:
        set_pose_proxy = rospy.ServiceProxy('set_pose', set_pose)
        set_pose_proxy(x, y, z, R, P, Y)
    except rospy.ServiceException as e:
        print("Service call failed: %s" % e)

if __name__ == "__main__":
    rospy.init_node("zed_pos_setter")
    rospy.Subscriber('/acml_pose', PoseWithCovarianceStamped, callback, queue_size=2)
    rospy.spin()
