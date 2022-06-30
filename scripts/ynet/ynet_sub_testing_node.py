import rospy
from zed_interfaces.msg import ObjectsStamped, Object

if __name__ == "__main__":
    rospy.init_node("ynet_sub_test")
    pub = rospy.Publisher('objects', ObjectsStamped, queue_size=2)
    rate = rospy.Rate(30)
    print("check")
    while not rospy.is_shutdown():
        object = Object()
        object.tracking_state = 1
        object.position = [-28.44, 3.90, 0.]

        object2 = Object()
        object2.tracking_state = 0

        object3 = Object()
        object3.tracking_state = 1
        object3.position = [28.73, 15.92, 0.]

        object4 = Object()
        object4.tracking_state = 0

        objects = ObjectsStamped()
        objects.header.stamp = rospy.Time.now()
        objects.objects = [object, object2, object3, object4]
        pub.publish(objects)
        rate.sleep()
