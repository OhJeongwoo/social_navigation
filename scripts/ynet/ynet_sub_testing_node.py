import rospy
from zed_interfaces.msg import ObjectsStamped, Object

if __name__ == "__main__":
    rospy.init_node("ynet_sub_test")
    pub = rospy.Publisher('objects', ObjectsStamped, queue_size=2)
    rate = rospy.Rate(30)
    while not rospy.is_shutdown():
        object = Object()
        object.tracking_state = 1
        objects = ObjectsStamped()
        objects.header.stamp = rospy.Time.now()
        objects.objects = [object, object, object, object]
        pub.publish(objects)
        rate.sleep()
