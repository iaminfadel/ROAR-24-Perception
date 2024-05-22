#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import String

label_with_coordinates = {'5': (406, 46), '3': (406, 407), '8': (165, 407), '1': (167, 45)}

def publish_dictionary():
    rospy.init_node('dictionary_publisher', anonymous=True)
    label_pub = rospy.Publisher('label_topic', String, queue_size=10)
    point_pub = rospy.Publisher('point_topic', Point, queue_size=10)
    rate = rospy.Rate(10) # 10hz
    
    while not rospy.is_shutdown():
        for label, coordinate in label_with_coordinates.items():
            # Publish label
            label_pub.publish(label)
            rospy.loginfo("Published label: %s", label)

            # Publish coordinates as Point message
            point_msg = Point()
            point_msg.x = coordinate[0]
            point_msg.y = coordinate[1]
            point_msg.z = 0.0  # Assuming z-coordinate is 0
            point_pub.publish(point_msg)
            rospy.loginfo("Published coordinates: x=%f, y=%f, z=%f", point_msg.x, point_msg.y, point_msg.z)

            rate.sleep()

if __name__ == '__main__':
    try:
        publish_dictionary()
    except rospy.ROSInterruptException:
        pass
