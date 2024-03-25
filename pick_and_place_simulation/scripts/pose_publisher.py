#!/usr/bin/env python

import rospy
from geometry_msgs.msg import PoseStamped

def publish_desired_pose():
    rospy.init_node('desired_pose_publisher', anonymous=True)
    desired_pose_pub = rospy.Publisher('/desired_pose', PoseStamped, queue_size=10)
    rate = rospy.Rate(1)  # 1 Hz

    # Create a PoseStamped message and fill in the desired pose
    desired_pose_msg = PoseStamped()
    desired_pose_msg.header.frame_id = "base_link"
    desired_pose_msg.pose.position.x = 0.4  # Example pose values
    desired_pose_msg.pose.position.y = 0.1
    desired_pose_msg.pose.position.z = 2
    desired_pose_msg.pose.orientation.x = 0.0
    desired_pose_msg.pose.orientation.y = 0.0
    desired_pose_msg.pose.orientation.z = 0.0
    desired_pose_msg.pose.orientation.w = 1.0

    while not rospy.is_shutdown():
        # Publish the desired pose
        desired_pose_pub.publish(desired_pose_msg)
        rate.sleep()

if __name__ == '__main__':
    try:
        publish_desired_pose()
    except rospy.ROSInterruptException:
        pass
