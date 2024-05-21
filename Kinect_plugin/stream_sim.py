#!/usr/bin/env python

import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
import cv2
import matplotlib as plt
import matplotlib.pyplot as plt
import random as rd
import os
import numpy as np
from geometry_msgs.msg import PoseStamped
import cv2.aruco as aruco


class SensorChecking:
    def __init__(self):
        # Initialize subscribers for both left and right cameras
        self.bridge = CvBridge()
        rospy.Subscriber('/3d_image/image_raw', Image, self.kinect_camera_callback)
        # Placeholder variables for images from each camera
        self.depth_pub = rospy.Publisher('/republished/depth/image_raw', Image, queue_size=100)

        self.depth_image = None
      
    def kinect_camera_callback(self, data):
        # Convert ROS image message to OpenCV image (right camera)
        self.depth_image = self.bridge.imgmsg_to_cv2(data)

        self.pub_frames()


    def pub_frames(self):
        # Check if both images are received
        self.depth_pub.publish(self.bridge.cv2_to_imgmsg(self.depth_image))
        rospy.loginfo('sbah elfoll')
        print(self.depth_image.shape)     

    def show_images(self):
        # Check if both images are received
        if self.depth_image is not None:
            cv2.imshow("depth image", self.depth_image)
            cv2.waitKey(1)

if __name__ == '__main__':
    rospy.init_node('camera_stream', anonymous=True)
    sensor_check = SensorChecking()
    # Main ROS loop
    rate = rospy.Rate(10)  # 10 Hz
    while not rospy.is_shutdown():
        sensor_check.show_images()
        rate.sleep()
