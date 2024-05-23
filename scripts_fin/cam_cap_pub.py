#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import os

'''The class used to capture image from the camera when it reaches its specified position and then saves 
it in a specific path and publish that path to another '''

class WriteImage(object):
    def __init__(self):
        self.bridge_object = CvBridge() #object to convert to the cv2 format and be able to de processes
        self.image_path_pub = rospy.Publisher('image_path', String, queue_size=10)
        # subscribes from camera topic , it requires the master always to strat the process
        self.right_image_sub = rospy.Subscriber('/3d_image/image_raw', Image, self.kinect_camera_callback) 

    def kinect_camera_callback(self, data):
        try:
            cv_image = self.bridge_object.imgmsg_to_cv2(data, desired_encoding="bgr8")
            # Save the captured photo
            path = '/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/number_data/'
            image_path = os.path.join(path, 'depth_image.png')
            cv2.imwrite(image_path, cv_image)
            # Publish the image path
            self.publish_image_path(image_path)
        except CvBridgeError as e:
            rospy.logerr(e)

    def publish_image_path(self, image_path):
        path_msg = String()
        path_msg.data = image_path
        self.image_path_pub.publish(path_msg)
        rospy.loginfo("Published image path: %s", image_path)

if __name__ == '__main__':
    rospy.init_node('write_image_node', anonymous=True)
    write_image_object = WriteImage()
    try:
        rospy.spin()
    except KeyboardInterrupt:
        rospy.loginfo("Shutting down")
