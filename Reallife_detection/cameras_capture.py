#!/usr/bin/env python

import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from std_msgs.msg import String
import os

class WriteImage(object):
    def __init__(self):
        self.topic_name_right = 'right_camera_image'
        self.topic_name_left = 'left_camera_image'
        self.bridge_object = CvBridge()  # object to convert to the cv2 format and be able to de processes

        self.right_cap = cv2.VideoCapture(4)
        self.left_cap = cv2.VideoCapture(2)

        self.image_right_path_pub = rospy.Publisher('right_image_path', String, queue_size=10)
        self.right_publisher = rospy.Publisher(self.topic_name_right, Image, queue_size=10)

        self.image_left_path_pub = rospy.Publisher('left_image_path', String, queue_size=10)
        self.left_publisher = rospy.Publisher(self.topic_name_left, Image, queue_size=10)

        self.image_save_path = '/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/real_life_images'

    def save_image(self, image, filename):
        image_path = os.path.join(self.image_save_path, filename)
        cv2.imwrite(image_path, image)
        return image_path

    def publish_right_camera_image(self):
        ret, frame = self.right_cap.read()
        if ret:
            rospy.loginfo("Right Image Published Successfully")
            img = self.bridge_object.cv2_to_imgmsg(frame, "bgr8")
            self.right_publisher.publish(img)
            image_path = self.save_image(frame, 'right_camera_img.png')
            self.publish_right_image_path(image_path)

    def publish_left_camera_image(self):
        ret, frame = self.left_cap.read()
        if ret:
            rospy.loginfo("Left Image Published Successfully")
            img = self.bridge_object.cv2_to_imgmsg(frame, "bgr8")
            self.left_publisher.publish(img)
            image_path = self.save_image(frame, 'left_camera_img.png')
            self.publish_left_image_path(image_path)

    def publish_right_image_path(self, image_path):
        path_msg = String()
        path_msg.data = image_path
        self.image_right_path_pub.publish(path_msg)
        rospy.loginfo("Published right image path: %s", image_path)

    def publish_left_image_path(self, image_path):
        path_msg = String()
        path_msg.data = image_path
        self.image_left_path_pub.publish(path_msg)
        rospy.loginfo("Published left image path: %s", image_path)

    def spin(self):
        rate = rospy.Rate(10)  # 10 Hz
        while not rospy.is_shutdown():
            self.publish_right_camera_image()
            self.publish_left_camera_image()
            rate.sleep()

if __name__ == '__main__':
    rospy.init_node('write_image_node', anonymous=True)
    write_image_object = WriteImage()
    try:
        write_image_object.spin()
    except rospy.ROSInterruptException:
        pass
    finally:
        write_image_object.right_cap.release()
        write_image_object.left_cap.release()
        cv2.destroyAllWindows()
        rospy.loginfo("Shutting down")
