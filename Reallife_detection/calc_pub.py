#!/usr/bin/env python

import rospy
import numpy as np
from std_msgs.msg import String, Float32
from geometry_msgs.msg import Point, PoseStamped
from cv_bridge import CvBridge, CvBridgeError

class calc_pub():
    def __init__(self):
        self.focal = 0.33594
        self.tantheta = 0.7301876930697709
        self.cx = 0
        self.cy = 0
        self.fx = 0
        self.fy = 0
        self.baseline = 10
        self.size = 640 
        self.depths_away = {}
        self.object_x_meters = {}
        self.object_y_meters = {}

        self.centers = {}
        self.last_label = None
        
        self.label_sub = rospy.Subscriber('label_topic', String, self.label_callback)
        self.point_sub = rospy.Subscriber('point_topic', Point, self.point_callback)
        self.pose_pub = rospy.Publisher('/point_poses', PoseStamped, queue_size=100)

        self.load_calid_data()

    def load_calid_data(self):
        data_calib_path = '/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/calibration_data/MultiMatrix.npz'
        data = np.load(data_calib_path)
        camMatrix = data["camMatrix"]

        self.fx = camMatrix[0, 0]
        self.fy = camMatrix[1, 1]
        self.cx = camMatrix[0, 2]
        self.cy = camMatrix[1, 2]

        rospy.loginfo(f"Loaded camera calibration data: fx={self.fx}, fy={self.fy}, cx={self.cx}, cy={self.cy}")

    def label_callback(self, msg):
        self.last_label = msg.data
        rospy.loginfo(f"Received center label: {self.last_label}")

    def point_callback(self, msg):
        if self.last_label is not None:
            self.centers[self.last_label] = (msg.x, msg.y, msg.z)
            rospy.loginfo(f"Updated centers with label: {self.last_label}, coordinates: ({msg.x}, {msg.y}, {msg.z})")
            
            self.calc_depth(self.last_label)
            self.calc_object_xy(self.last_label)
            self.publish_pose(self.last_label)

    def calc_depth(self, label):
        if label in self.centers:
            x, y, z = self.centers[label]
            depth = (self.baseline / 2) * self.size * (1 / self.tantheta) / z + self.focal
            self.depths_away[label] = depth
            rospy.loginfo(f"Calculated depth for label {label}: {depth}")
        else:
            rospy.logwarn(f"Cannot calculate depth for label {label}: Centers not available.")

    def calc_object_xy(self, label):
        if label in self.centers and label in self.depths_away:
            x, y, z = self.centers[label]
            depth = self.depths_away[label]
            object_x = (x - self.cx) * depth / self.fx
            object_y = (y - self.cy) * depth / self.fy
            self.object_x_meters[label] = object_x
            self.object_y_meters[label] = object_y
            rospy.loginfo(f"Calculated object X for label {label}: {object_x}")
            rospy.loginfo(f"Calculated object Y for label {label}: {object_y}")
        else:
            rospy.logwarn(f"Cannot calculate object X and Y for label {label}: Centers or depth not available.")

    def publish_pose(self, label):
        if label in self.centers and label in self.depths_away:
            objX = self.object_x_meters[label]
            objY = self.object_y_meters[label]
            objZ = self.depths_away[label]

            norm = np.linalg.norm([objX, objY, objZ])
            objX /= norm
            objY /= norm
            objZ /= norm

            camera_x = [1, 0, 0]
            camera_y = [0, 1, 0]
            camera_z = [0, 0, 1]

            dot_x = np.dot(camera_x, [objX, objY, objZ])
            dot_y = np.dot(camera_y, [objX, objY, objZ])
            dot_z = np.dot(camera_z, [objX, objY, objZ])

            rotation_matrix = np.array([[dot_x, dot_y, dot_z],
                                        [0, 0, 0],
                                        [0, 0, 0]])

            qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
            qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
            qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
            qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

            pose_msg = PoseStamped()
            pose_msg.header.stamp = rospy.Time.now()
            pose_msg.header.frame_id = label
            pose_msg.pose.position.x = objX
            pose_msg.pose.position.y = objY
            pose_msg.pose.position.z = objZ
            pose_msg.pose.orientation.x = qx
            pose_msg.pose.orientation.y = qy
            pose_msg.pose.orientation.z = qz
            pose_msg.pose.orientation.w = qw

            self.pose_pub.publish(pose_msg)
            rospy.loginfo(f"Published pose for label {label}: Position ({objX}, {objY}, {objZ}), Orientation ({qx}, {qy}, {qz}, {qw})")
        else:
            rospy.logwarn(f"Cannot publish pose for label {label}: Centers or depth not available.")

if __name__ == '__main__':
    rospy.init_node('track_subscriber', anonymous=True)
    calc = calc_pub()
    try:
        rospy.spin()
    except rospy.ROSInterruptException:
        pass
