#!/usr/bin/env python
import rospy
from cv_bridge import CvBridge, CvBridgeError
from sensor_msgs.msg import Image
import numpy as np
import cv2
from geometry_msgs.msg import PoseStamped
import matplotlib.pyplot as plt
from geometry_msgs.msg import Point
from std_msgs.msg import String

# label_with_coordinates = {'5': (406, 46), '3': (406, 407), '8': (165, 407), '1': (167, 45)}

'''in my approach for number detection i would use dictionary with labels and numbers and i would publish the poseStamp So 
Below there's a Function to extract the points from the dictionary and also the labels'''
# def extract_points(dictionary):
#     points = []
#     labels = []
#     for key, value in dictionary.items():
#         points.append(value)
#         labels.append(key)
#     return points , labels

# # Example usage:
# points , labels = extract_points(label_with_coordinates)
# print(points)
# print(labels)

'''If the wanted is to get depth from all the Kinect camera scene U could use the below Functions'''
    # def convert_depth_image(ros_image):
    #     bridge = CvBridge()
    #      # Use cv_bridge() to convert the ROS image to OpenCV format
    #     try:
    #      #Convert the depth image using the default passthrough encoding
    #         depth_image = bridge.imgmsg_to_cv2(ros_image, "passthrough")
    #         depth_array = np.array(depth_image, dtype=np.float32)
    #         rospy.loginfo("Depth array: %s", depth_array)
    #     except CvBridgeError as e:
    #         rospy.logerr("CvBridge Error: %s", e)

    # def pixel2depth():
    # 	rospy.init_node('pixel2depth',anonymous=True)
    # 	rospy.Subscriber("/3d_image/image_raw_depth", Image,callback=convert_depth_image, queue_size=1)
    # 	rospy.spin()

'''If the wanted is to get depth Specific Point in the Kinect camera scene U could use the below Functions'''
# the parameters used to test it with function without the API
# point_x_pixels = 216
# point_y_pixels = 412
# object_x_meters = 0 
# object_y_meters = 0
# point_z = 0
# position = [object_x_meters , object_y_meters , point_z]
# pose = []

# def depth_image_callback(msg):
#     try:
#         # Convert ROS Image message to OpenCV image
#         bridge = CvBridge()
#         depth_image = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        
#         # Convert the depth image to a Numpy array
#         depth_array = np.array(depth_image, dtype=np.float32)
        
#         # Extract depth at the specified point
#         depth_at_point = depth_array[point_y_pixels, point_x_pixels]
#         point_z = depth_at_point
#         rospy.loginfo("Depth at point (%d, %d): %.2f meters", point_x_pixels, point_y_pixels, depth_at_point)
#         print('point Z',point_z)

#         get_orientation_from_xyz(object_x_meters , object_y_meters , point_z)
#     except CvBridgeError as e:
#         rospy.logerr("CvBridge Error: %s", e)
    

# def get_x_y(depth):
    
#     fx = 525.0  # focal length in pixels
#     fy = 525.0
#     cx = 320.0  # principal point in pixels
#     cy = 240.0
#     # centers would get from the bounding box
#     object_x_meters = (point_x_pixels - cx)*depth / fx
#     object_y_meters = (point_y_pixels - cy)*depth / fy
#     print ('Center point x & y in meters' , (object_x_meters , object_y_meters))
#     #cx cy fx fy from camera calibration
#     return object_x_meters , object_y_meters , depth

# def get_orientation_from_xyz(objX, objY, objZ):
#     # Normalize the object's position vector
#     norm = np.linalg.norm([objX, objY, objZ])
#     objX /= norm
#     objY /= norm
#     objZ /= norm
    
#     # Define camera frame axes
#     camera_x = [1, 0, 0]
#     camera_y = [0, 1, 0]
#     camera_z = [0, 0, 1]

#     # Calculate the dot products between the camera frame axes and the object's position vector
#     dot_x = np.dot(camera_x, [objX, objY, objZ])
#     dot_y = np.dot(camera_y, [objX, objY, objZ])
#     dot_z = np.dot(camera_z, [objX, objY, objZ])

#     # Construct the rotation matrix
#     rotation_matrix = np.array([[dot_x, dot_y, dot_z],
#                                  [0, 0, 0],
#                                  [0, 0, 0]])

#     # Calculate the quaternion from the rotation matrix
#     qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
#     qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
#     qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
#     qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

#     pose = [qx , qy , qz , qw]
   
#     print([qx , qy , qz , qw])
#     # return pose


# def main():
   
#     rospy.init_node('depth_point_extraction', anonymous=True)
#     rospy.Subscriber('/3d_image/image_raw_depth', Image, depth_image_callback)
#     rospy.spin()



'''To publish the 6D pose as PoseStamp message just use the below function'''

# def publish_pose():
#     # Initialize the ROS node
#     rospy.init_node('object_pose_publisher', anonymous=True)

#     # Create a publisher for the object pose
#     pose_pub = rospy.Publisher('/object_pose', PoseStamped, queue_size=10)

#     # Set the publishing rate
#     rate = rospy.Rate(10)  # 10 Hz

#     while not rospy.is_shutdown():
#         # Determine the pose of the object (example values)
#         object_position = position # [x, y, z] # 0.5 would be equal to depth_at_point from the above func
#         object_orientation = pose  # Quaternion [x, y, z, w]

#         # Create a PoseStamped message
#         pose_msg = PoseStamped()
#         pose_msg.header.stamp = rospy.Time.now()
#         pose_msg.header.frame_id = "base_link"  # Set the frame of reference
#         pose_msg.pose.position.x = object_position[0]
#         pose_msg.pose.position.y = object_position[1]
#         pose_msg.pose.position.z = object_position[2]
#         pose_msg.pose.orientation.x = object_orientation[0]
#         pose_msg.pose.orientation.y = object_orientation[1]
#         pose_msg.pose.orientation.z = object_orientation[2]
#         pose_msg.pose.orientation.w = object_orientation[3]

#         # Publish the pose message
#         pose_pub.publish(pose_msg)

#         # Sleep to maintain the publishing rate
#         rate.sleep()
'''Class for subscribing the dictionary instead of manual feed'''


class DictionarySubscriber:
    def __init__(self):
        self.label_coordinates = {}
        rospy.Subscriber('label_topic', String, self.callback_label)
        rospy.Subscriber('point_topic', Point, self.callback_point)

    def callback_label(self, data):
        self.label = data.data
        if self.label not in self.label_coordinates:
            self.label_coordinates[self.label] = None

    def callback_point(self, data):
        x = data.x
        y = data.y
        # z = data.z  # Remove z coordinate
        # rospy.loginfo("Received coordinates: x=%f, y=%f", int(x), int(y))
        
        if self.label in self.label_coordinates:
            self.label_coordinates[self.label] = (int(x), int(y))  # Remove z coordinate

        # rospy.loginfo(self.label_coordinates)

    def ret_dic(self):
        return self.label_coordinates
    

'''Class for the Architecture'''
class DepthExtractor:
    def __init__(self, points_dictionary):
        # the below four parameters are gotten from the camera calibration file i would be more generic to make a function
        # to extract each value from the YAML calibration file

        self.fx = 525.0  # focal length in pixels
        self.fy = 525.0
        self.cx = 320.0  # principal point in pixels
        self.cy = 240.0

        self.bridge = CvBridge()
        rospy.Subscriber('/3d_image/image_raw_depth', Image, self.depth_image_callback)
        self.pose_pub = rospy.Publisher('/point_poses', PoseStamped, queue_size=100) # to pub the pose with the id
        self.points_dictionary = points_dictionary

    def depth_image_callback(self, msg):
        try:
            # Convert ROS Image message to OpenCV image
            depth_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
            
            # Convert the depth image to a Numpy array
            depth_array = np.array(depth_image, dtype=np.float32)
            
            for label, point in self.points_dictionary.items():
                self.get_depth_at_point(depth_array, label, point)
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def get_depth_at_point(self, depth_array, label, point):

        point_x_pixels, point_y_pixels = point
        depth_at_point = depth_array[point_y_pixels, point_x_pixels]
        object_z_meters = depth_at_point
         # Calculate x, y coordinates in meters
        object_x_meters = (point_x_pixels - self.cx) * object_z_meters / self.fx
        object_y_meters = (point_y_pixels - self.cy) * object_z_meters / self.fy
        
        rospy.loginfo("Depth at point %s (%d, %d): %.2f meters", label, point_x_pixels, point_y_pixels, object_z_meters)
        rospy.loginfo('Center point x & y in meters: (%f, %f)', object_x_meters, object_y_meters)
        self.get_orientation_from_xyz(object_x_meters, object_y_meters, object_z_meters, label)

    def get_orientation_from_xyz(self, objX, objY, objZ, label):
        # Normalize the object's position vector
        objx_copy = objX.copy()
        objy_copy = objY.copy()
        objz_copy = objZ.copy()
        norm = np.linalg.norm([objx_copy, objy_copy, objz_copy])
        objx_copy /= norm
        objy_copy /= norm
        objz_copy /= norm
        
        # Define camera frame axes
        camera_x = [1, 0, 0]
        camera_y = [0, 1, 0]
        camera_z = [0, 0, 1]

        # Calculate the dot products between the camera frame axes and the object's position vector
        dot_x = np.dot(camera_x, [objx_copy, objy_copy, objz_copy])
        dot_y = np.dot(camera_y, [objx_copy, objy_copy, objz_copy])
        dot_z = np.dot(camera_z, [objx_copy, objy_copy, objz_copy])

        # Construct the rotation matrix
        rotation_matrix = np.array([[dot_x, dot_y, dot_z],
                                     [0, 0, 0],
                                     [0, 0, 0]])

        # Calculate the quaternion from the rotation matrix
        qw = np.sqrt(1 + rotation_matrix[0, 0] + rotation_matrix[1, 1] + rotation_matrix[2, 2]) / 2
        qx = (rotation_matrix[2, 1] - rotation_matrix[1, 2]) / (4 * qw)
        qy = (rotation_matrix[0, 2] - rotation_matrix[2, 0]) / (4 * qw)
        qz = (rotation_matrix[1, 0] - rotation_matrix[0, 1]) / (4 * qw)

        pose = [qx , qy , qz , qw]
        
        # Create PoseStamped message
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
        
        # Publish PoseStamped message
        self.pose_pub.publish(pose_msg)

if __name__ == '__main__':

    rospy.init_node('depth_point_extraction', anonymous=True)
    try:

        subscriber = DictionarySubscriber()
        dict = subscriber.ret_dic()
        rospy.loginfo(dict)
        depth_extractor = DepthExtractor(dict)

        rospy.spin()

    finally:
        rospy.loginfo("Received dictionary: %s", subscriber.label_coordinates)

