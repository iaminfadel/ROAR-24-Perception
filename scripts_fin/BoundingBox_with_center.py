#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import torch

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

'''There're Two classes
The First One (ImageSubscriber): is used to subscribe the path of the image from (cam_cap_pub.py) python file and 
then it passes the path to the another class

The Second One (BoundingBoxExtractor): is used to detect the numbers first used a pretrained YoloV5 model
that was trained on a custom dataset and then receives the bouding box and then ge the center of each bounding 
box and store it in a dictionary and after all of that it publishes the dictionry , but all of the above to be done 
the class must receive the path from the first class

the reason why we publish the dictionay instead of using it directly in (trial.py) python file to make it generic
for the pipeline line ine the detection process'''

class ImageSubscriber:
    def __init__(self, node_handle):
        self.image_path = None
        self.image_received = False
        self.node_handle = node_handle
        rospy.Subscriber('image_path', String, self.image_path_callback)

    def image_path_callback(self, data):
        self.image_path = data.data
        self.image_received = True

    def get_image_path(self):
        rate = rospy.Rate(10)  # Rate at which to check if the image path has been received
        while not self.image_received:
            rospy.loginfo("No image path received. Waiting for image path...")
            rate.sleep()
        return self.image_path

class BoundingBoxExtractor:
    def __init__(self, node_handle, image_path):
        self.image_path = image_path
        self.label_with_coordinates = {}
        self.threshold = 0.4
        self.node_handle = node_handle

    def extract_bounding_boxes(self):
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom','/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/yolov5/runs/train/yolov5s_ODD_results2/weights/best.pt', force_reload=True)
        
        # Perform inference
        results = model(self.image_path)
        filtered_results = results.xyxy[0][results.xyxy[0][:, 4] >= self.threshold]
        
        # Initialize dictionary for label results
        label_results = {}
        
        # Iterate through filtered results
        for item in filtered_results:  
            class_id = int(item[-1])  
            class_name = model.names[class_id]  
            
            if class_name not in label_results:
                label_results[class_name] = []
                self.label_with_coordinates[class_name] = []  

            label_results[class_name].append(item)
            center = self._draw_center(label_results, class_name)
            self.label_with_coordinates[class_name].append(center)

    def _draw_center(self, label_results, class_name):

        #Getting the coordinated of the bounding box
        x, y, h, w = label_results[class_name][0][0], label_results[class_name][0][1], label_results[class_name][0][2], label_results[class_name][0][3]

        image = cv2.imread(self.image_path)
        #Center Calculation process to get the depth from a specific pixel later on
        center_x = int((x + h) / 2)
        center_y = int((y + w) / 2)
        center = (center_x, center_y)

        cv2.circle(image, (center_x, center_y), 5, (255, 255, 255), -1)
        cv2.imwrite(f'/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/yolov5/Centers/{class_name}_center.jpg', image)
        return center

    #Function to publish the dictionary based on publishing the label as string and the 
    #coordinates as a Point msg

    def publish_dictionary(self):
        
        label_pub = rospy.Publisher('label_topic', String, queue_size=10)
        point_pub = rospy.Publisher('point_topic', Point, queue_size=10)
        rate = rospy.Rate(10) # 10hz

        while not rospy.is_shutdown():
            for label, coordinates in self.label_with_coordinates.items():
                label_pub.publish(label)
                rospy.loginfo("Published label: %s", label)

                for coordinate in coordinates:
                    point_msg = Point()
                    point_msg.x = coordinate[0]
                    point_msg.y = coordinate[1]
                    point_msg.z = 0.0  
                    point_pub.publish(point_msg)
                    rospy.loginfo("Published coordinates: x=%f, y=%f, z=%f", point_msg.x, point_msg.y, point_msg.z)

                rate.sleep()


if __name__ == '__main__':

    #Node initialization for the ros communication process
    rospy.init_node('main_node', anonymous=True)
    #Subcriber Object Init
    image_subscriber = ImageSubscriber(rospy)
    #Getting the path from the ImageSubscriber Class
    image_path = image_subscriber.get_image_path()
    image_path = f"{image_path}"
    #Passing the path to the class to do the rest of the operations
    bbox_extractor = BoundingBoxExtractor(rospy, image_path)
    bbox_extractor.extract_bounding_boxes()
    bbox_extractor.publish_dictionary()
