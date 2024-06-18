#!/usr/bin/env python

import rospy
import cv2
from sensor_msgs.msg import Image
from geometry_msgs.msg import Point
from std_msgs.msg import String
from cv_bridge import CvBridge, CvBridgeError
import torch 
import numpy as np

import rospy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from supporting_functions import get_dist_to_centre_br , get_cost , get_tracks , get_horiz_dist_corner_tl , get_horiz_dist_corner_br

class sub_detect_extract_imgs():
    def __init__(self):
        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None
        self.right_image_sub = rospy.Subscriber('right_camera_image', Image, self.right_image_callback)
        self.left_image_sub = rospy.Subscriber('left_camera_image', Image, self.left_image_callback)
    def right_image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.right_image = cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def left_image_callback(self, data):
        try:
            cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
            self.left_image = cv_image
        except CvBridgeError as e:
            rospy.logerr("CvBridge Error: %s", e)

    def show_imgs(self):
        while not rospy.is_shutdown():
            if self.right_image is not None:
                cv2.imshow('right_image', self.right_image)
            if self.left_image is not None:
                cv2.imshow('left_image', self.left_image)
            # Wait for 1 millisecond and check for key press to close the windows
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cv2.destroyAllWindows()

    def get_images(self):
        # Return the left and right images as a list of tuples
        imgs = [('left_image', self.left_image), ('right_image', self.right_image)]
        return imgs

    def spin(self):
        rospy.spin()


class sub_detect_extract_path():
    def __init__(self):
        self.bridge = CvBridge()
        self.left_image = None
        self.right_image = None
        self.left_image_received = False
        self.right_image_received = False
        self.right_image_path_sub = rospy.Subscriber('right_image_path', String, self.right_image_callback)
        self.left_image_path_sub = rospy.Subscriber('left_image_path', String, self.left_image_callback)

      
    def left_image_path_callback(self, data):
        image_path = data.data
        rospy.loginfo("Received left image path: %s", image_path)
        self.left_image = self.load_image_from_path(image_path)
        self.left_image_received = True

    def right_image_path_callback(self, data):
        image_path = data.data
        rospy.loginfo("Received right image path: %s", image_path)
        self.right_image = self.load_image_from_path(image_path)
        self.right_image_received = True

    def load_image_from_path(self, image_path):
        try:
            cv_image = cv2.imread(image_path, cv2.IMREAD_COLOR)
            return cv_image
        except Exception as e:
            rospy.logerr("Error loading image from path %s: %s", image_path, str(e))
            return None

    def get_images(self):
        # Wait until both left and right images are received
        rate = rospy.Rate(10)  # Rate at which to check if both images are received
        while not (self.left_image_received and self.right_image_received):
            rospy.loginfo("Waiting for both left and right images to be received...")
            rate.sleep()
        imgs = [('left_image', self.left_image), ('right_image', self.right_image)]
        return imgs

class BoundingBoxExtractor:
    def __init__(self, node_handle, image_path):
        self.image_path = image_path
        self.label_with_coordinates = {}
        self.threshold = 0.5
        self.node_handle = node_handle

        self.image_subscriber = sub_detect_extract_imgs()
        # self.image_subscriber = sub_detect_extract_path()
        self.images = self.image_subscriber.get_images()

    def extract_bounding_boxes(self):
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom','/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/yolov5/runs/train/yolov5s_ODD_results2/weights/best.pt', force_reload=True)
        
        for img_name, img in self.images:

            results = model(img)
            # Extract results
            detections = results.xyxy[0].numpy()  # Convert to NumPy array for easier manipulation
            # Initialize lists to hold detection info
            xyxy = []
            labels = []

            # Draw bounding boxes on the image
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                if confidence >= self.threshold:  # Confidence threshold
                    # Store detection info
                    xyxy.append([x1, y1, x2, y2])
                    labels.append(results.names[int(class_id)])

                    # Draw bounding box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Draw label and confidence
                    label = f"{results.names[int(class_id)]}: {confidence:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Store the data in the dictionary
            self.data[img_name] = {
                "Boundary Boxes Coordinates (xyxy)": xyxy,
                "labels": labels
            }

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



class BoundingBoxExtractor_Maually:
    def __init__(self, ):
        right_img = cv2.imread('/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/real_life_images/right image0.png')
        left_img = cv2.imread('/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/real_life_images/left image3.png')
        self.imgs = [('left_image', left_img), ('right_image', right_img)]
        self.label_with_coordinates = {}
        self.threshold = 0.5
        self.data = {}
    def extract_bounding_boxes(self):
        # Load YOLOv5 model
        model = torch.hub.load('ultralytics/yolov5', 'custom', '/home/omar/RobotArmPerception/cameraSim/src/arm_urdf/yolov5/runs/train/yolov5s_ODD_results2/weights/best.pt', force_reload=True)
        for img_name, img in self.imgs:

            results = model(img)
            # Extract results
            detections = results.xyxy[0].numpy()  # Convert to NumPy array for easier manipulation
            # Initialize lists to hold detection info
            xyxy = []
            labels = []

            # Draw bounding boxes on the image
            for detection in detections:
                x1, y1, x2, y2, confidence, class_id = detection
                if confidence >= self.threshold:  # Confidence threshold
                    # Store detection info
                    xyxy.append([x1, y1, x2, y2])
                    labels.append(results.names[int(class_id)])

                    # Draw bounding box
                    cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
                    # Draw label and confidence
                    label = f"{results.names[int(class_id)]}: {confidence:.2f}"
                    cv2.putText(img, label, (int(x1), int(y1) - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

            # Store the data in the dictionary
            self.data[img_name] = {
                "Boundary Boxes Coordinates (xyxy)": xyxy,
                "labels": labels
            }

    def print_data(self):
        for img_name, info in self.data.items():
            print(f"Image: {img_name}")
            print("Bounding Boxes Coordinates (xyxy):")
            for box in info["Boundary Boxes Coordinates (xyxy)"]:
                print(f"  {box}")
            print("Labels:")
            for label in info["labels"]:
                print(f"  {label}")

    def de(self):
        right_img_detections = self.data['right_image']
        self.right_detections = right_img_detections['Boundary Boxes Coordinates (xyxy)']
        self.right_labels = right_img_detections['labels']

        left_img_detections = self.data['left_image']
        self.left_detections = left_img_detections['Boundary Boxes Coordinates (xyxy)']
        self.left_labels = left_img_detections['labels']

        self.left_detections = np.array(self.left_detections )
        self.right_detections = np.array(self.right_detections)

        tmp1 = get_dist_to_centre_br(self.left_detections)
        tmp2 = get_dist_to_centre_br(self.right_detections)

        self.det = [self.left_detections , self.right_detections]
        self.lbls = [np.array(self.left_labels) , np.array(self.right_labels)] 
        self.cost = get_cost(self.det, lbls = self.lbls)
   
    def publish_dictionary(self):

        image_name_pub = rospy.Publisher('image_name', String, queue_size=10)
        label_pub = rospy.Publisher('label_topic', String, queue_size=10)
        point_pub = rospy.Publisher('point_topic', Point, queue_size=10)
        rate = rospy.Rate(10)  # 10hz

        while not rospy.is_shutdown():
            for img_name, info in self.data.items():
                # Publish image name
                image_name_pub.publish(img_name)
                rospy.loginfo("Published image name: %s", img_name)

                for label, box in zip(info["labels"], info["Boundary Boxes Coordinates (xyxy)"]):
                    # Publish label
                    label_pub.publish(label)
                    rospy.loginfo("Published label: %s", label)

                    # Publish coordinates
                    point_msg = Point()
                    point_msg.x = (box[0] + box[2]) / 2  # center x
                    point_msg.y = (box[1] + box[3]) / 2  # center y
                    point_msg.z = 0.0  # Assuming z = 0.0
                    point_pub.publish(point_msg)
                    rospy.loginfo("Published coordinates: x=%f, y=%f, z=%f", point_msg.x, point_msg.y, point_msg.z)

                rate.sleep()

def main():
    rospy.init_node('bounding_box_extractor_node', anonymous=True)
    extractor = BoundingBoxExtractor_Maually()
    extractor.extract_bounding_boxes()
    extractor.print_data()
    extractor.publish_dictionary()

if __name__ == '__main__':
    try:
        main()
    except rospy.ROSInterruptException:
        pass

# add next take detections and add them too
# tmp1 = get_dist_to_centre_br(left_detections)
# tmp2 = get_dist_to_centre_br(right_detections)
# det = [left_detections , right_detections]
# lbls = [np.array(left_labels) , np.array(left_labels)]
# cost = get_cost(det, lbls = lbls)
#tracks = scipy.optimize.linear_sum_assignment(cost)
#     dists_tl =  get_horiz_dist_corner_tl(det)
# dists_br =  get_horiz_dist_corner_br(det)
#final_dists = []
# dctl = get_dist_to_centre_tl(det[0])
# dcbr = get_dist_to_centre_br(det[0])

# for i, j in zip(*tracks):
#     if dctl[i] < dcbr[i]:
#         final_dists.append((dists_tl[i][j],lbls[0][i]))
        
#     else:
#         final_dists.append((dists_br[i][j],lbls[0][i]))