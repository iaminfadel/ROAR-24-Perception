#!/usr/bin/env python

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import String

import rospy
from geometry_msgs.msg import Point
from std_msgs.msg import String


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
        rospy.loginfo("Received coordinates: x=%f, y=%f", int(x), int(y))
        
        if self.label in self.label_coordinates:
            self.label_coordinates[self.label] = (int(x), int(y))  # Remove z coordinate

        rospy.loginfo(self.label_coordinates)

    def ret_dic(self):
        return self.label_coordinates

if __name__ == '__main__':
    try:
        rospy.init_node('dictionary_subscriber', anonymous=True)

        subscriber = DictionarySubscriber()
        dict = subscriber.ret_dic()
        rospy.spin()

    finally:
        rospy.loginfo("Received dictionary: %s", subscriber.label_coordinates)
