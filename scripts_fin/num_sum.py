#!/usr/bin/env python

import sys
import rospy
from std_msgs.msg import String, Int32
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QPushButton, QLineEdit, QComboBox
from PyQt5.QtCore import QTimer
import random

class CustomPublisher(QWidget):
    def __init__(self):
        super().__init__()

        self.init_ui()
        self.init_ros()

    def init_ui(self):
        self.setWindowTitle('Custom Publisher')
        self.resize(300, 150)

        self.number_button = QPushButton('Publish Number', self)

        self.number_input = QLineEdit(self)

        layout = QVBoxLayout()
        layout.addWidget(self.number_input)
        layout.addWidget(self.number_button)

        self.setLayout(layout)

        self.number_button.clicked.connect(self.publish_number)

    def init_ros(self):
        rospy.init_node('gui_decision_publisher', anonymous=True)
        self.number_pub = rospy.Publisher('custom_number_topic', Int32, queue_size=10)
    def publish_number(self):
        number_msg_str = self.number_input.text()
        if number_msg_str.isdigit():
            number_msg = int(number_msg_str)
            rospy.loginfo("Publishing number: {}".format(number_msg))
            self.number_pub.publish(number_msg)


def main():
    app = QApplication(sys.argv)
    publisher = CustomPublisher()
    publisher.show()

    sys.exit(app.exec_())


if __name__ == '__main__':
    main()
