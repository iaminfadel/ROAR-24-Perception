import cv2
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge

from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel, QVBoxLayout, QPushButton, QLineEdit
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys

class ImageSubscriber(QThread):
    image_updated = pyqtSignal(QImage)
    camera_connected = pyqtSignal(bool)

    def __init__(self, topic):
        super().__init__()
        self.bridge = CvBridge()
        self.image = None
        self.subscriber = rospy.Subscriber(topic, Image, self.callback)
        self.camera_connected.emit(False)

    def callback(self, msg):
        try:
            rospy.loginfo("Received Successfully")
            self.image = self.bridge.imgmsg_to_cv2(msg)
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # Convert to grayscale

            h, w = self.image.shape
            bytes_per_line = w
            convert_to_Qt_format = QtGui.QImage(self.image.data, w, h, bytes_per_line, QtGui.QImage.Format_Grayscale8)
            self.image_updated.emit(convert_to_Qt_format)
            self.camera_connected.emit(True)
        except Exception as e:
            rospy.logerr(e)

    def run(self):
        rospy.spin()

    def stop(self):
        self.subscriber.unregister()

class App(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ROS Image Viewer")
        self.display_width = 640
        self.display_height = 480

        # create the label that holds the image
        self.image_label = QLabel(self)
        self.image_label.setFixedSize(self.display_width, self.display_height)

        # create a vertical box layout and add the image label
        vbox = QVBoxLayout()
        vbox.addWidget(self.image_label)

        # create a text box for camera connection status
        self.connection_status = QLineEdit("Waiting for camera to connect...")
        self.connection_status.setReadOnly(True)
        vbox.addWidget(self.connection_status)

        # create a button to stop streaming
        self.stop_button = QPushButton("Stop Streaming")
        self.stop_button.clicked.connect(self.stop_streaming)
        vbox.addWidget(self.stop_button)

        # set the vbox layout as the widgets layout
        self.setLayout(vbox)

        # create the image subscriber thread
        self.image_subscriber = ImageSubscriber('obj_detect')
        
        self.image_subscriber.image_updated.connect(self.update_image)
        self.image_subscriber.camera_connected.connect(self.update_connection_status)

        # start the thread
        self.image_subscriber.start()

    @pyqtSlot(QImage)
    def update_image(self, image):
        """Updates the image_label with a new QImage"""
        self.image_label.setPixmap(QPixmap.fromImage(image))

    @pyqtSlot(bool)
    def update_connection_status(self, connected):
        """Updates the connection status based on camera connection"""
        if connected:
            self.connection_status.setText("Camera connected.")
        else:
            self.connection_status.setText("Waiting for camera to connect...")

    def stop_streaming(self):
        """Stop streaming and close the application"""
        self.image_subscriber.stop()
        self.connection_status.setText("Camera off")
        self.close()

if __name__ == "__main__":
    rospy.init_node('camera_subscriber', anonymous=True)

    app = QApplication(sys.argv)
    a = App()
    a.show()
    sys.exit(app.exec_())
