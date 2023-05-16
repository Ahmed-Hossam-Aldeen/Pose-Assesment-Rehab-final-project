import cv2 as cv
import numpy as np
from PyQt5 import QtWidgets, uic
from PyQt5.QtCore import Qt
from PyQt5.uic.properties import QtCore
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QMainWindow
from PyQt5 import QtGui
from PyQt5.QtWidgets import QWidget, QApplication, QLabel
from PyQt5.QtCore import pyqtSignal, pyqtSlot, Qt, QThread
import sys

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

def calculate_angle(p1, p2, p3):
    """
    Calculates angle between three points using trigonometry.

    Args:
        p1 (tuple): First point.
        p2 (tuple): Second point (vertex).
        p3 (tuple): Third point.

    Returns:
        float: Angle in degrees.
    """
    if (p1== None) or (p2== None) or (p3 == None):
        return
    
    else:
        v1 = np.array([p1[0] - p2[0], p1[1] - p2[1]])
        v2 = np.array([p3[0] - p2[0], p3[1] - p2[1]])
        dot_product = np.dot(v1, v2)
        magnitudes_product = np.linalg.norm(v1) * np.linalg.norm(v2)
        cos_theta = dot_product / magnitudes_product
        angle_rad = np.arccos(cos_theta)
        angle_deg = np.degrees(angle_rad)
        angle_deg = round(angle_deg, 2)
        return angle_deg

def calculate_joint_angles(points):
    """
    Calculates joint angles from detected pose points.

    Args:
        points (list): List of pose points.

    Returns:
        dict: Dictionary containing joint angles.
    """
    # Extract pose points
    neck = points[1]
    r_shoulder = points[2]
    l_shoulder = points[5]
    r_elbow = points[3]
    l_elbow = points[6]
    r_wrist = points[4]
    l_wrist = points[7]
    
    # Calculate joint angles
    angles = {}
    angles['right_shoulder_angle'] = calculate_angle(neck, r_shoulder, r_elbow)
    angles['left_shoulder_angle'] = calculate_angle(neck, l_shoulder, l_elbow)

    angles['right_elbow_angle'] = calculate_angle(r_shoulder, r_elbow, r_wrist)
    angles['left_elbow_angle'] = calculate_angle(l_shoulder, l_elbow, l_wrist)

    return angles


class VideoThread(QThread):
    change_pixmap_signal = pyqtSignal(np.ndarray)


    def run(self):
        # capture from web cam
        cap = cv.VideoCapture(0)
        while cv.waitKey(1) < 0:
            hasFrame, frame = cap.read()
            
            inWidth = 256
            inHeight = 256
            confidince = 0.25

            net = cv.dnn.readNetFromTensorflow("graph_opt.pb")
            frameWidth = frame.shape[1]
            frameHeight = frame.shape[0]
            
            net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
            out = net.forward()
            out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

            assert(len(BODY_PARTS) == out.shape[1])

            points = []
            for i in range(len(BODY_PARTS)):
                # Slice heatmap of corresponging body's part.
                heatMap = out[0, i, :, :]

                # Originally, we try to find all the local maximums. To simplify a sample
                # we just find a global one. However only a single pose at the same time
                # could be detected this way.
                _, conf, _, point = cv.minMaxLoc(heatMap)
                x = (frameWidth * point[0]) / out.shape[3]
                y = (frameHeight * point[1]) / out.shape[2]
                # Add a point if it's confidence is higher than threshold.
                points.append((int(x), int(y)) if conf > confidince else None)

            for pair in POSE_PAIRS:
                partFrom = pair[0]
                partTo = pair[1]
                assert(partFrom in BODY_PARTS)
                assert(partTo in BODY_PARTS)

                idFrom = BODY_PARTS[partFrom]
                idTo = BODY_PARTS[partTo]

                if points[idFrom] and points[idTo]:
                    cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                    cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
                    cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)


            print(calculate_joint_angles(points))


            t, _ = net.getPerfProfile()
            freq = cv.getTickFrequency() / 1000
            cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))
            self.change_pixmap_signal.emit(frame)    

class MainWindow(QtWidgets.QMainWindow):      
    def __init__(self):   
        super(MainWindow, self).__init__()
        self.setAcceptDrops(True)
        uic.loadUi('rehab.ui', self)
        self.setWindowTitle("Instrument")

        self.disply_width = 640
        self.display_height = 480
        
        # create the video capture thread
        self.thread = VideoThread()
        # connect its signal to the update_image slot
        self.thread.change_pixmap_signal.connect(self.update_image)
        # start the thread
        self.StartVideo.clicked.connect(self.start_thread)
        self.StopVideo.clicked.connect(self.stop_thread)
        self.show()

    def start_thread(self):
        self.thread.start()
    def stop_thread(self):
        self.thread.terminate()

    @pyqtSlot(np.ndarray)
    def update_image(self, cv_img):
        """Updates the image_label with a new opencv image"""
        qt_img = self.convert_cv_qt(cv_img)
        self.image_label.setPixmap(qt_img)
    
    def convert_cv_qt(self, cv_img):
        """Convert from an opencv image to QPixmap"""
        rgb_image = cv.cvtColor(cv_img, cv.COLOR_BGR2RGB)
        h, w, ch = rgb_image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(rgb_image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(self.disply_width, self.display_height, Qt.KeepAspectRatio)
        return QPixmap.fromImage(p)    


app = 0            
app = QtWidgets.QApplication(sys.argv)
window = MainWindow()
app.exec_()                    
