import sys
import cv2
import socket
import struct
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
import torch
import json
from auto import *
from chess_game import *
from alpha_zero_model.config import Config
from alpha_zero_model.chess_env import ChessEnv
from alpha_zero_model.player_chess import ChessPlayer
from alpha_zero_model.model_chess import ChessModel
from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
import threading


from_class = uic.loadUiType("chessAI.ui")[0]

class ImageProcessor(QObject):
    imageProcessed = pyqtSignal(np.ndarray)

    def __init__(self, parent=None):
        super(ImageProcessor, self).__init__(parent)

    def processImage(self, frame):
        # Perform image processing tasks here
        # For example, YOLO object detection, contour detection, etc.
        # Processed image is then emitted to the main GUI thread
        results = yolo_model.predict(source=frame, classes=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], verbose=False)
        frame = results[0].plot(font_size=10, pil=True)
        processed_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.imageProcessed.emit(processed_image)

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CHESS AI")

        # Create worker and thread for image processing
        self.imageProcessor = ImageProcessor()
        self.workerThread = QThread()
        self.imageProcessor.moveToThread(self.workerThread)

        # Connect signals and slots
        self.imageProcessor.imageProcessed.connect(self.updateImage)
        self.workerThread.started.connect(self.processImages)

        # Start the worker thread
        self.workerThread.start()

    def processImages(self):
        while True:
            # Read frame from socket
            frame_data = b""
            while len(frame_data) < payload_size:
                frame_data += client_socket.recv(4096)
            packed_msg_size = frame_data[:payload_size]
            frame_data = frame_data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]
            while len(frame_data) < msg_size:
                frame_data += client_socket.recv(4096)
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Process frame
            self.imageProcessor.processImage(frame)

    def updateImage(self, image):
        # Update GUI with processed image
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QImage(image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(580, 580, Qt.KeepAspectRatio)
        self.chessScreen.setPixmap(QPixmap.fromImage(p))

    def closeEvent(self, event):
        # Stop the worker thread when the window is closed
        self.workerThread.quit()
        self.workerThread.wait()
        event.accept()

if __name__ == "__main__":
    # Initialize YOLO model
    yolo_model_path = "best_v8.pt"
    yolo_model = YOLO(yolo_model_path)

    # Initialize socket connection
    HOST_IP = "192.168.0.54"
    PORT = 18885
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP, PORT))

    # Initialize payload size for receiving frame
    payload_size = struct.calcsize(">L")

    # Start GUI application
    app = QApplication(sys.argv)
    myWindows= WindowClass()
    myWindows.show()
    sys.exit(app.exec_())
