import cv2
import socket
import struct
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
import torch
import pandas as pd

from auto import *


def main():

    model = YOLO("best_v8.pt")

    with open("data.yaml") as file:
        data = yaml.load(file, Loader= yaml.FullLoader)
        HOST_IP = data["ip"]
        PORT = data["port"]


    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP, PORT))


    data = b""
    payload_size = struct.calcsize(">L")
    while True:
        while len(data) < payload_size:
            data += client_socket.recv(4096)
        
        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)
        
        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
        cropped_images = []
        COLOR = (0, 200, 0) #Rectangle color
        
        results = model(frame)
        frame = results[0].plot(font_size = 10, pil = True)

        for cnt in contours:
            if cv2.contourArea(cnt) > 100000:
                x, y, width, height = cv2.boundingRect(cnt)
                # cv2.rectangle(image,(x, y),(x + width, y + height), COLOR, 2)
                cropped = frame[y:y+height, x:x+width]
        cv2.imshow('frame', cropped)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break


    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()