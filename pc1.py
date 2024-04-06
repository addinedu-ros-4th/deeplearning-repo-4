import cv2
import socket
import struct
import numpy as np
import yaml
import json
import pyautogui
import threading
from queue import Queue

from auto import *

class Server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.socket = None
        self.client_socket = None

    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.bind((self.host, self.port))
        self.socket.listen()
        print(f"Listening on {self.host}:{self.port}...")
        self.client_socket, addr = self.socket.accept()
        print(f"Connection from {addr} has been established.")

    def send(self, data):
        data = data.tobytes()
        size = len(data)
        self.client_socket.sendall(struct.pack(">L", size) + data)

    def recv(self):

        received_data = self.client_socket.recv(1024).decode('utf-8')
    
        try:
            pixel_data = json.loads(received_data)
        
            if pixel_data == "None":
                pixel_data = None
                
            return pixel_data
        
        except json.JSONDecodeError as e:
    
            print(f"JSON 디코드 오류: {e}")
            return None

    def close(self):
        if self.client_socket:
            self.client_socket.close()
        if self.socket:
            self.socket.close()
        print("Server closed.")

def main():

    with open("data.yaml") as file:
        data = yaml.load(file, Loader= yaml.FullLoader)
        HOST_IP = data["ip"]
        PORT = data["port"]

    x_norm = 1920/1850
    y_norm = 1080/1053
    pixel_data = None
    server_socket = Server(HOST_IP, PORT)
    server_socket.start()

    img = Img()
    
    while True:
        cap = img.capture()
        frame = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        _, frame = cv2.imencode('.jpg', frame)

        try:
            server_socket.send(frame)
        except Exception as e:
            print(f"Frame 전송 중 오류 발생: {e}")
            break 

        pixel_data = server_socket.recv()
        if pixel_data is not None:
            from_x = pixel_data["from_x"]
            from_y = pixel_data["from_y"]
            to_x = pixel_data["to_x"]
            to_y = pixel_data["to_y"]
            pyautogui.moveTo(from_x * x_norm, from_y * y_norm)
            pyautogui.click()
            pyautogui.moveTo(to_x * x_norm, to_y * y_norm)
            pyautogui.click()

if __name__ == "__main__":
    main()