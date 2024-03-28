import cv2
import socket
import struct
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
import torch
import threading
import queue

def receive_frame(client_socket, frame_queue):
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

        frame_queue.put(frame_data)

def process_and_display(model, frame_queue):
    while True:
        if not frame_queue.empty():
            frame_data = frame_queue.get()
            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            results = model(frame)
        
            cv2.imshow('frame', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

def main():
    model = YOLO("best_v8.pt")

    with open("data.yaml") as file:
        data = yaml.load(file, Loader=yaml.FullLoader)
        HOST_IP = data["ip"]
        PORT = data["port"]

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP, PORT))

    frame_queue = queue.Queue()

    # 서버로부터 이미지 데이터를 받는 스레드
    receive_thread = threading.Thread(target=receive_frame, args=(client_socket, frame_queue))
    receive_thread.start()

    # 받은 이미지 데이터를 처리하고 표시하는 스레드
    display_thread = threading.Thread(target=process_and_display, args=(model, frame_queue))
    display_thread.start()

    receive_thread.join()
    display_thread.join()

    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
