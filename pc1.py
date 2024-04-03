import cv2
import socket
import struct
import numpy as np
import yaml
import json
import pyautogui
import threading
import queue

from auto import *

def handle_client(connection,pixel_queue):
    while True:
        received_pixel_data = connection.recv(1024)
        if received_pixel_data:
            pixel_data = json.loads(received_pixel_data.decode('utf-8'))
            pixel_queue.put(pixel_data)
        elif not received_pixel_data:
            break

def main():
    pixel_queue = queue.Queue()
    with open("data.yaml") as file:
        data = yaml.load(file, Loader= yaml.FullLoader)
        HOST_IP = data["ip"]
        PORT = data["port"]

    x_norm = 1920/1850
    y_norm = 1080/1053

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(5)
    print("클라이언트 연결 대기중...")
    connection, addr = server_socket.accept()
    print(f"{addr}에서 연결되었습니다.")
    img = Img()

    while True:

        cap = img.capture()
        frame = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        _, frame = cv2.imencode('.jpg', frame)

        data = frame.tobytes() 
        size = len(data)
        connection.sendall(struct.pack(">L", size) + data)

        server_thread = threading.Thread(target=handle_client, args=(connection, pixel_queue))
        server_thread.start()
        if not pixel_queue.empty():
            pixel_data = pixel_queue.get()
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