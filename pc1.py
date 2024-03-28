import cv2
import socket
import struct
import numpy as np
import yaml

import pyautogui

from auto import *


def main():

    with open("data.yaml") as file:
        data = yaml.load(file, Loader= yaml.FullLoader)
        HOST_IP = data["ip"]
        PORT = data["port"]

    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((HOST_IP, PORT))
    server_socket.listen(5)
    print("클라이언트 연결 대기중...")
    connection, addr = server_socket.accept()
    print(f"{addr}에서 연결되었습니다.")

    # 화면 캡처 및 전송
    while True:
        img = Img()
        cap = img.capture()
        frame = cv2.cvtColor(cap, cv2.COLOR_BGR2RGB)
        _, frame = cv2.imencode('.jpg', frame) 

        # cropped_images = img.get_chessboard_image()

        # for i, cropped in enumerate(cropped_images):
        #     frame = cropped
        #     frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        #     _, frame = cv2.imencode('.jpg', frame) 

        data = frame.tobytes() 
        size = len(data)
        connection.sendall(struct.pack(">L", size) + data)







        # data = connection.recv(4096)
        # if data is not None:
        #     message = data.decode()
        #     print(data)


if __name__ == "__main__":
    main()