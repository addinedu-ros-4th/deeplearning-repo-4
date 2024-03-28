import socket
import pyautogui
import numpy as np
import cv2
import struct

from auto import *

# 소켓 연결 설정
HOST_IP = "192.168.0.41"  # PC2의 IP 주소로 변경하세요.
<<<<<<< HEAD
PORT = 9999
=======
PORT = 11111
>>>>>>> e6720ffa0b51061834709134e3d02f2fef5a6703
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(5)
print("클라이언트 연결 대기중...")
connection, addr = server_socket.accept()
print(f"{addr}에서 연결되었습니다.")

# 화면 캡처 및 전송
while True:
    img = Img()
    img.capture()
    cropped_images = img.get_chessboard_image()

    for i, cropped in enumerate(cropped_images):
        frame = cropped
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색상 변환
        _, frame = cv2.imencode('.jpg', frame)  # JPEG 형식으로 인코딩

        data = frame.tobytes()  # 전송을 위해 바이트 형태로 변환
        size = len(data)
        connection.sendall(struct.pack(">L", size) + data)  # 데이터 크기와 데이터 전송
        data = connection.recv(4096)
        if data is not None:
            message = data.decode()
            print(data)