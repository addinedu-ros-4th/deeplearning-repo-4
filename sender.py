import socket
import pyautogui
import numpy as np
import cv2
import struct

# 소켓 연결 설정
HOST_IP = "192.168.0.41"  # PC2의 IP 주소로 변경하세요.
PORT = 9999
server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.bind((HOST_IP, PORT))
server_socket.listen(5)
print("클라이언트 연결 대기중...")
connection, addr = server_socket.accept()

# 화면 캡처 및 전송
while True:
    screen = pyautogui.screenshot()  # 화면 캡처
    frame = np.array(screen)  # 이미지를 numpy 배열로 변환
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 색상 변환
    _, frame = cv2.imencode('.jpg', frame)  # JPEG 형식으로 인코딩

    data = frame.tobytes()  # 전송을 위해 바이트 형태로 변환
    size = len(data)
    connection.sendall(struct.pack(">L", size) + data)  # 데이터 크기와 데이터 전송
