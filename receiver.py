import cv2
import socket
import struct
import numpy as np

# YOLO 모델 로드
net = cv2.dnn.readNet("/home/addinedu/amr_ws/Yolo/yolov3.weights", "/home/addinedu/amr_ws/Yolo/yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers()

# OpenCV 버전에 따라 반환 형태가 다를 수 있으므로, 반환 형태에 따라 처리
if output_layers_indices.ndim == 1:
    output_layers = [layer_names[i - 1] for i in output_layers_indices]
else:
    # OpenCV의 이전 버전에서는 배열의 배열 형태로 반환될 수 있으므로 이를 처리
    output_layers = [layer_names[i[0] - 1] for i in output_layers_indices]

# 이후 코드는 동일하게 사용

# 소켓 연결 설정
HOST_IP = "192.168.0.41"  # PC2의 자신의 IP 주소로 변경하세요.
PORT = 11111
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((HOST_IP, PORT))

# 데이터 수신 및 처리
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

    frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)  # 이미지 디코딩

    # 결과화면 표시
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' 키를 누르면 종료
        break
    message_1 = "Hi"
    client_socket.send(message_1.encode())

# 자원 해제
client_socket.close()
cv2.destroyAllWindows()
