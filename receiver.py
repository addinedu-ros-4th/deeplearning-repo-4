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

    # # YOLO를 사용한 객체 인식
    # height, width, channels = frame.shape
    # blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    # net.setInput(blob)
    # outs = net.forward(output_layers)

    # # 인식된 객체 정보 처리
    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         if confidence > 0.5:  # 정확도가 50% 이상인 경우만 처리
    #             # 객체 인식 처리 로직
    #             pass

    cv2.imshow('frame', frame)
        # 인식된 객체 정보 처리
    # for out in outs:
    #     for detection in out:
    #         scores = detection[5:]
    #         class_id = np.argmax(scores)
    #         confidence = scores[class_id]
    #         if confidence > 0.5:  # 정확도가 50% 이상인 경우만 처리
    #             # 객체의 위치 정보 계산
    #             center_x = int(detection[0] * width)
    #             center_y = int(detection[1] * height)
    #             w = int(detection[2] * width)
    #             h = int(detection[3] * height)

    #             # 객체의 사각형 테두리 계산
    #             x = int(center_x - w / 2)
    #             y = int(center_y - h / 2)

    #             # 인식된 객체에 대해 사각형 및 텍스트로 표시
    #             cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
    #             cv2.putText(frame, f"ID: {class_id}, Confidence: {confidence:.2f}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # # 결과 화면 표시
    cv2.imshow('frame', frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):  # 'q' 키를 누르면 종료
        break
    message_1 = "Hi"
    client_socket.send(message_1.encode())

# 자원 해제
client_socket.close()
cv2.destroyAllWindows()
