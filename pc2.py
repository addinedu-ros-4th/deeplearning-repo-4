import cv2
import socket
import struct
import numpy as np
import yaml

from ultralytics import YOLO

def main():

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

        cv2.imshow('frame', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break

    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()