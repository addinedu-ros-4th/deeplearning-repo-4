import cv2
import socket
import struct
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
import json
from auto import *
from chess_game import *
from alpha_zero_model.config import Config
from alpha_zero_model.chess_env import ChessEnv
from alpha_zero_model.player_chess import ChessPlayer
from alpha_zero_model.model_chess import ChessModel

from queue import Queue
import threading

from supervision import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette

def data_receiving_thread(client_socket, frame_queue):
    payload_size = struct.calcsize(">L")
    data = b""

    while True:
        while len(data) < payload_size:
            packet = client_socket.recv(4096)
            if not packet: break
            data += packet

        if not data:
            break

        packed_msg_size = data[:payload_size]
        data = data[payload_size:]
        msg_size = struct.unpack(">L", packed_msg_size)[0]

        while len(data) < msg_size:
            data += client_socket.recv(4096)

        frame_data = data[:msg_size]
        data = data[msg_size:]

        frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        frame_queue.put(frame)

 

def main():
    with open("data.yaml") as file:
        data = yaml.load(file, Loader= yaml.FullLoader)
        HOST_IP = data["ip"]
        PORT = data["port"]
        yolo_model_path = data["yolo_model_path"]
        chess_model_path = data["chess_model_path"]

    img = Img()

    yolo_model = YOLO(yolo_model_path)
    config = Config()
    chess_model = ChessModel(config)
    chess_model.build()
    chess_model.load(chess_model_path)

    if not chess_model.load(chess_model_path):
        raise RuntimeError("Failed to load the trained model weights")
    
    class_mapping = {
    0: "chessboard",
    1: "b",
    2: "k",
    3: "n",
    4: "p",
    5: "q",
    6: "r",
    7: "B",
    8: "K",
    9: "N",
    10: "P",
    11: "Q",
    12: "R"
    }

    count = 0
    x_left = 0
    y_top = 0
    width = 0
    height = 0
    prev_fen = None
    fen = None
    frame_queue = Queue(maxsize = 1)
    box_annotator = BoxAnnotator(color = ColorPalette.default(), thickness = 2, text_thickness = 1, text_scale = 0.3)
    chess_player = ChessPlayer(config, chess_model.get_pipes(config.play.search_threads))
    env = ChessEnv().reset()
    is_turn = True
    is_moving = False
    is_start = False

    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    client_socket.connect((HOST_IP, PORT))

    receiving_thread = threading.Thread(target=data_receiving_thread, args=(client_socket, frame_queue))
    receiving_thread.start()
    
    while True:
        if not frame_queue.empty():
            rgb = frame_queue.get()
            gray = cv2.cvtColor(rgb, cv2.COLOR_BGR2GRAY)
            ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

            results = yolo_model(source= rgb, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], verbose = False, conf = 0.5)
            boxes = results[0].boxes
            xywh_np = boxes.xywh.cpu().numpy()
            xyxy_np = boxes.xyxy.cpu().numpy()
            classes_np = boxes.cls.cpu().numpy()
            confidence_np =  boxes.conf.cpu().numpy()
            class_name = results[0].names
            detections = Detections(
                xyxy = xyxy_np,
                confidence = confidence_np,
                class_id = classes_np.astype(int)
            )

            labels = [
                f"{class_name[class_id]} {confidence:0.2f}"
                for _, _, confidence, class_id, _, _
                in detections
            ]

            frame = box_annotator.annotate(scene = rgb, detections = detections, labels= labels)

            for cnt in contours:
                if cv2.contourArea(cnt) > 500000:
                    x, y, width, height = cv2.boundingRect(cnt)
                    ratio = width / height 
                    if 0.9 < ratio < 1.1:
                        x_left, y_top, width, height = cv2.boundingRect(cnt)
                        cropped = frame[y_top : y_top + height, x_left : x_left + width]
                        cv2.imshow('frame', cropped)
                        

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
        

            vectorized_mapping = np.vectorize(class_mapping.get)

            mapped_classes_np = vectorized_mapping(classes_np)
            xy_np = xywh_np[:, :2]
            positions = []
            for x,y in xy_np:
                position = pixel_to_chess_coord(int(x), int(y), (x_left, y_top), 100)
                positions.append(position)
            positions = np.array(positions)
            pieces_positions = dict(zip(positions, mapped_classes_np))
            fen = create_fen_from_positions(pieces_positions)

        if prev_fen == None:
            prve_fen = fen

        if is_turn == True and fen == env.board.fen().split(' ')[0]:
            count = 0
            action = chess_player.action(env)
            env.step(action)
            from_x, from_y, to_x, to_y = img.automouse(action, (x_left, y_top), width // 8)
            pixel_data = {"from_x" : from_x, "from_y" : from_y, "to_x" : to_x, "to_y" : to_y}
            json_pixel_data = json.dumps(pixel_data) 
            client_socket.sendall(json_pixel_data.encode('utf-8'))
            is_moving = True
            is_turn = False
            if action:
                is_start = True
        elif is_moving == True and fen == env.board.fen().split(' ')[0]:
            is_moving = False
        elif is_moving == False and fen != env.board.fen().split(' ')[0] and is_start == True:
            if prev_fen == fen:
                count += 1
                if count >= 5000:
                    changes = compare_positions(fen, env.board.fen().split(' ')[0])
                    print(f"changes : {changes}")
                    env.step(changes)
                    is_turn = True
                    count = 0

        prev_fen = fen


    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()