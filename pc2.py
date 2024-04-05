import cv2
import chess
import sys
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
import threading
from supervision import Detections, BoxAnnotator
from supervision.draw.color import ColorPalette

from PyQt5 import QtWidgets, uic, QtGui, QtCore
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *

from_class = uic.loadUiType("chessAI.ui")[0]

class WindowClass(QMainWindow, from_class) :
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CHESS AI")
    
    def updateImage(self, image):
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(image.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(580, 580, Qt.KeepAspectRatio)
        self.chessScreen.setPixmap(QtGui.QPixmap.fromImage(p))
    
    def mycurrentMove(self, action):
        split_position = 2 
        move = action
        first_part = move[:split_position]
        second_part = move[split_position:]
        self.mymove.setText(f"{first_part}  to {second_part}")
        move = f"Addin-Pago: {first_part}  to {second_part}"
        myWindows.gameLog(move)

    def aicurrentMove(self, changes):
        split_position = 2 
        move = changes
        first_part = move[:split_position]
        second_part = move[split_position:]
        self.compmove.setText(f"{first_part} to {second_part}")
        move = f"Chess.com: {first_part} to {second_part}"
        myWindows.gameLog(move)
    
    def gameLog(self, moves):
        current_text = self.gamehistory.text()
        text = current_text+'\n' + moves
        self.gamehistory.setText(text)

    def captured(self, cur_piece, prev_piece):
        piece_map = {
            "p": "Black Pawn", "r": "Black Rook", "n": "Black Knight", 
            "b": "Black Bishop", "k": "Black King", "q": "Black Queen",
            "P": "White Pawn", "R": "White Rook", "N": "White Knight", 
            "B": "White Bishop", "K": "White King", "Q": "White Queen"
        }

        # 문자열로 변환
        cur_piece_str = str(cur_piece)
        prev_piece_str = str(prev_piece)

        # 말의 심볼을 이름으로 변환, 맵에 없는 경우 'Unknown piece'로 처리
        current = str(piece_map.get(cur_piece_str, "Unknown piece"))
        previous = str(piece_map.get(prev_piece_str, "Unknown piece"))

        if previous.islower():
            current_text = self.blackpieces.text()
            text = current_text + previous + ','
            self.blackpieces.setText(text)
        else:
            current_text = self.whitepieces.text()
            text = current_text + previous + ','
            self.whitepieces.setText(text)
        



def model_workers():
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
    box_annotator = BoxAnnotator(color = ColorPalette.default(), thickness = 2, text_thickness = 1, text_scale = 0.3)
    chess_player = ChessPlayer(config, chess_model.get_pipes(config.play.search_threads))
    env = ChessEnv().reset()
    is_turn = True
    is_moving = False
    is_start = False

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
        rgb = frame.copy()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
        COLOR = (0, 200, 0) #Rectangle color
        
        results = yolo_model.predict(source= frame, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], verbose=False, conf= 0.5)
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
                    if 'myWindows' in globals():
                        myWindows.updateImage(cv2.cvtColor(cropped, cv2.COLOR_BGR2RGB))     


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
            prev_fen = fen

        if is_turn == True and fen == env.board.fen().split(' ')[0]:
            count = 0
            action = chess_player.action(env)
            print(f"action :{action}")
            myWindows.mycurrentMove(action)
            prev_pos = action[2:]
            prev_pos = chess.parse_square(prev_pos)
            prev_piece = env.board.piece_at(prev_pos)
            env.step(action)
            cur_pos = action[2:]
            cur_pos = chess.parse_square(cur_pos)
            cur_piece = env.board.piece_at(cur_pos)
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
                if count >= 5:
                    changes = compare_positions(fen, env.board.fen().split(' ')[0])
                    print(f"changes : {changes}")
                    myWindows.aicurrentMove(changes)
                    prev_pos = changes[2:]
                    prev_pos = chess.parse_square(prev_pos)
                    prev_piece = env.board.piece_at(prev_pos)
                    env.step(changes)
                    cur_pos = changes[2:]
                    cur_pos = chess.parse_square(cur_pos)
                    cur_piece = env.board.piece_at(cur_pos)
                    is_turn = True
                    count = 0

        if prev_piece != None and prev_piece != cur_piece:
            print(f"{cur_piece}가 {prev_piece}를 잡았습니다.")
            myWindows.captured(cur_piece, prev_piece)

        prev_fen = fen


    client_socket.close()


if __name__ == "__main__":
    worker_thread = threading.Thread(target= model_workers)
    worker_thread.start()
    app = QApplication(sys.argv)
    myWindows= WindowClass()   
    myWindows.show()   
    sys.exit(app.exec_())