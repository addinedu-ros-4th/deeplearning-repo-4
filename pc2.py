import cv2
from collections import Counter
import sys
import socket
import struct
import numpy as np
import yaml
import chess
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

class WindowClass(QMainWindow, from_class):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.setWindowTitle("CHESS AI")
        self.client_thread = ClienThread()
        self.client_thread.updateImage.connect(self.updateImage)
        self.client_thread.start()
    
    def updateImage(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        h, w, ch = image.shape
        bytes_per_line = ch * w
        convert_to_Qt_format = QtGui.QImage(image.data.tobytes(), w, h, bytes_per_line, QtGui.QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(540, 540, Qt.KeepAspectRatio)
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

class ClienThread(QThread):
    updateImage = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()
        with open("data.yaml") as file:
            self.yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.client_socket.connect((self.yaml_data["ip"], self.yaml_data["port"]))
        self.yolo_model = YOLO(self.yaml_data["yolo_model_path"])
        self.config = Config()
        self.chess_model = ChessModel(self.config)
        self.chess_model.build()
        self.chess_model.load(self.yaml_data["chess_model_path"])
        self.env = ChessEnv().reset()
        self.img = Img()
        self.chess_player = ChessPlayer(self.config, self.chess_model.get_pipes(self.config.play.search_threads))
        self.box_annotator = BoxAnnotator(color = ColorPalette.default(), thickness = 2, text_thickness = 1, text_scale = 0.3)
        if not self.chess_model.load(self.yaml_data["chess_model_path"]):
            raise RuntimeError("Failed to load the trained model weights")

    def run(self):
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
        prev_yolo_fen = None
        yolo_fen = None
        is_turn = True
        is_moving = False
        start = False
        changes = None
        action_flag = False

        data = b""
        payload_size = struct.calcsize(">L")

        while True:
            while len(data) < payload_size:
                data += self.client_socket.recv(4096)
            
            packed_msg_size = data[:payload_size]
            data = data[payload_size:]
            msg_size = struct.unpack(">L", packed_msg_size)[0]

            while len(data) < msg_size:
                data += self.client_socket.recv(4096)
            
            frame_data = data[:msg_size]
            data = data[msg_size:]

            frame = cv2.imdecode(np.frombuffer(frame_data, np.uint8), cv2.IMREAD_COLOR)
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb = frame.copy()

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
            
            results = self.yolo_model.predict(source= frame, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], verbose=False, conf= 0.6)
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

            frame = self.box_annotator.annotate(scene = rgb, detections = detections, labels= labels)

            for cnt in contours:
                if cv2.contourArea(cnt) > 500000:
                    x, y, width, height = cv2.boundingRect(cnt)
                    ratio = width / height 
                    if 0.9 < ratio < 1.1:
                        x_left, y_top, width, height = cv2.boundingRect(cnt)
                        cropped = frame[y_top : y_top + height, x_left : x_left + width]
                        self.updateImage.emit(cropped)


            vectorized_mapping = np.vectorize(class_mapping.get, otypes= [str])

            mapped_classes_np = vectorized_mapping(classes_np)
            xy_np = xywh_np[:, :2]
            positions = []
            for x,y in xy_np:
                position = pixel_to_chess_coord(int(x), int(y), (x_left, y_top), 100)
                positions.append(position)
            positions = np.array(positions)
            pieces_positions = dict(zip(positions, mapped_classes_np))

            if yolo_fen is not None:
                prev_yolo_fen = yolo_fen

            yolo_fen = create_fen_from_positions(pieces_positions)
            chess_module_fen = self.env.board.fen().split(' ')[0]

            if yolo_fen == chess_module_fen:
                start = True

            if start == True:
                if is_turn == True and yolo_fen == chess_module_fen and is_moving == False:
                    action = self.chess_player.action(self.env, False)
                    print(f"white moves : {action}")
                    prev_pos = action[2:4]
                    prev_pos = chess.parse_square(prev_pos)
                    prev_piece = self.env.board.piece_at(prev_pos)
                    self.env.step(action)
                    action_flag = True
                    cur_pos = action[2:4]
                    cur_pos = chess.parse_square(cur_pos)
                    cur_piece = self.env.board.piece_at(cur_pos)
                    from_x, from_y, to_x, to_y = self.img.automouse(action, (x_left, y_top), width // 8)
                    pixel_data = {"from_x" : from_x, "from_y" : from_y, "to_x" : to_x, "to_y" : to_y}
                    is_moving = True
                    is_turn = False
                    if prev_piece != None and prev_piece != cur_piece:
                        print(f"{cur_piece}가 {prev_piece}를 잡았습니다.")
                elif is_moving == True and yolo_fen != chess_module_fen:

                    is_turn = False
                elif is_turn == False and chess_module_fen== yolo_fen:
                    is_moving = False
                elif is_turn == False and chess_module_fen!= yolo_fen and is_moving == False:
                    
                    if prev_yolo_fen == yolo_fen:
                        count += 1
                    if count > 10:
                        changes = compare_positions(yolo_fen, chess_module_fen)
                        if chess.Move.from_uci(changes) in self.env.board.legal_moves:
                            counter = Counter(list(results[0].boxes.cls.cpu().numpy()))
                            prev_classes_number = list(counter.items())
                            prev_classes_number = dict(map(lambda x: (class_name[x[0]],x[1]),prev_classes_number))
                            prev_pos = changes[2:4]
                            prev_pos = chess.parse_square(prev_pos)
                            prev_piece = self.env.board.piece_at(prev_pos)
                            if is_promotion_move(changes, self.env.board):
                                print(f"Promotion detected: {changes}")
                                counter = Counter(list(results[0].boxes.cls.cpu().numpy()))
                                cur_classes_number = list(counter.items())
                                cur_classes_number = dict(map(lambda x: (class_name[x[0]],x[1]),cur_classes_number))
                                promotion = identify_promotion(prev_classes_number, cur_classes_number)
                                changes += promotion  
                            print(f"black moves : {changes}")
                            self.env.step(changes)
                            cur_pos = changes[2:4]
                            cur_pos = chess.parse_square(cur_pos)
                            cur_piece = self.env.board.piece_at(cur_pos)
                            is_turn = True
                            if prev_piece != None and prev_piece != cur_piece:
                                print(f"{cur_piece}가 {prev_piece}를 잡았습니다.")

                        count = 0
                if self.env.board.is_game_over():
                    print("game is over")
                    print(self.env.board.result())
                    break
            if action_flag == True:
                json_pixel_data = json.dumps(pixel_data) 
                self.client_socket.sendall(json_pixel_data.encode('utf-8'))
                action_flag = False
            else:
                json_pixel_data = json.dumps("None")
                self.client_socket.sendall(json_pixel_data.encode('utf-8'))
    
    def stop(self):
        self.client_socket.close()
        self.terminate()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    myWindows= WindowClass()   
    myWindows.show()   
    sys.exit(app.exec_())