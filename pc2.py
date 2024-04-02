import cv2
import socket
import struct
import numpy as np
import yaml
from PIL import Image
from ultralytics import YOLO
import torch

from auto import *
from chess_game import *

from alpha_zero_model.config import Config
from alpha_zero_model.chess_env import ChessEnv
from alpha_zero_model.player_chess import ChessPlayer
from alpha_zero_model.model_chess import ChessModel

import time

def find_move(board1, board2):
    for move in board1.legal_moves:
        board1.push(move)
        if board1.fen() == board2.fen():
            return move
        board1.pop()
    return None

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


    x_norm = 1920/1850
    y_norm = 1080/1053

    count = 0

    x_left = 0
    y_top = 0
    width = 0
    height = 0
    prev_fen = None

    pipe = chess_model.get_pipes(num = 18)
    chess_player = ChessPlayer(config,pipes= pipe)

    env = ChessEnv().reset()
    is_turn = True
    is_moving = False




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

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        ret, otsu = cv2.threshold(gray, -1,255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        contours, hierarchy = cv2.findContours(otsu, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE) 
        COLOR = (0, 200, 0) #Rectangle color
        
        results = yolo_model.predict(source= frame, classes = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12], verbose=False)
        frame = results[0].plot(font_size = 10, pil = True)

        for cnt in contours:
            if cv2.contourArea(cnt) > 600000:
                x, y, width, height = cv2.boundingRect(cnt)
                ratio = width / height 
                if 0.9 < ratio < 1.1:
                    x_left, y_top, width, height = cv2.boundingRect(cnt)
                    cropped = frame[y_top : y_top + height, x_left : x_left + width]
                    cv2.imshow('frame', cropped)
                    
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
    
        boxes = results[0].boxes
        xywh_np = boxes.xywh.cpu().numpy()
        classes_np = boxes.cls.cpu().numpy()

        vectorized_mapping = np.vectorize(class_mapping.get)

        mapped_classes_np = vectorized_mapping(classes_np)
        xy_np = xywh_np[:, :2]
        positions = []
        for x,y in xy_np:
            position = pixel_to_chess_coord(int(x), int(y), (x_left, y_top), width // 8)
            positions.append(position)
        positions = np.array(positions)
        pieces_positions = dict(zip(positions, mapped_classes_np))
        fen = create_fen_from_positions(pieces_positions)

        if prev_fen == None:
            prve_fen = fen

        if is_turn == True and fen == env.board.fen().split(' ')[0]:
            count = 0
            print(fen)
            print(env.board.fen().split(' ')[0])
            action = chess_player.action(env)
            env.step(action)
            print(f"move : {action}")
            from_x, from_y, to_x, to_y = img.automouse(action, (x_left, y_top), width // 8)
            pyautogui.click()
            pyautogui.moveTo(from_x * x_norm, from_y * y_norm)
            pyautogui.click()
            pyautogui.moveTo(to_x * x_norm, to_y * y_norm)
            pyautogui.click()
            print(env.board)
            is_moving = True
            print(is_moving)
            is_turn = False
        elif is_moving == True and fen == env.board.fen().split(' ')[0]:
            print("중간")
            is_moving = False
        elif is_moving == False and fen != env.board.fen().split(' ')[0]:
            pyautogui.moveTo(1900, 1000)
            if prev_fen == fen:
                count += 1
                if count >= 15:
                    changes = compare_positions(fen, env.board.fen().split(' ')[0])
                    print(f"black action {changes}")
                    env.step(changes)
                    print(env.board)
                    is_turn = True
                    count = 0

        prev_fen = fen


    client_socket.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()