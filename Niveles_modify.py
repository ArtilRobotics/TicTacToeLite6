import random
from inference import get_model
import supervision as sv
import os
import cv2
import numpy as np
import sys
import math
import time
import queue
import datetime
import traceback
import threading
from xarm import version
from xarm.wrapper import XArmAPI

# Load a pre-trained yolov8n model
model = get_model(model_id="tic-tac-toe-fctyp/1")

# Create supervision annotators
bounding_box_annotator = sv.BoundingBoxAnnotator()
label_annotator = sv.LabelAnnotator()

# Capture video from the default camera (camera index 0)
cap = cv2.VideoCapture(1)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# Define a callback function for the trackbar (does nothing but required by OpenCV)
def nothing(x):
    pass

# Create a window
cv2.namedWindow('Annotated Frame')

# Create a trackbar for adjusting focus
cv2.createTrackbar('Focus', 'Annotated Frame', 0, 255, nothing)

# Initialize the game board (3x3 grid)
board = [['' for _ in range(3)] for _ in range(3)]
difficulty = 1
winner = None
winning_line = None
player_marker = ''
computer_marker = ''
winning_flag = False

# Define the initial ROI (x_min, y_min, x_max, y_max)
roi = [215, 118, 431, 342]

# Create trackbars for modifying ROI
cv2.createTrackbar('X Min', 'Annotated Frame', roi[0], 640, nothing)
cv2.createTrackbar('Y Min', 'Annotated Frame', roi[1], 480, nothing)
cv2.createTrackbar('X Max', 'Annotated Frame', roi[2], 640, nothing)
cv2.createTrackbar('Y Max', 'Annotated Frame', roi[3], 480, nothing)

# Function to draw the 3x3 grid within the ROI
def draw_grid(frame, bbox):
    x_min, y_min, x_max, y_max = bbox
    cell_height = (y_max - y_min) // 3
    cell_width = (x_max - x_min) // 3

    for i in range(1, 3):
        cv2.line(frame, (x_min, y_min + i * cell_height), (x_max, y_min + i * cell_height), (255, 255, 255), 2)
        cv2.line(frame, (x_min + i * cell_width, y_min), (x_min + i * cell_width, y_max), (255, 255, 255), 2)

# Function to place "x" or "o" on the grid
def place_marker(frame, marker, position, bbox):
    x_min, y_min, x_max, y_max = bbox
    cell_height = (y_max - y_min) // 3
    cell_width = (x_max - x_min) // 3

    x, y = position
    center_x = x_min + x * cell_width + cell_width // 2
    center_y = y_min + y * cell_height + cell_height // 2

    if marker == 'x':
        cv2.line(frame, (center_x - 20, center_y - 20), (center_x + 20, center_y + 20), (0, 0, 255), 5)
        cv2.line(frame, (center_x + 20, center_y - 20), (center_x - 20, center_y + 20), (0, 0, 255), 5)
    elif marker == 'o':
        cv2.circle(frame, (center_x, center_y), 20, (0, 255, 0), 5)

# Function to draw the winning line
def draw_winning_line(frame, bbox, line):
    global winning_flag
    x_min, y_min, x_max, y_max = bbox
    cell_height = (y_max - y_min) // 3
    cell_width = (x_max - x_min) // 3

    start_cell, end_cell = line

    if winning_flag == False:
        print(start_cell,end_cell)
        robot_main.winner_line(start_cell[0],start_cell[1])
        robot_main.finish_line(end_cell[0],end_cell[1])
        robot_main.camara()
        winning_flag=True


    start_x = x_min + start_cell[0] * cell_width + cell_width // 2
    start_y = y_min + start_cell[1] * cell_height + cell_height // 2
    end_x = x_min + end_cell[0] * cell_width + cell_width // 2
    end_y = y_min + end_cell[1] * cell_height + cell_height // 2

    cv2.line(frame, (start_x, start_y), (end_x, end_y), (0, 255, 255), 5)

# Function for computer to play the opposite marker
def computer_play():
    best_score = -float('inf')
    best_move = None
    for y in range(3):
        for x in range(3):
            if board[y][x] == '':
                board[y][x] = computer_marker
                score = minimax(board, 0, False, difficulty)
                board[y][x] = ''
                if score > best_score:
                    best_score = score
                    best_move = (x, y)
    if best_move:
        board[best_move[1]][best_move[0]] = computer_marker
        print(computer_marker)
        x,y=best_move
        time.sleep(1)
        robot_main.mov_robot(x,y)
        return best_move
    return None

def minimax(board, depth, is_maximizing, depth_limit):
    scores = {computer_marker: 1, player_marker: -1, 'tie': 0}
    result, _ = check_winner()
    if result:
        return scores[result]

    if depth >= depth_limit:
        return 0

    if is_maximizing:
        best_score = -float('inf')
        for y in range(3):
            for x in range(3):
                if board[y][x] == '':
                    board[y][x] = computer_marker
                    score = minimax(board, depth + 1, False, depth_limit)
                    board[y][x] = ''
                    best_score = max(score, best_score)
        return best_score
    else:
        best_score = float('inf')
        for y in range(3):
            for x in range(3):
                if board[y][x] == '':
                    board[y][x] = player_marker
                    score = minimax(board, depth + 1, True, depth_limit)
                    board[y][x] = ''
                    best_score = min(score, best_score)
        return best_score

def check_winner():
    # Check rows, columns, and diagonals for a winner
    for i in range(3):
        if board[i][0] == board[i][1] == board[i][2] != '':
            return board[i][0], [(0, i), (2, i)]
        if board[0][i] == board[1][i] == board[2][i] != '':
            return board[0][i], [(i, 0), (i, 2)]
    if board[0][0] == board[1][1] == board[2][2] != '':
        return board[0][0], [(0, 0), (2, 2)]
    if board[0][2] == board[1][1] == board[2][0] != '':
        return board[0][2], [(2, 0), (0, 2)]
    if all(board[y][x] != '' for y in range(3) for x in range(3)):
        return 'tie', None
    return None, None

class RobotMain(object):
    """Robot Main Class"""
    def __init__(self, robot, **kwargs):
        self.alive = True
        self._arm = robot
        self._tcp_speed = 100
        self._tcp_acc = 2000
        self._angle_speed = 100
        self._angle_acc = 500
        self._vars = {}
        self._funcs = {}
        self._robot_init()

    # Robot init
    def _robot_init(self):
        self._arm.clean_warn()
        self._arm.clean_error()
        self._arm.motion_enable(True)
        self._arm.set_mode(0)
        self._arm.set_state(0)
        time.sleep(1)
        self._arm.register_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.register_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'register_count_changed_callback'):
            self._arm.register_count_changed_callback(self._count_changed_callback)

    # Register error/warn changed callback
    def _error_warn_changed_callback(self, data):
        if data and data['error_code'] != 0:
            self.alive = False
            self.pprint('err={}, quit'.format(data['error_code']))
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)

    # Register state changed callback
    def _state_changed_callback(self, data):
        if data and data['state'] == 4:
            self.alive = False
            self.pprint('state=4, quit')
            self._arm.release_state_changed_callback(self._state_changed_callback)

    # Register count changed callback
    def _count_changed_callback(self, data):
        if self.is_alive:
            self.pprint('counter val: {}'.format(data['count']))

    def _check_code(self, code, label):
        if not self.is_alive or code != 0:
            self.alive = False
            ret1 = self._arm.get_state()
            ret2 = self._arm.get_err_warn_code()
            self.pprint('{}, code={}, connected={}, state={}, error={}, ret1={}. ret2={}'.format(label, code, self._arm.connected, self._arm.state, self._arm.error_code, ret1, ret2))
        return self.is_alive

    @staticmethod
    def pprint(*args, **kwargs):
        try:
            stack_tuple = traceback.extract_stack(limit=2)[0]
            print('[{}][{}] {}'.format(time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time())), stack_tuple[1], ' '.join(map(str, args))))
        except:
            print(*args, **kwargs)

    @property
    def arm(self):
        return self._arm

    @property
    def VARS(self):
        return self._vars

    @property
    def FUNCS(self):
        return self._funcs

    @property
    def is_alive(self):
        if self.alive and self._arm.connected and self._arm.error_code == 0:
            if self._arm.state == 5:
                cnt = 0
                while self._arm.state == 5 and cnt < 5:
                    cnt += 1
                    time.sleep(0.1)
            return self._arm.state < 4
        else:
            return False

    # Robot Main Run
    def tablero(self):
        try:
            code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            code = self._arm.set_servo_angle(angle=[-10.8, 68.9, 58.8, 170.8, 99.7, -175.7], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            code = self._arm.set_position(*[350.8, -47.2, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[350.8, 151.7, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[350.8, 151.7, 77.0, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_servo_angle(angle=[27.9, 79.4, 92.2, 210.3, 78.6, -180.5], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            code = self._arm.set_position(*[415.8, 151.7, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[415.8, -42.5, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[415.8, -42.5, 77.0, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_servo_angle(angle=[3.2, 84.8, 104.8, 185.2, 69.8, -175.9], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            code = self._arm.set_position(*[480.9, 16.6, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[289.8, 16.6, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[289.8, 16.6, 77.0, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_servo_angle(angle=[26.5, 67.1, 46.8, 209.7, 107.5, -164.2], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            code = self._arm.set_position(*[289.8, 79.8, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[476.5, 79.8, 63.4, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_position(*[476.5, 79.8, 77.0, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)

    def camara(self):
        try:
            code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

        except Exception as e:
            self.pprint('MainException: {}'.format(e))
        self.alive = False
        self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
        self._arm.release_state_changed_callback(self._state_changed_callback)
        if hasattr(self._arm, 'release_count_changed_callback'):
            self._arm.release_count_changed_callback(self._count_changed_callback)

    def cruz(self,x,y):
        z_aprox = 77
        z_draw = 63.4
        code = self._arm.set_position(*[x+10, y-10, z_aprox, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x+10, y-10, z_draw, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x-10, y+10, z_draw, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x-10, y+10, z_aprox, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x-10, y-10, z_aprox, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x-10, y-10, z_draw, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x+10, y+10, z_draw, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

        code = self._arm.set_position(*[x+10, y+10, z_aprox, -92.6, -84.1, -89.1], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
        

    def mov_robot(self, coord_y, coord_i):
        if coord_i == 0 and coord_y==0:
            try:
                x_var=384.6+60
                y_var=44.0+60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 0 and coord_y==1:
            try:
                x_var=384.6+60
                y_var=44.0
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 0 and coord_y==2:
            try:
                x_var=384.6+60
                y_var=44.0-60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 1 and coord_y==0:
            try:
                x_var=384.6
                y_var=44.0+60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 1 and coord_y==1:
            try:
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                robot_main.cruz(384.6,44.0)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 1 and coord_y==2:
            try:
                x_var=384.6
                y_var=44.0-60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==0:
            try:
                x_var=384.6-60
                y_var=44.0+60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==1:
            try:
                x_var=384.6-60
                y_var=44.0
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==2:
            try:
                x_var=384.6-60
                y_var=44.0-60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                robot_main.cruz(x_var, y_var)
                code = self._arm.set_servo_angle(angle=[-20.2, 15.5, 118.3, 55.7, 22.5, -47.8], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        time.sleep(2)

    
    def winner_line(self, coord_y, coord_i):
        z_draw = 63.4
        if coord_i == 0 and coord_y==0:
            try:
                x_var=384.6+60
                y_var=44.0+60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 0 and coord_y==1:
            try:
                x_var=384.6+60
                y_var=44.0
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
               

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 0 and coord_y==2:
            try:
                x_var=384.6+60
                y_var=44.0-60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
               
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 1 and coord_y==0:
            try:
                x_var=384.6
                y_var=44.0+60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                
               
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        
        elif coord_i == 1 and coord_y==2:
            try:
                x_var=384.6
                y_var=44.0-60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==0:
            try:
                x_var=384.6-60
                y_var=44.0+60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==1:
            try:
                x_var=384.6-60
                y_var=44.0
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)                
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==2:
            try:
                x_var=384.6-60
                y_var=44.0-60
                code = self._arm.set_servo_angle(angle=[10.3, 71.6, 69.0, 192.0, 92.3, -173.6], speed=self._angle_speed, mvacc=self._angle_acc, wait=True, radius=0.0)
                code = self._arm.set_position(*[x_var, y_var, 74.6, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
    

    def finish_line(self, coord_y, coord_i):
        z_draw = 63.4
        if coord_i == 0 and coord_y==0:
            try:
                x_var=384.6+60
                y_var=44.0+60
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 0 and coord_y==1:
            try:
                x_var=384.6+60
                y_var=44.0
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
               

            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 0 and coord_y==2:
            try:
                x_var=384.6+60
                y_var=44.0-60
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
               
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 1 and coord_y==0:
            try:
                x_var=384.6
                y_var=44.0+60
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                
               
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        
        elif coord_i == 1 and coord_y==2:
            try:
                x_var=384.6
                y_var=44.0-60
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==0:
            try:
                x_var=384.6-60
                y_var=44.0+60
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
                
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==1:
            try:
                x_var=384.6-60
                y_var=44.0
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)                
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)
        elif coord_i == 2 and coord_y==2:
            try:
                x_var=384.6-60
                y_var=44.0-60
                code = self._arm.set_position(*[x_var, y_var, z_draw, -92.4, -84.1, -89.3], speed=self._tcp_speed, mvacc=self._tcp_acc, radius=0.0, wait=True)
            
            except Exception as e:
                self.pprint('MainException: {}'.format(e))
            self.alive = False
            self._arm.release_error_warn_changed_callback(self._error_warn_changed_callback)
            self._arm.release_state_changed_callback(self._state_changed_callback)
            if hasattr(self._arm, 'release_count_changed_callback'):
                self._arm.release_count_changed_callback(self._count_changed_callback)


RobotMain.pprint('xArm-Python-SDK Version:{}'.format(version.__version__))
arm = XArmAPI('192.168.1.179', baud_checkset=False)
robot_main = RobotMain(arm)

while True:
    # Read a frame from the camera
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    # Get the current position of the trackbars for ROI
    roi[0] = cv2.getTrackbarPos('X Min', 'Annotated Frame')
    roi[1] = cv2.getTrackbarPos('Y Min', 'Annotated Frame')
    roi[2] = cv2.getTrackbarPos('X Max', 'Annotated Frame')
    roi[3] = cv2.getTrackbarPos('Y Max', 'Annotated Frame')

    # Get the current position of the trackbar for focus
    focus_value = cv2.getTrackbarPos('Focus', 'Annotated Frame')

    # Set the camera focus (if supported)
    cap.set(cv2.CAP_PROP_FOCUS, focus_value)

    # Run inference on the current frame
    results = model.infer(frame)[0]

    # Load the results into the supervision Detections API
    detections = sv.Detections.from_inference(results)

    # Annotate the frame with our inference results
    annotated_frame = bounding_box_annotator.annotate(
        scene=frame, detections=detections)
    annotated_frame = label_annotator.annotate(
        scene=annotated_frame, detections=detections)

    # Draw the 3x3 grid within the ROI
    draw_grid(annotated_frame, roi)

    # Place markers based on detections
    for detection in detections:
        bbox = detection[0]
        label = detection[5]['class_name']

        if label in ['x', 'o']:
            x, y = bbox[0], bbox[1]
            cell_x = int((x - roi[0]) // ((roi[2] - roi[0]) // 3))
            cell_y = int((y - roi[1]) // ((roi[3] - roi[1]) // 3))

            # Ensure cell_x and cell_y are within bounds
            cell_x = min(2, max(0, cell_x))
            cell_y = min(2, max(0, cell_y))

            if board[cell_y][cell_x] == '':
                if player_marker == '':
                    player_marker = label
                    computer_marker = 'o' if player_marker == 'x' else 'x'

                board[cell_y][cell_x] = label
                place_marker(annotated_frame, label, (cell_x, cell_y), roi)

                # Check for a winner after player's move
                winner, winning_line = check_winner()
                if winner:
                    break

                # Computer plays with the opposite marker
                computer_position = computer_play()
                if computer_position:
                    place_marker(annotated_frame, computer_marker, computer_position, roi)

                    # Check for a winner after computer's move
                    winner, winning_line = check_winner()
                    if winner:
                        break

    # Place existing markers on the grid
    for y in range(3):
        for x in range(3):
            if board[y][x] != '':
                place_marker(annotated_frame, board[y][x], (x, y), roi)

    # Draw the winning line if there is one
    if winning_line:
        draw_winning_line(annotated_frame, roi, winning_line)


    # Display the annotated frame
    cv2.imshow('Annotated Frame', annotated_frame)

    # Display winner if there is one
    if winner:
        cv2.putText(annotated_frame, f'{winner.upper()} wins!', (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('Annotated Frame', annotated_frame)

    # Handle key presses
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('1'):
        board = [['' for _ in range(3)] for _ in range(3)]
        player_marker = ''
        computer_marker = ''
        difficulty = 1  # Easy difficulty, depth limit 1
        winner = None
        winning_line = None
        winning_flag=False
    elif key == ord('2'):
        board = [['' for _ in range(3)] for _ in range(3)]
        player_marker = ''
        computer_marker = ''
        difficulty = 3  # Medium difficulty, depth limit 3
        winner = None
        winning_line = None
        winning_flag=False
    elif key == ord('3'):
        board = [['' for _ in range(3)] for _ in range(3)]
        player_marker = ''
        computer_marker = ''
        difficulty = 9  # Hard difficulty, no depth limit, full minimax
        winner = None
        winning_line = None
        winning_flag=False
    elif key == ord('4'):
        board = [['' for _ in range(3)] for _ in range(3)]
        player_marker = 'o'
        computer_marker = 'x'
        winning_flag=False
        difficulty = 1  # Easy difficulty, depth limit 1
        computer_position = computer_play()
        if computer_position:
            place_marker(annotated_frame, 'x', computer_position, roi)
    elif key == ord('5'):
        board = [['' for _ in range(3)] for _ in range(3)]
        player_marker = 'o'
        computer_marker = 'x'
        winning_flag=False
        difficulty = 3  # Medium difficulty, depth limit 3
        computer_position = computer_play()
        if computer_position:
            place_marker(annotated_frame, 'x', computer_position, roi)
    elif key == ord('6'):
        board = [['' for _ in range(3)] for _ in range(3)]
        player_marker = 'o'
        computer_marker = 'x'
        winning_flag=False
        difficulty = 9  # Hard difficulty, no depth limit, full minimax
        computer_position = computer_play()
        if computer_position:
            place_marker(annotated_frame, 'x', computer_position, roi)
    elif key == ord('t'):  # Presionar 's' para activar la detección de círculos
        robot_main.tablero()
    elif key == ord('c'):  # Presionar 's' para activar la detección de círculos
        robot_main.camara()
        
# Release the camera and close the window
cap.release()
cv2.destroyAllWindows()
