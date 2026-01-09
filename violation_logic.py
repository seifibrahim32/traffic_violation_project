import cv2
import numpy as np


def is_red_light(frame, roi):
    x, y, w, h = roi
    light = frame[y:y + h, x:x + w]

    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)
    red1 = cv2.inRange(hsv, (0, 120, 70), (10, 255, 255))
    red2 = cv2.inRange(hsv, (170, 120, 70), (180, 255, 255))

    return cv2.countNonZero(red1 + red2) > 100


def crossed_stop_line(centroid_y, stop_line_y):
    return centroid_y > stop_line_y
