import cv2
import numpy as np


# ================= LIGHT COLOR CLASSIFICATION =================
def classify_traffic_light_color(frame, box):
    x1, y1, x2, y2 = box
    light = frame[y1:y2, x1:x2]
    if light.size == 0:
        return "UNKNOWN"

    light = cv2.bilateralFilter(light, 5, 75, 75)
    hsv = cv2.cvtColor(light, cv2.COLOR_BGR2HSV)

    v = hsv[:, :, 2]
    if np.mean(v) < 90:
        hsv[:, :, 2] = cv2.normalize(v, None, 80, 255, cv2.NORM_MINMAX)

    h, w = hsv.shape[:2]
    h3 = h // 3

    zones = {
        "RED": hsv[0:h3],
        "YELLOW": hsv[h3:2 * h3],
        "GREEN": hsv[2 * h3:h]
    }

    ranges = {
        "RED": [((0, 70, 80), (10, 255, 255)),
                ((170, 70, 80), (180, 255, 255))],
        "YELLOW": [((18, 70, 80), (35, 255, 255))],
        "GREEN": [((40, 70, 80), (90, 255, 255))]
    }

    kernel = np.ones((3, 3), np.uint8)
    scores = {}

    for color, zone in zones.items():
        mask = np.zeros(zone.shape[:2], dtype=np.uint8)
        for low, high in ranges[color]:
            mask |= cv2.inRange(zone, low, high)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        scores[color] = cv2.countNonZero(mask)

    color, area = max(scores.items(), key=lambda x: x[1])

    return color


# ================= LINE CROSSING =================

def line_intersects_box(p1, p2, box):
    """
    p1, p2: stop line endpoints
    box: (x, y, w, h)
    """
    x, y, w, h = box

    # Bottom edge of vehicle
    bx1 = x
    by1 = y + h
    bx2 = w
    by2 = h

    def ccw(A, B, C):
        return (C[1]-A[1])*(B[0]-A[0]) > (B[1]-A[1])*(C[0]-A[0])

    def intersect(A, B, C, D):
        return ccw(A, C, D) != ccw(B, C, D) and ccw(A, B, C) != ccw(A, B, D)

    return intersect(p1, p2, (bx1, by1), (bx2, by2))