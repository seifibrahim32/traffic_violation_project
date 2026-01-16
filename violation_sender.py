import requests
import cv2


def send_violation(frame, box):
    x, y, w, h = box
    if x <= 0 or y <= 0 or w <= 30 or h <= 30:
        return
    crop = frame[y: h, x:w]

    _, img_encoded = cv2.imencode(".jpg", crop)

    files = {
        'image': ('violation.jpg', img_encoded.tobytes(), 'image/jpeg')
    }

    data = {
        'light_state': 'RED',
        'camera_id': 'CAM_01',
        'x': x,
        'y': y,
        'w': w,
        'h': h
    }

    requests.post(
        "http://127.0.0.1:8000/api/violations/",
        data=data,
        files=files
    )
