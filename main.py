import cv2
import numpy as np
import threading
from collections import deque

from vehicle_detection_threading import detection_worker, done_event, det_event
from traffic_light_detector import TrafficLightsDetector
from violation_logic import classify_traffic_light_color, crossed_line

# ================= CONFIG =================
VIDEO_PATH = "videos/8.mp4"
FRAME_SKIP = 19

# ================= INIT =================
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.setUseOptimized(True)
cv2.setNumThreads(4)

traffic_light_detector = TrafficLightsDetector()

multi_tracker = cv2.legacy.MultiTracker.create()

shared_frame = None
detections = []

threading.Thread(target=detection_worker, daemon=True).start()

# ================= ROI SELECTION =================
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")
first_frame = cv2.resize(first_frame, (900, 600))

# ---- STOP LINE SELECTION ----
stop_line_pts = []


def select_stop_line(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN and len(stop_line_pts) < 2:
        stop_line_pts.append((x, y))
        cv2.circle(first_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Stop Line", first_frame)


print("🔴 Click TWO points for STOP LINE")
cv2.imshow("Select Stop Line", first_frame)
cv2.setMouseCallback("Select Stop Line", select_stop_line)
while len(stop_line_pts) < 2:
    cv2.waitKey(1)
cv2.destroyWindow("Select Stop Line")
p1, p2 = stop_line_pts

# ---- TRAFFIC LIGHT ROI SELECTION ----
traffic_roi = []  # [x1, y1, x2, y2]
drawing = False
ix, iy = -1, -1


def draw_roi(event, x, y, flags, param):
    global ix, iy, drawing, traffic_roi, first_frame
    temp_frame = first_frame.copy()
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        ix, iy = x, y
    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        cv2.rectangle(temp_frame, (ix, iy), (x, y), (255, 255, 0), 2)
        cv2.imshow("Select Traffic Light ROI", temp_frame)
    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        traffic_roi = [ix, iy, x, y]
        cv2.rectangle(temp_frame, (ix, iy), (x, y), (0, 255, 255), 2)
        cv2.imshow("Select Traffic Light ROI", temp_frame)


print("🟢 Draw a rectangle around the TRAFFIC LIGHT and release mouse")
cv2.imshow("Select Traffic Light ROI", first_frame)
cv2.setMouseCallback("Select Traffic Light ROI", draw_roi)
while len(traffic_roi) < 4:
    cv2.waitKey(1)
cv2.destroyWindow("Select Traffic Light ROI")

x1, y1, x2, y2 = traffic_roi
print("✅ Traffic light ROI:", (x1, y1, x2, y2))

# ---- INITIALIZE LIGHT STATE ----
light_history = deque(maxlen=7)
# Detect initial traffic light inside ROI
roi_frame = first_frame[y1:y2, x1:x2]
yolo_boxes = traffic_light_detector.detect(roi_frame)
if yolo_boxes:
    bx1, by1, bx2, by2 = yolo_boxes
    refined_box = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
else:
    refined_box = (x1, y1, x2, y2)

initial_state = classify_traffic_light_color(first_frame, refined_box)
light_history.append(initial_state)
light_state = initial_state

# ================= TRACK STATE =================
track_history = {}
violations = set()
frame_count = 0

# ================= MAIN LOOP =================
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_count += 1
    if frame_count % FRAME_SKIP != 0:
        continue

    frame = cv2.resize(frame, (900, 600))

    # ---------- VEHICLE TRACKING ----------
    success, boxes = multi_tracker.update(frame)
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255
    if success:
        for box in boxes:
            x, y, bw, bh = map(int, box)
            cv2.rectangle(mask, (x, y), (x + bw, y + bh), 0, -1)
    else:
        multi_tracker = cv2.legacy.MultiTracker.create()

    # ---------- VEHICLE DETECTION ----------
    shared_frame = cv2.bitwise_and(frame, frame, mask=mask)
    done_event.clear()
    det_event.set()
    done_event.wait()

    for box in detections:
        x1, y1, x2, y2 = box
        bw, bh = x2 - x1, y2 - y1
        if bw <= 0 or bh <= 0:
            continue
        tracker = cv2.legacy.TrackerMOSSE.create()
        multi_tracker.add(tracker, frame, (x1, y1, bw, bh))

    # ---------- TRAFFIC LIGHT DETECTION INSIDE ROI ----------
    roi_frame = frame[y1:y2, x1:x2]
    yolo_boxes = traffic_light_detector.detect(roi_frame)
    if yolo_boxes:
        bx1, by1, bx2, by2 = yolo_boxes[0]
        traffic_light_box = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
    else:
        traffic_light_box = (x1, y1, x2, y2)

    # Traffic light color
    state = classify_traffic_light_color(frame, traffic_light_box)
    light_history.append(state)
    light_state = max(set(light_history), key=light_history.count)

    # ---------- DISPLAY TRAFFIC LIGHT ----------
    x1_t, y1_t, x2_t, y2_t = traffic_light_box
    cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 255), 2)
    color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0), "UNKNOWN": (50, 50, 50)}
    cv2.rectangle(frame, (x1_t, y1_t - 30), (x2_t, y1_t - 10), color_map.get(light_state, (50, 50, 50)), -1)
    cv2.putText(frame, light_state, (x1_t, y1_t - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # ---------- DISPLAY VEHICLES + VIOLATIONS ----------
    for i, box in enumerate(boxes):
        x, y, bw, bh = map(int, box)
        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)
        cv2.putText(frame, f"Vehicle {i}", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

        bottom_center_curr = (x + bw // 2, y + bh)
        if i in track_history:
            bottom_center_prev = track_history[i]
            if crossed_line(bottom_center_prev, bottom_center_curr, p1, p2) and light_state in ("RED", "YELLOW"):
                violations.add(i)
                cv2.putText(frame, "VIOLATION!", (x, y - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            track_history[i] = bottom_center_curr
        else:
            track_history[i] = bottom_center_curr

    # ---------- DRAW STOP LINE ----------
    cv2.line(frame, p1, p2, (0, 0, 255), 2)

    # ---------- SHOW FRAME ----------
    cv2.imshow("Traffic Light Violation Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# ================= CLEANUP =================
stop_thread = True
det_event.set()
cap.release()
cv2.destroyAllWindows()
print("✅ Finished successfully")
