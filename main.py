
import threading
from collections import deque

import cv2
import numpy as np

from traffic_light_detector import TrafficLightsDetector
from vehicle_detector import VehicleDetector
from violation_logic import classify_traffic_light_color, line_intersects_box

# ================= CONFIG =================
VIDEO_PATH = "videos/8.mp4"
FRAME_SKIP = 15

# ================= INIT =================
cap = cv2.VideoCapture(VIDEO_PATH)
cv2.setUseOptimized(True)
cv2.setNumThreads(90)

traffic_light_detector = TrafficLightsDetector()
multi_trackers = []

detections = []

vehicle_detector = VehicleDetector()
# ================= VEHICLE DETECTION THREAD =================
shared_frame = None
det_event = threading.Event()
done_event = threading.Event()
stop_thread = False


def detection_worker():
    global detections
    while not stop_thread:
        det_event.wait()
        det_event.clear()
        if stop_thread:
            break
        detections = vehicle_detector.detect(shared_frame)
        done_event.set()


threading.Thread(target=detection_worker, daemon=True).start()

# ================= ROI SELECTION =================
ret, first_frame = cap.read()
if not ret:
    raise RuntimeError("Cannot read video")
first_frame = cv2.resize(first_frame, (900, 600))

# ---- STOP LINE SELECTION ----
cross_line_points = []


def select_stop_line(event, x, y, _, __):
    if event == cv2.EVENT_LBUTTONDOWN and len(cross_line_points) < 2:
        cross_line_points.append((x, y))
        cv2.circle(first_frame, (x, y), 5, (0, 0, 255), -1)
        cv2.imshow("Select Stop Line", first_frame)


print("🔴 Click TWO points for STOP LINE")
cv2.imshow("Select Stop Line", first_frame)
cv2.setMouseCallback("Select Stop Line", select_stop_line)
while len(cross_line_points) < 2:
    cv2.waitKey(1)
cv2.destroyWindow("Select Stop Line")
p1, p2 = cross_line_points
ref_point = (
    int((p1[0] + p2[0])),
    int((p1[1] + p2[1]) + 40)
)

# ---- TRAFFIC LIGHT ROI SELECTION ----
traffic_roi = []
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
roi_frame = first_frame[y1:y2, x1:x2]

# Enhance traffic light ROI for night-time
roi_frame_enhanced = cv2.convertScaleAbs(roi_frame, alpha=1.5, beta=30)

yolo_boxes = traffic_light_detector.detect(roi_frame_enhanced)
if yolo_boxes:
    # Pick largest detected box
    areas = [((bx2 - bx1) * (by2 - by1), (bx1, by1, bx2, by2)) for bx1, by1, bx2, by2 in yolo_boxes]
    _, largest_box = max(areas, key=lambda x: x[0])
    bx1, by1, bx2, by2 = largest_box
    refined_box = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
else:
    print("\nTraffic Light is a must\n")
    exit(0)

initial_state = classify_traffic_light_color(first_frame, refined_box)
light_history.append(initial_state)
light_state = initial_state

# ================= TRACK STATE =================
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

    # ---------- ENHANCE TRAFFIC LIGHT ROI ----------
    roi_frame = frame[y1:y2, x1:x2]
    roi_frame_enhanced = cv2.convertScaleAbs(roi_frame, alpha=1.5, beta=30)

    # ---------- TRAFFIC LIGHT DETECTION ----------
    yolo_boxes = traffic_light_detector.detect(roi_frame_enhanced)
    if yolo_boxes:
        areas = [((bx2 - bx1) * (by2 - by1), (bx1, by1, bx2, by2)) for bx1, by1, bx2, by2 in yolo_boxes]
        _, largest_box = max(areas, key=lambda x: x[0])
        bx1, by1, bx2, by2 = largest_box
        traffic_light_box = (x1 + bx1, y1 + by1, x1 + bx2, y1 + by2)
    else:
        traffic_light_box = (x1, y1, x2, y2)

    # ---------- TRAFFIC LIGHT COLOR ----------
    state = classify_traffic_light_color(frame, traffic_light_box)
    light_history.append(state)
    light_state = max(set(light_history), key=light_history.count)

    # ---------- DISPLAY TRAFFIC LIGHT ----------
    x1_t, y1_t, x2_t, y2_t = traffic_light_box
    cv2.rectangle(frame, (x1_t, y1_t), (x2_t, y2_t), (0, 255, 255), 2)
    color_map = {"RED": (0, 0, 255), "YELLOW": (0, 255, 255), "GREEN": (0, 255, 0)}
    cv2.rectangle(frame, (x1_t, y1_t - 30), (x2_t, y1_t - 10), color_map.get(light_state, (50, 50, 50)), -1)
    cv2.putText(frame, light_state, (x1_t, y1_t - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)

    # ---------- VEHICLE DETECTION ----------
    mask = np.ones(frame.shape[:2], dtype=np.uint8) * 255

    for tracker in multi_trackers[:]:
        success, bbox = tracker.update(frame)
        print(f"Tracking status: {success}")
        if not success:
            multi_trackers.remove(tracker)
            continue

        # Handle bbox safely
        if isinstance(bbox, (list, tuple)) and len(bbox) == 2:
            bbox = bbox[0]  # unwrap ((x,y,w,h), score)

        x, y, bw, bh = map(int, bbox)

        # Ignore invalid boxes
        if bw <= 0 or bh <= 0:
            continue
        cv2.rectangle(
            mask,
            (x, y),
            (x + bw, y + bh),
            0,
            thickness=-1
        )
        is_violation = (light_state == "RED" and line_intersects_box(p1, p2, (x, y, bw, bh)))

        if is_violation:
            cv2.putText(
                frame,
                "VIOLATION",
                (x, y - 25),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (0, 0, 255),
                2
            )

        cv2.rectangle(frame, (x, y), (x + bw, y + bh), (0, 255, 0), 2)

    shared_frame = cv2.bitwise_and(frame, frame, mask=mask).copy()

    done_event.clear()
    det_event.set()
    done_event.wait()

    # Add new trackers for detected vehicles
    for box in detections:
        x1_v, y1_v, x2_v, y2_v = box
        bw, bh = x2_v - x1_v, y2_v - y1_v
        if bw <= 8 or bh <= 9:
            continue
        tracker = cv2.legacy.TrackerMOSSE.create()
        tracker.init(frame, (x1_v, y1_v, bw, bh))
        multi_trackers.append(tracker)
        cv2.rectangle(frame, (x1_v, y1_v), (x2_v, y2_v), (255, 0, 0), 2)

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