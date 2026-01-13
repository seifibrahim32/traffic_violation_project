import threading

import vehicle_detector

det_event = threading.Event()
done_event = threading.Event()
stop_thread = False

shared_frame = None
vehicle_detector = vehicle_detector.VehicleDetector()


# ================= VEHICLE DETECTION THREAD =================
def detection_worker():
    global detections
    while not stop_thread:
        det_event.wait()
        det_event.clear()
        if stop_thread:
            break
        detections = vehicle_detector.detect(shared_frame)
        done_event.set()
