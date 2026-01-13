from ultralytics import YOLO


class TrafficLightsDetector:
    def __init__(self):
        self.model = YOLO("models/yolov8n.pt")

    def detect(self, frame):
        results = self.model(frame, conf=0.9, device="cpu", verbose=True)[0]
        traffic_lights = []

        for box in results.boxes:
            cls = int(box.cls[0])
            if cls == 9:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                traffic_lights.append((x1, y1, x2, y2))

        return traffic_lights
