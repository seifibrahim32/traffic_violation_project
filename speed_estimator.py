import time


class SpeedEstimator:
    def __init__(self, pixel_distance, real_distance):
        self.pixel_distance = pixel_distance
        self.real_distance = real_distance
        self.timestamps = {}

    def estimate(self, object_id):
        now = time.time()

        if object_id not in self.timestamps:
            self.timestamps[object_id] = now
            return None

        elapsed = now - self.timestamps[object_id]
        speed = (self.real_distance / elapsed) * 3.6  # km/h
        return round(speed, 2)
