from django.db import models


class TrafficViolation(models.Model):
    timestamp = models.DateTimeField(auto_now_add=True)
    image = models.ImageField(upload_to='violations/')
    light_state = models.CharField(max_length=10)
    camera_id = models.CharField(max_length=50)
    x = models.IntegerField()
    y = models.IntegerField()
    w = models.IntegerField()
    h = models.IntegerField()

    def __str__(self):
        return f"Violation @ {self.timestamp}"
