from rest_framework import serializers
from models import TrafficViolation


class TrafficViolationSerializer(serializers.ModelSerializer):
    class Meta:
        model = TrafficViolation
        fields = '__all__'
