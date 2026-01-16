from django.http import HttpResponse
from django.shortcuts import render
from django.template import loader
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework import status

from .models import TrafficViolation
from .serializers import TrafficViolationSerializer


def all_violations(request):
    template = loader.get_template('all_violations.html')

    violations = TrafficViolation.objects.all().order_by('-timestamp')

    context = {
        'violations': violations
    }

    return HttpResponse(template.render(context, request))

class ViolationCreateView(APIView):

    def get(self, request):
        violations = TrafficViolation.objects.all().order_by('-timestamp')
        serializer = TrafficViolationSerializer(violations, many=True)
        return Response(serializer.data)

    def post(self, request):
        serializer = TrafficViolationSerializer(data=request.data)
        if serializer.is_valid():
            serializer.save()
            return Response(serializer.data, status=status.HTTP_201_CREATED)
        return Response(serializer.errors, status=status.HTTP_400_BAD_REQUEST)