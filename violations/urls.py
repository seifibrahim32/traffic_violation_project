from django.urls import path
from views import ViolationCreateView

urlpatterns = [
    path('violations/', ViolationCreateView.as_view()),
]
