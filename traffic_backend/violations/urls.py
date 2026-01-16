from django.urls import path

from . import views
from .views import ViolationCreateView

urlpatterns = [
    path('all_violations/', views.all_violations, name='violations'),
    path('violations/', ViolationCreateView.as_view()),
]
