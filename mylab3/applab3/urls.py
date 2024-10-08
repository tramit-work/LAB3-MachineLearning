from django.urls import path
from . import views

urlpatterns = [
    path('', views.home, name='home'),
    path('analyze_sentiment/', views.analyze_sentiment, name='analyze_sentiment'),
]
