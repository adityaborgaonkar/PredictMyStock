from django.contrib import admin
from django.urls import path
from app.views import predict, home


urlpatterns = [
    path('', home),
    path('predict/', predict),
]
