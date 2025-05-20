from django.contrib import admin
from django.urls import path
from .views import *

urlpatterns = [
    path('', home, name="home"),
    path('create_process/', createProcess, name="createProcess")
]
