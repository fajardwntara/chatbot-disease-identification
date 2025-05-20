
from django.contrib import admin
from django.urls import path, include
from project import urls as project_urls

urlpatterns = [
    path('', include(project_urls)),
    path('admin/', admin.site.urls),
]
