from django.urls import path, include
from gauge import views

urlpatterns = [
    path('', views.index, name='gauge_index'),
]
