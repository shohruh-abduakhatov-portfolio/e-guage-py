from django.urls import path, include
from raspi import views

urlpatterns = [
    path('current_value', views.get_current_value, name='gauge_index'),
    path('', views.index, name='gauge_index'),

]
