#import patterns as patterns
from django.conf.urls import url
from django.urls import path
# from django.conf.urls.defaults import *
from . import views


# ETO OCHEN VAJNO app_name must be specified like below!!!
app_name = 'schetchik'
urlpatterns = [
    path('', views.display_image, name='display_image'),
    path('test_api', views.test_api),
    path('test_camera', views.test_camera),
    path('current_value', views.current_value),
    url(r'^get_image?$', view=views.get_image, name='get_image')
]
