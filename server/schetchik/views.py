import shutil

import requests
from PIL import Image
from django.http import HttpResponse
from django.shortcuts import render


# Create your views here.


INK = "red", "blue", "green", "yellow"


def test_api(request):
    print('>>> OK -> ', request)
    return HttpResponse('OK')


def test_camera(param):
    url = 'http://192.168.0.111:5000/test_camera'
    response = requests.get(url, stream=True)
    save_path = 'schetchik/static/img.png'
    with open(save_path, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return HttpResponse(save_path)


def current_value(param):
    """{"img_0": {"value": 3.0, "id": "src/img/saved/20180918//img_02:09:56-1.jpg"}}"""
    url = 'http://192.168.0.111:5000/current_value'
    response = requests.get(url)
    print("response = ", response)
    return HttpResponse(response)


def get_image(param):
    """get image of the gauge by id of the image"""
    print('>>>>>>>', param.GET.get('id'))
    ids = param.GET.get('id')
    url = 'http://192.168.0.111:5000/pressure_gauge_image?id={}'.format(ids)
    response = requests.get(url, stream=True)
    save_to = 'schetchik/static/{}'.format(ids.split('/')[-1])
    print("saving point: ", save_to)
    with open(save_to, 'wb') as out_file:
        shutil.copyfileobj(response.raw, out_file)
    del response
    return HttpResponse(save_to)


def display_image(request):
    print(">>>>", request)
    return render(request, 'mainContent/First.html')  # , {'form': response})
