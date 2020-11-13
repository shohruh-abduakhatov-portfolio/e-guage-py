from django.http import HttpResponse
from django.shortcuts import render
#from firebase import firebase
from base64 import b64encode
import requests
from PIL import Image
# Create your views here.
import random
INK = "red", "blue", "green", "yellow"


def mm(request):
    image_data = ...  # byte values of the image
    image = Image.frombytes('RGBA', (128, 128), image_data)
    #image = Image.new("RGB", (800, 600), random.choice(INK))

    # serialize to HTTP response
    response = HttpResponse(mimetype="image/png")
    image.save(response, "PNG")
    return response


def display_image(request):
    # fbase = firebase.FirebaseApplication('https://schetchik-kamera.firebaseio.com')
    # result = fbase.get('/photo', None)
    url = 'http://192.168.43.179:5000/pressure_gauge'
    response = requests.get(url)
    if response.status == '500':
        result = "Server error"
    else:
        image = b64encode(response.content)
        result = image
        image = Image.frombytes('RGBA', (128, 128), result)
        response = HttpResponse(mimetype=image)
        image.save(response, "PNG")
        result = response

    return render(request, 'mainContent/First.html', {'form': result})
