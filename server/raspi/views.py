from django.shortcuts import render

import json

import requests
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect, HttpResponse
from django.shortcuts import render

from gauge.forms import GaugeForm

# Create your views here.

from wiut.settings import IP_RASPI, API_HEADERS


def index(request):
    return render(request, 'raspi/index.html', {})


def get_current_value(request):
    gauge_data = requests.get(IP_RASPI + "/current_value", headers=API_HEADERS)
    # return render(request, 'gauge/view.html', {'gaugeData': json.loads(gauge_data.content.decode())})
    return gauge_data

# def create(request):
#     header = API_HEADERS
#     header['Authorization'] = "Bearer {}".format(request.session['token'])
#     if request.method == 'POST':
#         id = request.POST['identifier']
#         # Создаем экземпляр формы и заполняем данными из запроса (связывание, binding):
#         form = GaugeForm(request.POST)
#         # Проверка валидности данных формы:
#         if form.is_valid():
#             if id:
#                 response = requests.post(IP_RASPI + "/gauges/update", data=json.dumps(form.as_dict(True)  ),
#                                          headers=header)
#                 return HttpResponseRedirect('/gauge')
#             else:
#                 uid = requests.post(IP_RASPI + "/gauges/register", data=json.dumps(form.as_dict()),
#                                     headers=API_HEADERS)
#                 if uid is None:
#                     messages.error(request, 'Could not create new User')
#     # Если это GET (или какой-либо еще)
#     else:
#         form = GaugeForm()
#     id = request.GET.get('id') # get id of the gauge
#     if id: # If id exists
#         gauge = requests.post(IP_RASPI + "/gauges/get/" + id, headers=header) # вызываются данные
#         form = GaugeForm(json.loads(gauge.content.decode())) # Форма Запоняется данными
#     return render(request, 'gauge/create.html', {'form': form})
#
#
# def view(request):
#     header = API_HEADERS
#     header['Authorization'] = "Bearer {}".format(request.session['token']) # сохраняю данные о юзере (sission, cookies)
#     id = request.GET.get('id') # возьми Image id from the request
#     gauge_data = requests.post(IP_RASPI + "/gauges/get/" + id, headers=header)
#     return render(request, 'gauge/view.html', {'gaugeData': json.loads(gauge_data.content.decode())})
#
#
# def table_list(request): # возвращает list of gauges in json
#     header = API_HEADERS
#     header['Authorization'] = "Bearer {}".format(request.session['token'])
#     data = {
#         "filters": json.loads(request.body)
#     }
#     dataTableRow = requests.post(IP_RASPI + "/gauges/list", data=json.dumps(data), headers=header)
#     return JsonResponse(json.loads(dataTableRow.content.decode()))
#
