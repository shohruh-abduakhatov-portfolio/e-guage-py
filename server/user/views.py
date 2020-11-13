import json

import requests
from django.contrib import messages
from django.http import JsonResponse, HttpResponseRedirect, HttpResponse
from django.shortcuts import render

from railwayadmin.settings import MIDDLEWARE_URL, API_HEADERS
from user.forms import CreateUserForm
# Create your views here.
from utils.helpers import check_session


@check_session()
def index(request):
    return render(request, 'user/index.html', {})


@check_session()
def create(request):
    header = API_HEADERS
    header['Authorization'] = "Bearer {}".format(request.session['token'])
    if request.method == 'POST':
        id = request.POST['identifier']
        form = CreateUserForm(request.POST)
        if form.is_valid():
            if id:
                response = requests.post(MIDDLEWARE_URL + "/users/update", data=json.dumps(form.as_dict(True)),
                                         headers=header)
                return HttpResponseRedirect('/user')
            else:
                uid = requests.post(MIDDLEWARE_URL + "/users/register", data=json.dumps(form.as_dict()),
                                    headers=API_HEADERS)
                if uid is None:
                    messages.error(request, 'Could not create new User')
    else:
        form = CreateUserForm()
    id = request.GET.get('id')
    if id:
        user = requests.post(MIDDLEWARE_URL + "/users/get/" + id, headers=header)
        form = CreateUserForm(json.loads(user.content.decode()))
    return render(request, 'user/create.html', {'form': form})


@check_session()
def view(request):
    header = API_HEADERS
    header['Authorization'] = "Bearer {}".format(request.session['token'])
    id = request.GET.get('id')
    user_data = requests.post(MIDDLEWARE_URL + "/users/get/" + id, headers=header)
    return render(request, 'user/view.html', {'userData': json.loads(user_data.content.decode())})


@check_session()
def table_list(request):
    header = API_HEADERS
    header['Authorization'] = "Bearer {}".format(request.session['token'])
    data = {
        "filters": json.loads(request.body)
    }
    dataTableRow = requests.post(MIDDLEWARE_URL + "/users/list", data=json.dumps(data), headers=header)
    return JsonResponse(json.loads(dataTableRow.content.decode()))


@check_session()
def activate(request):
    if request.method == 'POST':
        data = json.loads(request.body.decode())
        if data['userId']:
            r_data = requests.post(MIDDLEWARE_URL + "/users/activate", data=json.dumps(data), headers=API_HEADERS)
            if r_data.status_code == 200:
                return HttpResponse()
            else:

                return HttpResponse(status=r_data.status_code)
        else:
            return HttpResponse(status=404)
