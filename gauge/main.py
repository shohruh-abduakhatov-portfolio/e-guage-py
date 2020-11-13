import json
import os
import time

import requests
from flask import Flask, request
from flask import send_file
from flask_restful import abort
from requests.exceptions import HTTPError

from config import server
from src.camera import GaugeDetector as detector
from src.utils import Utils

from src.camera import PhotoCapturer as camera
# from flask_upload import UploadSet, configure_uploads, IMAGES

app = Flask(__name__)


@app.route('/test_camera', methods=['GET'])
def test_camera():
    saved_imgs_names = camera.take_photo()[0]
    return send_file(saved_imgs_names, mimetype='image/jpg')
    pass


@app.route('/pressure_gauge_image', methods=['GET'])
def get_pressure_gauge():
    """This returns the image of the value by ID"""
    param_id = request.args.get('id')
    if param_id is None:
        return '[ERROR] >>> {param_id} is missing!'
    # if not pathlib.Path(param_id).exists():
    if not os.path.exists(param_id):
        return abort(404, message='File Not Found')

    return send_file('src/img/wiut-2.jpg', mimetype='image/jpg')


@app.route('/current_value', methods=['GET'])
def get_current_value():
    """This returns exactly current val which gauge showing"""
    results = get_current_gauge_value()
    return results


@app.route('/send_current_value', methods=['GET']) # this is for cron script call only
def send_current_value():
    """This returns exactly current val which gauge showing"""
    data_json = get_current_gauge_value()
    url = '%s:%s/%s?%s' % (server['ip'], server['port'], server['path'], server['param1'])
    headers = server['Content-type']
    ok = False
    while not ok:
        try:
            response = requests.post(url, data=data_json, headers=headers)
            ok = True
        except HTTPError:
            time.sleep(1)
    print(response)


# CONTROLLER HELPER
def get_current_gauge_value():
    # saved_imgs_names = camera.take_photo() # todo uncomment line
    pic_name_tmp = Utils.generate_img_title()
    pic_name_main = 'src/img/wiut-2.jpg'
    saved_img_names = (pic_name_main, pic_name_tmp)
    dict_results = detector.main(saved_img_names)
    results = json.dumps(dict_results)
    return results


if __name__ == '__main__':
    app.run(host='0.0.0.0	', port=5000, debug=True)
