from src.utils import Utils
from datetime import datetime
from time import strftime, strptime
from src.camera import GaugeDetector as detector
import json


def photo_title_test():
    today = datetime.now().strftime(Utils.STRFTIME)
    return '../img/{}.jpg'.format(today)
    # return


if __name__ == '__main__':
    # pic_name_tmp = Utils.generate_img_title()
    # pic_name_main = '../img/wiut-2.jpg'
    # saved_img_names = (pic_name_main, pic_name_tmp)
    # gauge_result = detector.main(saved_img_names)
    # results = json.dumps(gauge_result)
    print(Utils.generate_img_title())
