from time import sleep
from picamera import PiCamera

from src.utils import Utils
from src.utils.Loger import *


@throws()
def take_photo():
    camera = PiCamera()
    camera.resolution = (1640, 922)#(1024, 768)
    camera.start_preview()
    sleep(2)
    pic_name_tmp = Utils.generate_img_title()
    pic_name_main = pic_name_tmp.format('orig')
    try:
        # camera.capture('../img/{}.jpg'.format(pic_name))
        camera.capture(pic_name_main)
    except Exception as exception:
        print(exception, "\n")
        raise exception
        return exception
    finally:
        camera.close()
    print("pic_name_main", pic_name_main, "\npic_name_tmp", pic_name_tmp)
    return (pic_name_main, pic_name_tmp)
