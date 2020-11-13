from src.utils import Utils
from datetime import datetime
import os
from config import camera as cfg


IMAGE_TITLE = "{timestamp:%d%m%Y}/img-{timestamp:%d%m%Y %H:%M:%S}"
IMAGE_TITLE_NUMBER = "-{counter:03d}"
TEST_IMAGE_PATH = 'wiut-2.jpg'

DIR_DATE_TITLE = '%Y%m%d'
STRFTIME = '%H%M%S'


# raspi_id, camera_id, gauge_id


def generate_img_title():
    now = datetime.now()
    date = now.strftime(DIR_DATE_TITLE)
    time = now.strftime(STRFTIME)
    top_dir = 'src/img/saved/%s/' % date
    if not os.path.exists(top_dir):
        os.makedirs(top_dir)
    img_title = '-'.join(['img', date, time, cfg["raspi_id"], cfg["camera_id"], cfg["gauge_prefix"]])
    return top_dir, img_title
