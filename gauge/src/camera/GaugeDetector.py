import cv2
import numpy as np
from scipy.spatial import distance

from src.utils import Utils
from src.utils.Loger import *
import config as cfg
import json

cropped_imgs = None
img = None
radiuses_list = None
x_center = None
y_center = None
radius = None
min_angle = None
max_angle = None
min_angle_coords = None
final_angle = None
x_current_val = None
y_current_val = None


@throws()
def img_show(title, img):
    cv2.imshow(str(title), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# @throws()
def detect_circles():
    global cropped_imgs, img, radiuses_list
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 90)
    circles = np.uint16(np.around(circles))
    height, width = img.shape[:2]
    # Copying.
    cropped_imgs = []
    radiuses_list = []

    i = 0
    for circle_item in circles[0, :]:
        if circle_item[2] < 150: continue
        cv2.circle(img, (circle_item[0], circle_item[1]), 2, (0, 255, 255), 3)
        mask = np.zeros((height, width), np.uint8)
        cv2.circle(mask, (circle_item[0], circle_item[1])
                   , circle_item[2] - 15, (255, 255, 255), thickness=-1)
        # Copy that image using that mask
        masked_data = cv2.bitwise_and(img, img, mask=mask)
        # Apply Threshold
        _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
        # Find Contour
        contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        x, y, w, h = cv2.boundingRect(contours[0])
        # Crop masked_data
        crop = masked_data[y:y + h, x:x + w]
        cropped_imgs.append(crop.copy())
        radiuses_list.append(circle_item[2])
        i += 1


@throws()
def fix_rotation():
    """ the image is not still, so it fixes it """
    global cropped_imgs
    angles = cfg.circles_rotation['rotation_angles']
    for i in range(len(cropped_imgs)):
        circle_item = cropped_imgs[i]
        num_rows, num_cols = circle_item.shape[:2]
        # crop_copy = crop.copy()
        rotation_matrix = cv2.getRotationMatrix2D((num_cols / 2, num_rows / 2), angles[i], 1)
        img_rotation = cv2.warpAffine(circle_item, rotation_matrix, (num_cols, num_rows))
        cropped_imgs[i] = img_rotation


@throws()
def crop_left_bottom(crop):
    """ LEFT BOTTOM CORNER CROP """
    global x_center, y_center, radius
    radius = int(crop.shape[1] / 2)
    x_center, y_center = tuple((np.array(crop.shape[:2]) / 2).astype('int'))
    bottom_left = crop[x_center:, :y_center].copy()
    return bottom_left


@throws()
def detect_min_max_angles(bottom_left_image):
    """ Edge Minimum angle identification """
    global min_angle, max_angle, min_angle_coords
    bottom_left = bottom_left_image.copy()
    blurred = cv2.GaussianBlur(bottom_left, (5, 5), 0)
    edged = cv2.Canny(blurred, 55, 255, 255)
    # 1st layer of line detection
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 30, minLineLength=7, maxLineGap=100)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        cv2.line(bottom_left, (x1, y1), (x2, y2), (0, 0, 255), 2)

    blurred = cv2.GaussianBlur(bottom_left, (5, 5), 0)
    edged = cv2.Canny(blurred, 55, 255, 255)
    # 2nd layer of line drawing
    lines = cv2.HoughLines(edged, 1, np.pi / 180, 40)
    length = int('1' + str(0) * (len(str(bottom_left.shape[0])) - 1))
    for rho, theta in lines[0]:
        a, b = np.cos(theta), np.sin(theta)
        x0, y0 = a * rho, b * rho
        x1, y1 = int(x0 + length * (-b)), int(y0 + length * a)
        x2, y2 = bottom_left.shape[0], 0
        print('x0, y0 = ', x0, ' ', y0, '\nx1, y1 = ', x1, ' ', y1, '\nx2, y2 = ', x2, ' ', y2)
        cv2.line(bottom_left, (x1, y1), (x2, y2), (0, 255, 255), 2)
        cv2.line(bottom_left, (0, 0), (x2, y2), (0, 255, 255), 2)

    angle_a = 0, 0
    angle_b = x1, y1
    angle_o = x2, y2
    min_angle_coords = angle_b
    min_angle = calc_angle(angle_a, angle_b, angle_o)
    max_angle = min_angle * 2 + 180


@throws(ZeroDivisionError)
def calc_angle(A, B, O):
    try:
        slope_a = (O[1] - A[1]) / (O[0] - A[0])
        slope_b = (O[1] - B[1]) / (O[0] - B[0])
        tanA = abs((slope_b - slope_a) / (1 + slope_b * slope_a))
    except ZeroDivisionError:
        raise ZeroDivisionError
        return 0
    return np.rad2deg(np.arctan(tanA))


@throws()
def dist_2_pts(x1, y1, x2, y2) -> int:
    return distance.euclidean((x1, y1), (x2, y2))


@throws()
def print_found_params(i):
    """ Just print what we have in terms of params needed"""
    print('>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>')
    print('x_center = ', x_center, 'y_center = ', y_center)
    print('min_angle = ', min_angle, 'max_angle = ', max_angle)
    print('min_val = ', cfg.circles_rotation['min_max_values'][i][0], 'max_val = ',
          cfg.circles_rotation['min_max_values'][i][1])
    print('radius = ', radius, ' .. radiuses_list:', radiuses_list)
    print('title = ', cfg.circles_rotation['titles'][i])
    print('units = ', cfg.circles_rotation['units'][i])
    print('<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<')


# todo create api for returning image with title


@throws()
def detect_current_value(croped_img):
    # Current Value
    global x_current_val, y_current_val
    crop = croped_img.copy()
    circle_image = crop.copy()
    blurred = cv2.GaussianBlur(circle_image, (5, 5), 0)
    edged = cv2.Canny(blurred, 55, 255, 255)
    lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=100)
    for line in lines:
        x1, y1, x2, y2 = line[0]
        x2 = (x_center - x1) + x_center
        y2 = (y_center - y1) + y_center

    cv2.line(circle_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    x_current_val, y_current_val = x1, y1


@throws()
def cal_final_angle(i):
    global final_angle, x_current_val, y_current_val
    x1, y1 = min_angle_coords
    dist_pt_0 = dist_2_pts(x_center, y_center, x1, y1)
    dist_pt_1 = dist_2_pts(x_center, y_center, x_current_val, y_current_val)

    if dist_pt_0 < dist_pt_1:
        x_angle = x1 - x_center
        y_angle = y_center - y1
    else:
        x_angle = x_current_val - x_center
        y_angle = y_center - y_current_val
    # take the arc tan of y/x to find the angle
    res = np.arctan(np.divide(float(y_angle), float(x_angle)))
    # np.rad2deg(res) #coverts to degrees
    # these were determined by trial and error
    res = np.rad2deg(res)
    if x_angle > 0 and y_angle > 0:  # in quadrant I
        final_angle = 270 - res
    if x_angle < 0 and y_angle > 0:  # in quadrant II
        final_angle = 90 - res
    if x_angle < 0 and y_angle < 0:  # in quadrant III
        final_angle = 90 - res
    if x_angle > 0 and y_angle < 0:  # in quadrant IV
        final_angle = 270 - res
    # full_angle = 269.6034909976016
    final_angle = final_angle + cfg.circles_rotation['final_make_up_angle'][i]
    print("final_angle = ", final_angle)


# |------------------------------------------------------------------------------------------------------------------------|
# |@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ |
# |@---------------------------------------------------------------------------------------------------------------------|@|
# |@|....................................................................................................................|@|
# |@|....................................................................................................................|@|
# |@|||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||@|
# |@|||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||@|
# |@|||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||@|
# |@|....................................................................................................................|@|
# |@|....................................................................................................................|@|
# |@|......*/\........../\........../\........../\........../\........../\........../\........../\........../\........../|@|
# |@|....../  \......../  \......../  \......../  \......../  \......../  \......../  \......../  \......../  \......../ |@|
# |@|*/\../ /\ \../\../ /\ \../\../ /\ \../\../ /\ \../\../ /\ \../\../ /\ \../\../ /\ \../\../ /\ \../\../ /\ \../\../ /|@|
# |@|/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@|@|
# |@| /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@|@|
# |@|/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/|@|
# |@|\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/..\@@\/\/@@/-|@|
# |@|@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/....\@@\/@@/..|@|
# |@|@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..../@@/\@@\..|@|
# |@|/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\../@@/\/\@@\-|@|
# |@|\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\|@|
# |@| \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@|@|
# |@|\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@|@|
# |@|*\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \/ /..\/..\ \|@|
# |@|......\  /........\  /........\  /........\  /........\  /........\  /........\  /........\  /........\  /........\ |@|
# |@|......*\/..........\/..........\/..........\/..........\/..........\/..........\/..........\/..........\/..........\|@|
# |@|....................................................................................................................|@|
# |@|....................................................................................................................|@|
# |@|||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||@|
# |@|||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||..*||@|
# |@|||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||..*||===||@|
# |@|....................................................................................................................|@|
# |@|....................................................................................................................|@|
# |@---------------------------------------------------------------------------------------------------------------------|@|
# |@ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ @ |
# |-------------------------------------------------------------------------------------------------------------------------

@throws()
def get_current_gauge_val(i):
    global max_angle, final_angle, x_current_val, y_current_val
    gauge_cfg = cfg.circles_rotation['min_max_values'][i]
    xp = gauge_cfg
    fp = [0, max_angle]

    curr_val = np.interp(final_angle, fp, xp)
    curr_val = curr_val.round(2)
    print("Current Value is = ", curr_val)
    print("Current Value2 is = ", 3.006944648031637)
    return curr_val


@throws()
def save_result(save_path, img):
    cv2.imwrite(save_path, img)


def main(img_names):
    global img, cropped_imgs
    gauge_name = img_names[0]
    img = cv2.imread(gauge_name)
    detect_circles()
    fix_rotation()
    dict_response = {}
    for image, i in zip(cropped_imgs, np.arange(len(cropped_imgs))):
        print(i)
        bottom_left = crop_left_bottom(image)
        detect_min_max_angles(bottom_left)
        print_found_params(i)
        detect_current_value(image)
        cal_final_angle(i)
        current_value = get_current_gauge_val(i)
        saving_path = img_names[1].format(str(i + 1))
        save_result(saving_path, image)
        key = 'img_' + str(i)
        dict_response[key] = {
            'id': saving_path,
            'value': current_value
        }
    return dict_response


if __name__ == '__main__':
    pic_name_tmp = Utils.generate_img_title()
    pic_name_main = '../img/wiut-2.jpg'
    saved_img_names = (pic_name_main, pic_name_tmp)
    gauge_result = main(saved_img_names)
    results = json.dumps(gauge_result)
