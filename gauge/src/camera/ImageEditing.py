import imutils
import cv2
import numpy as np
from scipy.spatial import distance
from src.utils.Loger import *


@throws(ZeroDivisionError)
def calc_angle(A, B, O):
    try:
        slope_a = (O[1] - A[1]) / (O[0] - A[0])
        slope_b = (O[1] - B[1]) / (O[0] - B[0])
        tanA = abs((slope_b - slope_a) / (1 + slope_b * slope_a))
    except ZeroDivisionError:
        return 0
    return np.rad2deg(np.arctan(tanA))

@throws(ZeroDivisionError)
def dist_2_pts(x1, y1, x2, y2) -> int:
    a = (x1, y1)
    b = (x2, y2)
    return distance.euclidean(a, b)


# if __name__ == '__main__':
#     print(dist_2_pts(10, 0, 0, 0))
#     quit()

saving_path = 'output'
gauge_name = '../img/wiut-2.jpg'

img = cv2.imread(gauge_name)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
height, width = img.shape[:2]
circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 0.5, 90)
circles = np.uint16(np.around(circles))

# Copying.
cropped_imgs = []
j = 1
for i in circles[0, :]:
    # draw the outer circle
    # cv2.circle(img, (i[0], i[1]), i[2], (255, 255, 0), 2)
    # draw the center of the circle
    cv2.circle(img, (i[0], i[1]), 2, (0, 255, 255), 3)
    print('cv2.circle(img, (i[0], i[1]), 2, (0, 255, 255), 3')
    print(i)

    # UNCOMMENT LATER,. I NEED IT
    # mask = np.zeros((height,width), np.uint8)
    # cv2.circle(mask,(i[0],i[1])
    #            ,i[2],(255,255,255),thickness=-1)
    #
    # # Copy that image using that mask
    # masked_data = cv2.bitwise_and(img, img, mask=mask)
    #
    # # Apply Threshold
    # _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
    #
    # # Find Contour
    # contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    # x,y,w,h = cv2.boundingRect(contours[0])
    #
    # # Crop masked_data
    # crop = masked_data[y:y+h,x:x+w]
    # cropped_imgs.append(crop.copy())

# CROPPING
mask = np.zeros((height, width), np.uint8)
cv2.circle(mask, (circles[0][0][0], circles[0][0][1])
           , circles[0][0][2] - 15, (255, 255, 255), thickness=-1)
# Copy that image using that mask
masked_data = cv2.bitwise_and(img, img, mask=mask)
# Apply Threshold
_, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)
# Find Contour
contours = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
print('contours = ', contours)
x, y, w, h = cv2.boundingRect(contours[0])
# Crop masked_data
crop = masked_data[y:y + h, x:x + w]
print(crop)

num_rows, num_cols = crop.shape[:2]

# rotation_matrix = cv2.getRotationMatrix2D((num_cols/2, num_rows/2), 2, 1)
# img_rotation = cv2.warpAffine(crop, rotation_matrix, (num_cols, num_rows))
# cv2.imshow('Rotation', img_rotation)
# cv2.waitKey()
# quit(0)

cropped_imgs.append(crop.copy())

# LEFT BOTTOM CORNER CROP
crop_height, crop_width = crop.shape[:2]
crop_height = int(crop_height / 2)
crop_width = int(crop_width / 2)
bottom_left = crop[crop_height:, :crop_width].copy()

bottom_left_copy = bottom_left.copy()
# # strengthen black-white cols
# img_gray = cv2.cvtColor(bottom_left, cv2.COLOR_BGR2GRAY)
# img_gray = cv2.medianBlur(img_gray, 5)
# bottom_left_edges = cv2.Laplacian(img_gray, cv2.CV_8U, ksize=5)

@throws()
def img_show(title, img):
    cv2.imshow(str(title), img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# Edge
blurred = cv2.GaussianBlur(bottom_left, (5, 5), 0)
edged = cv2.Canny(blurred, 55, 255, 255)
lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 30, minLineLength=7, maxLineGap=150)
for line in lines:
    x1, y1, x2, y2 = line[0]
    cv2.line(bottom_left, (x1, y1), (x2, y2), (0, 255, 255), 2)
# img_show('bottom_left1', bottom_left)
blurred = cv2.GaussianBlur(bottom_left, (5, 5), 0)
edged = cv2.Canny(blurred, 55, 255, 255)
lines = cv2.HoughLines(edged, 1, np.pi / 180, 40)
print(lines)
len = int('1' + str(0) * (len(str(bottom_left.shape[0])) - 1))
for rho, theta in lines[0]:
    a = np.cos(theta)
    b = np.sin(theta)
    x0 = a * rho
    y0 = b * rho
    x1 = int(x0 + len * (-b))
    y1 = int(y0 + len * a)
    x2 = bottom_left.shape[0]
    y2 = 0
    # x2 = int(x0 - len * (-b))
    # y2 = int(y0 - len * a)
    print('rho = ', rho)
    print('theta = ', theta)
    print('a, b = ', a, ' ', b)
    print('x0, y0 = ', x0, ' ', y0)
    print('x1, y1 = ', x1, ' ', y1)
    print('x2, y2 = ', x2, ' ', y2)

    cv2.line(bottom_left_copy, (x1, y1), (x2, y2), (0, 0, 255), 2)
    cv2.line(bottom_left_copy, (0, 0), (x2, y2), (0, 0, 255), 2)
angle_a = 0, 0
angle_b = x1, y1
angle_o = x2, y2
print('>>>>>>>\n', angle_a)
print(angle_b)
print(angle_o)
alpha = calc_angle(angle_a, angle_b, angle_o)
print(alpha)
full_angle = alpha * 2 + 180
print('full_angle = ', full_angle)

# RESULT calc_angle()
min_angle, max_angle = alpha, 360 - alpha
min_val, max_val = 0, 10
x_center, y_center = tuple((np.array(crop.shape[:2]) / 2).astype('int'))
radius = circles[0, 0][2]
units = '(P)'
title = 'Pressure Gauge 1'
print('x_center = ', x_center, 'y_center = ', y_center)
print('min_angle = ', min_angle, 'max_angle = ', max_angle)
print('min_angle = ', min_angle, 'max_angle = ', max_angle)
print('radius = ', radius)
print('>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>.')
print('>>>>>>>>>>>>>>>>.')

# Current Value
circle_image = crop.copy()
blurred = cv2.GaussianBlur(circle_image, (5, 5), 0)
edged = cv2.Canny(blurred, 55, 255, 255)
lines = cv2.HoughLinesP(edged, 1, np.pi / 180, 100, minLineLength=200, maxLineGap=100)
for line in lines:
    x1, y1, x2, y2 = line[0]
    x2 = (x_center - x1) + x_center
    y2 = (y_center - y1) + y_center
    print(x1, ' -- ', y1)
    print(x2, ' -- ', y2)

    cv2.line(circle_image, (x1, y1), (x2, y2), (0, 255, 0), 2)
x_current_val, y_current_val = x1, y1




x1, y1 = angle_b
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

# print x_angle
# print y_angle
# print res
# print np.rad2deg(res)

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
final_angle -= 13.3  # todo this is config file
print("final_angle = ", final_angle.round(2))
print("final_angle2 = ", 84.74734279032292)

# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||
# |||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||
# |||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/\~~~~~~~~~~/|
# |~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/  \~~~~~~~~/ |
# |./\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /\ \~~/\~~/ /|
# |/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@@\ \/  \/ /@|
# | /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@@@\  /\  /@@|
# |/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/\@@\/  \/@@/|
# |\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/~~\@@\/\/@@/-|
# |@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~~~\@@\/@@/~~|
# |@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~~~/@@/\@@\~~|
# |/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\~~/@@/\/\@@\-|
# |\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\/@@/\  /\@@\|
# | \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@@@/  \/  \@@|
# |\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@@/ /\  /\ \@|
# |.\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \/ /~~\/~~\ \|
# |~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\  /~~~~~~~~\ |
# |~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\/~~~~~~~~~~\|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||
# |||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||._.||
# |||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||._.||===||
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|
# |~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~|


xp = [min_val, max_val]
fp = [0, full_angle]

curr_val = np.interp(final_angle, fp, xp)
print("Current Value is = ", curr_val)

cv2.imshow("img", img)
cv2.imshow("blurred", blurred)
cv2.imshow("edged", edged)
cv2.imshow("circle_image", circle_image)
cv2.imshow("bottom_left_copy", bottom_left_copy)

cv2.waitKey(0)
cv2.destroyAllWindows()

# full_angle = 269.6034909976016
