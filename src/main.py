import cv2
import sys
from copy import deepcopy
import os

IMG_PATH = sys.argv[1]

original = cv2.imread(IMG_PATH)

# resize the height to 512px while keeping the aspect ratio
height = 512
scale_rate = height / original.shape[0]
width = int(original.shape[1] * scale_rate)
dim = (width, height)

img = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 1.0, 100, 30, 60)

# find the biggest circle
max_radius = 0
target_index = 0

try:
    for index, circle in enumerate(circles[0]):
        if max_radius < circle[2]:
            max_radius = circle[2]
            target_index = index
except IndexError:
    print("failed to detect circles")
    exit(1)

target_circle = circles[0][target_index]

# draw circle on the image
img_res = deepcopy(img)

# 円周を描画する
cv2.circle(img_res, (int(target_circle[0]), int(target_circle[1])), int(target_circle[2]), (0, 165, 255), 5)
# 中心点を描画する
cv2.circle(img_res, (int(target_circle[0]), int(target_circle[1])), 2, (0, 0, 255), 3)

# save the result
filename = os.path.basename(IMG_PATH)
cv2.imwrite('dist/' + filename, img_res)