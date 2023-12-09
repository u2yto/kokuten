import json
import os
import sys
from copy import deepcopy

import cv2
import numpy as np

IMG_PATH = sys.argv[1]

original = cv2.imread(IMG_PATH)

print(f"Starting processing {IMG_PATH}...")

# resize the height to 1024px while keeping the aspect ratio
height = 1024
scale_rate = height / original.shape[0]
width = int(original.shape[1] * scale_rate)
dim = (width, height)

img = cv2.resize(original, dim, interpolation=cv2.INTER_AREA)

# detect circle
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

gray_blurred = cv2.medianBlur(gray, 3)

circles = cv2.HoughCircles(gray_blurred, cv2.HOUGH_GRADIENT, 1.0, 100, 30, 60)

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

target_circle = circles[0][target_index]  # [center_x, center_y, radius]
target_circle = np.uint16(np.around(target_circle))  # convert into int

# crop
target_circle = np.uint16(np.around(target_circle))
crop_start_x = target_circle[0] - target_circle[2]
crop_end_x = target_circle[0] + target_circle[2]
crop_start_y = target_circle[1] - target_circle[2]
crop_end_y = target_circle[1] + target_circle[2]
cropped = img[crop_start_y:crop_end_y, crop_start_x:crop_end_x]

# make a mask
gap = int(height / 100)  # 1% of height
mask = np.full(cropped.shape[:2], 255, dtype=img.dtype)
mask = cv2.circle(
    mask,
    (target_circle[2], target_circle[2]),
    target_circle[2] - gap,
    color=0,
    thickness=-1,
)  # draw inside the circle
mask = cv2.circle(
    mask, (target_circle[2], target_circle[2]), gap, color=255, thickness=-1
)  # draw center point

# mask
masked = deepcopy(cropped)

masked[mask == 255] = 255

# threshold
masked_blurred = cv2.bilateralFilter(masked, 9, 75, 75)
masked_hsv = cv2.cvtColor(masked_blurred, cv2.COLOR_BGR2HSV)
masked_threshold = cv2.bitwise_not(cv2.inRange(masked_hsv, (0, 0, 0), (255, 30, 140)))

# detect sunspots
masked_height = img.shape[0]

contours, hierarchy = cv2.findContours(
    masked_threshold, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE
)

# filter sunspots
sunspots = list(
    filter(
        lambda x: masked_height / 150 < cv2.arcLength(x, True) < masked_height / 45,
        contours,
    )
)

# draw sunspots
img_res = deepcopy(masked)

img_res = cv2.drawContours(img_res, sunspots, -1, color=(0, 255, 0), thickness=1)

# save the result as JSON
filename = os.path.basename(IMG_PATH) + ".json"
result_sunspots = list(map(lambda x: cv2.moments(x), sunspots))  # calculate moments
result_sunspots = list(
    map(
        lambda m: {"x": m["m10"] / m["m00"], "y": m["m01"] / m["m00"]}, result_sunspots
    )  # calculate centroids
)
result = {
    "entire_height": img_res.shape[0],
    "entire_width": img_res.shape[1],
    "sunspots": result_sunspots,
}

try:
    with open("dist/" + filename, "w") as f:
        json.dump(result, f)
except FileNotFoundError:
    print("failed to save the result as JSON")
    exit(1)

# save the result as an image
filename = os.path.basename(IMG_PATH)

save_status = cv2.imwrite("dist/" + filename, img_res)
if not save_status:
    print("failed to save the result as an image")
    exit(1)

print(f"Succeeded processing {IMG_PATH}")
