import cv2
import numpy as np
from matplotlib import pyplot as plt

plt.figure(figsize=(12, 8))

fire_color_lower = np.array([5, 150, 150])
fire_color_upper = np.array([35, 255, 255])

image = cv2.imread('firepic.png')
img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
plt.subplot(221)
plt.title('original image')
plt.imshow(img_rgb)

hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
hsv_rgb = cv2.cvtColor(hsv, cv2.COLOR_BGR2RGB)
plt.subplot(222)
plt.title('hsv transformation')
plt.imshow(hsv_rgb)

mask = cv2.inRange(hsv, fire_color_lower, fire_color_upper)

kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

mask_rgb = cv2.cvtColor(mask, cv2.COLOR_BGR2RGB)
plt.subplot(223)
plt.title('masks applied')
plt.imshow(mask_rgb)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
max_peremeters = [[]]
for cnt in contours:
    x, y, w, h = cv2.boundingRect(cnt)
    rect_param = ((x + w), (y + h), ((x + w) + (y + h)) * 2)
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 0, 255), 2)

img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

plt.subplot(224)
plt.title('result')
plt.imshow(img_rgb)

plt.show()
