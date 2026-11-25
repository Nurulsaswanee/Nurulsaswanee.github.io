# นางสาวนูรูลซัซวานี เด็นอะสัน

import cv2
import numpy as np

img = cv2.imread('car.jpg', cv2.IMREAD_GRAYSCALE)

# กำหนด Kernel อนุพันธ์อันดับ 1 ของ Sobel ด้วยตัวเอง
kernel_x = np.array([
    [-1, 0, 1],
    [-2, 0, 2],
    [-1, 0, 1]
], dtype=np.float32)

kernel_y = np.array([
    [-1, -2, -1],
     [0,  0,  0],
     [1,  2,  1]
], dtype=np.float32)

# ทำ Convolution (Filtering)
gx = cv2.filter2D(img, cv2.CV_32F, kernel_x)
gy = cv2.filter2D(img, cv2.CV_32F, kernel_y)

# คำนวณ Magnitude
gradient = np.sqrt(gx**2 + gy**2)
# แปลงค่าเป็น uint8 (0-255) สำหรับแสดงผลภาพ
gradient = cv2.convertScaleAbs(gradient)

# กำหนดค่าอ้างอิง
threshold_value = 90
_, edge_sharp = cv2.threshold(gradient, threshold_value, 255, cv2.THRESH_BINARY)

# Stack images horizontally to compare original and processed results
result = np.hstack((gradient, edge_sharp))

# Display the output
cv2.imshow('Original Image', img)
cv2.imshow('Manual Sobel Edge vs Thresholded Edge', result)
cv2.waitKey(0)
cv2.destroyAllWindows()


