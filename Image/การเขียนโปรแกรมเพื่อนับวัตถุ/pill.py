#นางสาวนูรูลซัซวานี เด็นอะสัน

import cv2
import numpy as np
 
# Load the input image
image = cv2.imread('segment-1.png')
image = cv2.resize(image, (300, 300)) # Resize for faster processing and consistency
 
# Convert the image to Grayscale (Mandatory step for Otsu's Thresholding)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
 
# Apply Otsu's Thresholding
# เพื่อให้พื้นหลังสว่างกลายเป็นสีดำ (0) และวัตถุกลายเป็นสีขาว (255)
threshold_value, thresh_image = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
 
print(f"Otsu's Calculated Threshold Value: {threshold_value}")
 
#ทำ Connected Component Labeling เพื่อนับจำนวนวัตถุ
num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(thresh_image, 8, cv2.CV_32S)
 
pill_count = 0
 
#วนลูปเพื่อคัดกรองและวาดกรอบ (เริ่มที่ 1 เพื่อข้าม Background)
for i in range(1, num_labels):
    area = stats[i, cv2.CC_STAT_AREA]
    # กรอง Object ที่มีขนาดเล็กเกินไป (Noise)
    # เนื่องจากเราย่อภาพเหลือ 300x300 อาจจะต้องปรับเกณฑ์พื้นที่ (area) ให้เล็กลงตามสัดส่วน
    if area > 30: 
        pill_count += 1
        # ดึงพิกัดเพื่อวาดกรอบ Bounding Box
        x = stats[i, cv2.CC_STAT_LEFT]
        y = stats[i, cv2.CC_STAT_TOP]
        w = stats[i, cv2.CC_STAT_WIDTH]
        h = stats[i, cv2.CC_STAT_HEIGHT]
        # วาดกรอบสีเขียวและใส่หมายเลข
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
        cv2.putText(image, str(pill_count), (x, y - 5), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
 
print(f"Number of pills: {pill_count}")
 
cv2.imshow('Final Result', image)
cv2.imshow("Otsu's Thresholding (Mask)", thresh_image)
 
cv2.waitKey(0)
cv2.destroyAllWindows()
