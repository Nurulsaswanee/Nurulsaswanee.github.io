#นางสาวนูรูลซัซวานี เด็นอะสัน

import cv2

# Load the input image
img = cv2.imread('objects.png')
# Convert the image to Grayscale
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# Apply Otsu's Thresholding
_, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

# ค้นหาคอนทัวร์และประมวลผล
contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
sizes = []

for c in contours:
    if cv2.contourArea(c) > 50:
        # หาจุดศูนย์กลาง (Centroid)
        M = cv2.moments(c)
        if M["m00"] != 0:
            cx, cy = int(M["m10"] / M["m00"]), int(M["m01"] / M["m00"])
            
            # หาขนาดรัศมีเพื่อจัดกลุ่มขนาด
            _, radius = cv2.minEnclosingCircle(c)
            sizes.append(int(round(radius)))

            # วาดเส้นขอบและพิกัดลงภาพ
            cv2.drawContours(img, [c], -1, (0, 255, 0), 2)
            cv2.circle(img, (cx, cy), 4, (0, 0, 255), -1)
            cv2.putText(img, f"({cx},{cy})", (cx+5, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 0, 0), 1)

# สรุปผลขนาดและจำนวนวงกลม
size_counts = {}
for r in sizes:
    size_counts[r] = size_counts.get(r, 0) + 1

print(f"วงกลมทั้งหมดมี {len(size_counts)} ขนาด:")
for r, count in sorted(size_counts.items()):
    print(f"- ขนาดรัศมีประมาณ {r} px: จำนวน {count} วง")

# Display the results on screen
cv2.imshow('Threshold', thresh)
cv2.imshow('Result', img)
cv2.waitKey(0)
cv2.destroyAllWindows()
