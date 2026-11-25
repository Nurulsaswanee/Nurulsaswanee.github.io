#นางสาวนูรูลซัซวานี เด็นอะสัน 6720310107

import cv2
import numpy as np

# 1. อ่านไฟล์ภาพ
tem = cv2.imread('template.png')
test1 = cv2.imread('test1.png')
test2 = cv2.imread('test2.png')

# 2. แปลงเป็น Grayscale
gray_tem = cv2.cvtColor(tem, cv2.COLOR_BGR2GRAY)
gray_test1 = cv2.cvtColor(test1, cv2.COLOR_BGR2GRAY)
gray_test2 = cv2.cvtColor(test2, cv2.COLOR_BGR2GRAY)

# 3. แปลงภาพเป็นชุดรหัสแบบ 8 จุดภาพ
def get_chain_code(img):
    
    # Threshold
    _, binary = cv2.threshold(
        img, 127, 255, cv2.THRESH_BINARY_INV
    )
    
    # หาเส้นขอบของวัตถุ
    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE
    )
    
    if len(contours) == 0:
        return []

    # เลือกเส้นขอบที่ใหญ่ที่สุด
    contour = max(contours, key=cv2.contourArea)

    # จุดของเส้นขอบ
    points = contour[:, 0, :]

    # กำหนดรหัส 8 ทิศทาง
    direction = {
        (1, 0): 0,    # →
        (1, -1): 1,   # ↗
        (0, -1): 2,   # ↑
        (-1, -1): 3,  # ↖
        (-1, 0): 4,   # ←
        (-1, 1): 5,   # ↙
        (0, 1): 6,    # ↓
        (1, 1): 7     # ↘
    }

    chain_code = []

    # เดินตามเส้นขอบ และกลับมาจุดแรก
    for i in range(len(points)):

        x1, y1 = points[i]
        x2, y2 = points[(i + 1) % len(points)]

        dx = np.sign(x2 - x1)
        dy = np.sign(y2 - y1)

        if (dx, dy) in direction:
            chain_code.append(direction[(dx, dy)])


    return chain_code

code_tem = get_chain_code(gray_tem)
code_test1 = get_chain_code(gray_test1)
code_test2 = get_chain_code(gray_test2)

# 4. จำนวนคู่ของรหัสที่เหมือนกัน Template กับ Test1
length1 = min(len(code_tem), len(code_test1))
M1 = sum(
    code_tem[i] == code_test1[i]
    for i in range(length1)
)

# 5. จำนวนคู่ของรหัสที่เหมือนกัน Template กับ Test2
length2 = min(len(code_tem), len(code_test2))
M2 = sum(
    code_tem[i] == code_test2[i]
    for i in range(length2)
)

# 6. คะแนนความเหมือน Template กับ Test1
R1 = (M1 / length1) * 100

# 7. คะแนนความเหมือน Template กับ Test2
R2 = (M2 / length2) * 100

# 8. สรุปผล
print("M1 =", M1)
print("R1 =", R1, "%")

print("M2 =", M2)
print("R2 =", R2, "%")

if R1 > R2:
    print("ภาพ Template เหมือนกับ Test1 มากกว่า")
elif R2 > R1:
    print("ภาพ Template เหมือนกับ Test2 มากกว่า")
else:
    print("ภาพ Template เหมือนกับ Test1 และ Test2 เท่ากัน")


