import cv2
import numpy as np

MIN_CONTOUR_AREA = 20  # พื้นที่ contour ขั้นต่ำ เพื่อกันจุดรบกวน (noise) เล็กๆ

# อ่านภาพและเตรียมภาพ
def read_binary(filename):
    img = cv2.imread(filename, cv2.IMREAD_GRAYSCALE)
    if img is None:
        print(f"เตือน: ไม่พบไฟล์ '{filename}'")
        return None

    # ลอง Threshold แบบ Otsu ทั้งสองขั้ว (ตัวอักษรเข้ม/พื้นสว่าง และ ตัวอักษรสว่าง/พื้นเข้ม)
    _, binary_dark_on_light = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    _, binary_light_on_dark = cv2.threshold(
        img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    ratio_dark_on_light = cv2.countNonZero(binary_dark_on_light) / binary_dark_on_light.size
    ratio_light_on_dark = cv2.countNonZero(binary_light_on_dark) / binary_light_on_dark.size

    binary = binary_dark_on_light if ratio_dark_on_light <= ratio_light_on_dark else binary_light_on_dark
    return binary

# หา Contour ที่ใหญ่ที่สุดของตัวอักษร
def get_contour(filename):
    binary = read_binary(filename)
    if binary is None:
        return None

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    if len(contours) == 0:
        print(f"เตือน: ไม่พบ Contour ใดๆ ในไฟล์ '{filename}'")
        return None

    # เลือก contour ที่มีพื้นที่มากที่สุด (ตัด Noise เล็กๆ ออก)
    contour = max(contours, key=cv2.contourArea)

    if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
        print(f"เตือน: Contour ในไฟล์ '{filename}' มีขนาดเล็กเกินไป (อาจเป็น noise)")
        return None

    return contour

# ขั้นตอนที่ 1: สร้างฐานข้อมูลตัวอย่าง (Templates)
templates = {
    "A": [],
    "B": [],
    "C": []
}

for letter in templates:
    for i in range(1, 6):
        filename = f"{letter}{i}.png"
        contour = get_contour(filename)

        if contour is not None:
            templates[letter].append(contour)
            print(f"เรียนรู้สำเร็จ: {filename}")
        else:
            print(f"ข้ามไฟล์ (ไม่พบรูปหรือรูปไม่สมบูรณ์): {filename}")
print("--------------------------------------")

# ขั้นตอนที่ 2: ฟังก์ชันจำแนกตัวอักษร (ปรับใช้ค่าเฉลี่ย)
def recognize(filename):
    print(f"\nกำลังทดสอบไฟล์: {filename}")
    test_contour = get_contour(filename)
    if test_contour is None:
        print("ไม่สามารถอ่านภาพหรือหา Contour ได้")
        return

    scores = {}
    # วนลูปเทียบกับตัวอักษรแต่ละกลุ่ม (A, B, C)
    for letter, contour_list in templates.items():
        if len(contour_list) == 0:
            continue

        distances = []
        for template_contour in contour_list:
            try:
                # ใช้ Hu Moments ผ่าน cv2.matchShapes (สูตร I1)
                dist = cv2.matchShapes(
                    test_contour,
                    template_contour,
                    cv2.CONTOURS_MATCH_I1,
                    0.0
                )
                distances.append(dist)
            except cv2.error:
                # ป้องกันโปรแกรมล่มหาก contour ผิดปกติจนคำนวณ Hu Moments ไม่ได้
                print(f"เตือน: เปรียบเทียบรูปร่างกับตัวอย่างบางไฟล์ในกลุ่ม '{letter}' ไม่สำเร็จ (ข้ามไฟล์นั้น)")
                continue

        if len(distances) == 0:
            continue

        # คำนวณระยะห่างเฉลี่ยของตัวอักษรหมวดนั้นๆ (ยิ่งน้อยยิ่งเหมือน)
        distances.sort()
        best_distances = distances[:2] if len(distances) >= 2 else distances
        scores[letter] = np.mean(best_distances)

    if not scores:
        print("ไม่สามารถประเมินผลได้ (ไม่มีข้อมูลเทมเพลต)")
        return

    # เรียงลำดับจากคะแนนดีที่สุด (น้อยสุด) ไปมากสุด
    sorted_scores = sorted(scores.items(), key=lambda x: x[1])

    print("ผลคะแนนเฉลี่ย (Distance):")
    for letter, avg_dist in sorted_scores:
        print(f"  - ตัวอักษร '{letter}': {avg_dist:.4f}")

    # ตัวอักษรที่มีค่าเฉลี่ยระยะห่างต่ำที่สุดคือผู้ชนะ
    best_answer, best_score = sorted_scores[0]
    print(f"ภาพนี้คือตัวอักษร -> {best_answer} (ด้วยค่าระยะห่าง {best_score:.4f})")
    print("--------------------------------------")

# ขั้นตอนที่ 3: ทดสอบ
recognize("char_test1.png")
recognize("char_test2.png")
