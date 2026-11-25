import tkinter as tk
from tkinter import filedialog, messagebox #ลักษณะเป็นหน้าจอ มีปุ๋ม
from PIL import Image, ImageTk
import cv2
import numpy as np


class OMRChecker:
    def __init__(self, root):
        # Set up the main window //หน้าจอจะเเอาขนาดเท่าไหร่ ถ้าเล็กไปรูปอาจจะเห็นไม่หมด
        self.root = root
        self.root.title("OMR Checker")
        self.root.geometry("900x850")

        # Create a button for selecting an answer-sheet image //สร้างปุ๋ม
        self.load_button = tk.Button(
            root,
            text="Select Answer Sheet", #มีข้อความบนปุ๋ม
            command=self.load_image, #ถ้ากดปุ๋มนี้จะทำงานฟังก์ชันไหน
            font=("Arial", 12)
        )
        self.load_button.pack(pady=10) #เรียกมาใช้ให้ห่างจากบนล่างเท่าไหร่

        # Create labels for Student ID and score #สร้างข้อความ
        self.id_label = tk.Label(
            root,
            text="Student ID: -",
            font=("Arial", 14)
        )
        self.id_label.pack(pady=5) #student id scoreจะโชว์ตรงนี้

        self.score_label = tk.Label(
            root,
            text="Score: -",
            font=("Arial", 14)
        )
        self.score_label.pack(pady=5)

        # Create a canvas for displaying the result image #ฉากใบพื้นหลังเพื่อเราจะแสดงตำแหน่งนี้
        self.canvas = tk.Canvas(root, width=800, height=650, bg="gray")
        self.canvas.pack(pady=10)

        # Keep a reference to the image for Tkinter
        self.tk_image = None #มีค่าหลอกๆไว้ก่อน

        # Load correct answers from Answer.txt
        self.answer_key = self.load_answer_key("Answer.txt") #เรียกอ่านไฟล์เฉลย แล้วเก็บไว้ในตัวแปร

    def load_answer_key(self, file_path):#คำอธิบายเฉลย
        """
        Read correct answers from Answer.txt. 

        Example:
        1
        3
        2

        This means:
        Question 1 = A
        Question 2 = C
        Question 3 = B
        """

        answer_key = {}

        try:
            with open(file_path) as file:
                for question_number, line in enumerate(file):
                    # Convert 1-5 into 0-4 for A-E 
                    answer_key[question_number] = int(line.strip()) - 1

        except FileNotFoundError:
            messagebox.showerror("Error", f"Cannot find file: {file_path}")

        return answer_key

    # ---------- Image Processing Methods ----------

    def find_markers(self, image, min_area=80):
        """Find a marker near the top-right corner of the answer sheet."""

        height, width = image.shape[:2]

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 0) #หามาร์กเกอร์

        binary_image = cv2.threshold(
            blurred_image,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        contours, _ = cv2.findContours( #หาเส้นขอบ
            binary_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        markers = []

        for contour in contours:
            x, y, contour_width, contour_height = cv2.boundingRect(contour)

            if contour_width * contour_height > min_area:
                center_x = x + contour_width // 2
                center_y = y + contour_height // 2

                if center_x >= width * 0.90 and center_y <= height * 0.10:
                    markers.append((center_x, center_y))

        return markers

    def fix_orientation(self, image):
        """Rotate the answer sheet until the marker is top-right."""

        for _ in range(4):
            height, width = image.shape[:2]
            markers = self.find_markers(image)

            if markers:
                marker_x, marker_y = max(
                    markers,
                    key=lambda point: (point[0], -point[1])
                )

                if marker_x > width * 0.95 and marker_y < height * 0.05:
                    return image

            image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)

        return image

    def reorder_points(self, points):
        """Reorder four corners for perspective transformation."""

        points = points.reshape((4, 2))
        ordered_points = np.zeros((4, 1, 2), np.int32)

        point_sum = points.sum(axis=1)

        ordered_points[0] = points[np.argmin(point_sum)]
        ordered_points[3] = points[np.argmax(point_sum)]

        point_difference = np.diff(points, axis=1)

        ordered_points[1] = points[np.argmin(point_difference)]
        ordered_points[2] = points[np.argmax(point_difference)]

        return ordered_points

    def get_warped_images(self, image, answer_width=1800, answer_height=2339):
        """
        Find answer-sheet rectangles and create warped images.

        Return:
        warped_id: Student ID area
        warped_answers: Answer-sheet area
        """

        image = cv2.resize(image, (1654, 2339))

        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        blurred_image = cv2.GaussianBlur(gray_image, (5, 5), 1)
        edge_image = cv2.Canny(blurred_image, 10, 50)

        contours, _ = cv2.findContours(
            edge_image,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_NONE
        )

        rectangles = []

        for contour in contours:
            area = cv2.contourArea(contour)

            if area > 50:
                perimeter = cv2.arcLength(contour, True)
                corners = cv2.approxPolyDP(contour, 0.02 * perimeter, True)

                if len(corners) == 4:
                    rectangles.append(contour)

        rectangles = sorted(rectangles, key=cv2.contourArea, reverse=True)

        if len(rectangles) < 2:
            return None, None

        # The largest rectangle is the answer-sheet area
        answer_corners = self.reorder_points(
            cv2.approxPolyDP(
                rectangles[0],
                0.02 * cv2.arcLength(rectangles[0], True),
                True
            )
        )

        # The second largest rectangle is the Student ID area
        id_corners = self.reorder_points(
            cv2.approxPolyDP(
                rectangles[1],
                0.02 * cv2.arcLength(rectangles[1], True),
                True
            )
        )

        answer_matrix = cv2.getPerspectiveTransform(
            np.float32(answer_corners),
            np.float32([
                [0, 0],
                [answer_width, 0],
                [0, answer_height],
                [answer_width, answer_height]
            ])
        )

        warped_answers = cv2.warpPerspective(
            image,
            answer_matrix,
            (answer_width, answer_height)
        )

        id_matrix = cv2.getPerspectiveTransform(
            np.float32(id_corners),
            np.float32([
                [0, 0],
                [325, 0],
                [0, 190],
                [325, 190]
            ])
        )

        warped_id = cv2.warpPerspective(image, id_matrix, (325, 190))

        return warped_id, warped_answers

    # ---------- OMR Detection Methods ---------- ปรับตรงนี้

    def detect_student_id(self, binary_region, digits=10, choices=10, top_offset=40):
        """Detect a 10-digit Student ID."""

        height, width = binary_region.shape

        usable_height = height - top_offset
        column_width = width // digits
        row_height = usable_height // choices

        student_id = ""

        for digit_position in range(digits): #digitแรก
            x1 = digit_position * column_width
            x2 = (digit_position + 1) * column_width

            digit_column = binary_region[top_offset:, x1:x2]
            scores = []

            for number in range(choices): #ฝนที่เลขไหน
                y1 = number * row_height
                y2 = (number + 1) * row_height

                cell = digit_column[y1:y2, :]
                black_pixels = cv2.countNonZero(cell)

                scores.append(black_pixels)

            if max(scores) < 50: #pixelถึง50ไหม
                student_id += "?" #ถ้าอ่านไม่ได้จะขึ้น ?
            else:
                student_id += str(np.argmax(scores))

        return student_id

    def check_answers_and_draw(
        self,
        binary_region,
        result_image,
        start_question,
        column_x,
        questions=35,
        choices=5
    ):
        """
        Detect answers, calculate score, and draw markers.

        Green circle = correct answer from Answer.txt
        Red X = student-selected answer
        """

        height, width = binary_region.shape

        top_offset = 20
        fill_threshold = 0.25
        answer_y = 130

        usable_height = height - top_offset
        row_height = usable_height // questions
        column_width = width // choices

        score = 0

        # Check each question in this answer column
        for question in range(questions):
            question_number = start_question + question

            # Stop when all Answer.txt questions are processed
            if question_number >= len(self.answer_key):
                break

            # Get the correct answer for this question
            correct_answer = self.answer_key[question_number]

            # Find the bubble area for the correct answer
            correct_x1 = correct_answer * column_width
            correct_x2 = (correct_answer + 1) * column_width

            correct_y1 = top_offset + question * row_height
            correct_y2 = top_offset + (question + 1) * row_height

            # Find the center of the correct bubble
            correct_center_x = (correct_x1 + correct_x2) // 2
            correct_center_y = (correct_y1 + correct_y2) // 2

            # Draw a green circle on the correct answer bubble
            cv2.circle(
                result_image,
                (
                    column_x + correct_center_x,
                    answer_y + correct_center_y
                ),
                14,
                (0, 255, 0),  # Green in BGR format
                2
            )

             
            selected_answer = -1
            marked_count = 0  # <--- 1. เพิ่มตัวแปรนับจำนวนข้อที่ถูกฝนในข้อนี้

            # Check every choice from A to E
            for choice in range(choices):
                x1 = choice * column_width
                x2 = (choice + 1) * column_width

                y1 = top_offset + question * row_height
                y2 = top_offset + (question + 1) * row_height

                # Crop one bubble area
                bubble = binary_region[y1:y2, x1:x2]

                # Calculate the filled area ratio
                filled_ratio = cv2.countNonZero(bubble) / bubble.size

                # Check whether this bubble is marked
                if filled_ratio > fill_threshold:
                    marked_count += 1  # <--- 2. นับเพิ่มเมื่อเจอช่องที่ถูกฝน
                    selected_answer = choice

                    # Find the center of the marked bubble
                    center_x = (x1 + x2) // 2
                    center_y = (y1 + y2) // 2

                    # Draw a red X on the marked bubble
                    cv2.drawMarker(
                        result_image,
                        (
                            column_x + center_x,
                            answer_y + center_y
                        ),
                        (0, 0, 255),  # Red in BGR format
                        markerType=cv2.MARKER_TILTED_CROSS,
                        markerSize=18,
                        thickness=2
                    )

            # --- 3. เพิ่มเงื่อนไขดักจับ: ถ้าฝนมากกว่า 1 ช่อง (รวมถึงฝนทุกข้อ) ให้ถือว่าผิด (ได้ 0) ---
            if marked_count > 1:
                selected_answer = -1

            # Give one point only when one answer is selected
             
            if selected_answer == correct_answer: #ฝนข้อเดียวกับที่เฉลยไหม
                score += 1

        return score

    # ---------- Main GUI Method ----------

    def load_image(self):
        """Load an answer sheet, detect Student ID, and calculate score."""

        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png")]
        )

        if not file_path:
            return

        image = cv2.imread(file_path)

        if image is None:
            messagebox.showerror("Error", "Cannot open the selected image.")
            return

        # Rotate the image to the correct orientation
        image = self.fix_orientation(image)

        # Create warped Student ID and answer-sheet regions
        warped_id, warped_answers = self.get_warped_images(image)

        if warped_id is None or warped_answers is None:
            messagebox.showerror(
                "Error",
                "Cannot detect the answer-sheet areas."
            )
            return

        # Convert the Student ID area into a binary image
        gray_id = cv2.cvtColor(warped_id, cv2.COLOR_BGR2GRAY)
        blurred_id = cv2.GaussianBlur(gray_id, (5, 5), 0)

        binary_id = cv2.threshold(
            blurred_id,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        # Detect and display Student ID
        student_id = self.detect_student_id(binary_id)
        self.id_label.config(text=f"Student ID: {student_id}")

        # Convert the answer area into a binary image
        gray_answers = cv2.cvtColor(warped_answers, cv2.COLOR_BGR2GRAY)
        blurred_answers = cv2.GaussianBlur(gray_answers, (5, 5), 0)

        binary_answers = cv2.threshold(
            blurred_answers,
            0,
            255,
            cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
        )[1]

        # Crop the left and right answer columns
        column_1 = binary_answers[130:2335, 50:305]
        column_2 = binary_answers[130:2335, 350:605]

        # Copy the answer sheet before drawing markers
        result_image = warped_answers.copy()

        # Detect and score questions 1-35 in the left column
        score_1 = self.check_answers_and_draw(
            column_1,
            result_image,
            start_question=0,
            column_x=50
        )

        # Detect and score questions 36-70 in the right column
        score_2 = self.check_answers_and_draw(
            column_2,
            result_image,
            start_question=35,
            column_x=350
        )

        # Combine scores from both columns
        total_score = score_1 + score_2

        # Display the final score
        self.score_label.config(
            text=f"Score: {total_score}/{len(self.answer_key)}"
        )

        # Convert the result image for Tkinter
        result_image = cv2.cvtColor(result_image, cv2.COLOR_BGR2RGB)
        result_image = Image.fromarray(result_image)
        result_image.thumbnail((800, 650))

        self.tk_image = ImageTk.PhotoImage(result_image)

        # Clear the previous image and show the new image
        self.canvas.delete("all")
        self.canvas.create_image(
            0,
            0,
            anchor=tk.NW,
            image=self.tk_image
        )


if __name__ == "__main__":
    root = tk.Tk() #สร้างกรอบของแอป
    app = OMRChecker(root) #เรียกคลาส
    root.mainloop()
