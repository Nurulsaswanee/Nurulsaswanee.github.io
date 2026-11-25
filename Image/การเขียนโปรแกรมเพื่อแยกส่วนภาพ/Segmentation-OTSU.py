import cv2

# Load the input image
image = cv2.imread('coin.jpg')  # Replace with your actual image path
image = cv2.resize(image, (300, 300))  # Resize for faster processing and consistency

# Convert the image to Grayscale (Mandatory step for Otsu's Thresholding)
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# Apply Otsu's Thresholding
threshold_value, thresh_image = cv2.threshold(
    gray_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
)

# Print the automatically calculated threshold value to the terminal
print(f"Otsu's Calculated Threshold Value: {threshold_value}")

# Display the results on screen
cv2.imshow('Original Image', image)
cv2.imshow("Otsu's Thresholding Result", thresh_image)

# Keep windows open until any key is pressed, then clean up memory
cv2.waitKey(0)
cv2.destroyAllWindows()
