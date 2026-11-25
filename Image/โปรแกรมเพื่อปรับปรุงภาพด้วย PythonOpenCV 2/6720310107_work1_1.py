import cv2
import numpy as np

gamma = 0.5
c = 15

# Load an image in color format
img_color = cv2.imread('church.png', cv2.IMREAD_COLOR) 

# Convert the color image to a grayscale image
img_gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

# Initialize a new blank image with the same size as img_gray, using float64 to prevent overflow
newimage = np.zeros_like(img_gray, dtype=np.float64)

# Adjust all pixels
newimage = c * img_gray.astype(np.float64) ** gamma
        
# Constrain pixel values between 0 and 255 to ensure valid image data limits
newimage_clipped = np.clip(newimage, 0, 255)

# Convert data type back to uint8 so OpenCV GUI can display the image correctly
newimage_to_show = newimage_clipped.astype(np.uint8)

# Show both the original and modified images in separate windows
cv2.imshow('Original image', img_gray)
cv2.imshow('New image', newimage_to_show)

# Hold the screen window open until the user presses any key
cv2.waitKey(0)

# Remove and delete all created GUI windows from screen and system memory
cv2.destroyAllWindows()
