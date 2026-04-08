import cv2
import numpy as np

path = '../../testimage/s1.jpg'  # Replace with your image path
img = cv2.imread(path)
if img is None:
    raise FileNotFoundError(f"Cannot open image: {path}")

def measure_blue_line(img):
    # 1. Focus on the bottom-right corner (last 25% of height and width)
    h, w = img.shape[:2]
    roi = img[int(h*0.75):h, int(w*0.75):w]

    # 2. Convert to HSV and mask the blue color
    hsv_roi = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    lower_blue = np.array([100, 150, 50]) # Adjust saturation/value if line is pale
    upper_blue = np.array([140, 255, 255])
    mask = cv2.inRange(hsv_roi, lower_blue, upper_blue)

    # 3. Find contours of the blue line
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return "No blue line detected in the bottom-right corner."

    # Get the largest blue contour (the line)
    largest_cnt = max(contours, key=cv2.contourArea)

    # 4. Use MinAreaRect to handle the slant
    rect = cv2.minAreaRect(largest_cnt)
    (x, y), (w_line, h_line), angle = rect

    # The length is the longer of the two dimensions
    line_length = max(w_line, h_line)

    # Optional: Draw for verification
    box = cv2.boxPoints(rect)
    box = np.intp(box)
    cv2.drawContours(roi, [box], 0, (0, 255, 0), 2)

    return line_length

# Usage in your main loop:
blue_line_len = measure_blue_line(img)
print(f"Blue Line Length: {blue_line_len:.2f} px")