# 📏 Lens Boundary Detection & Measurement (Local Windows + OpenCV)
# Install dependencies (run once in terminal):
# pip install opencv-python numpy

import cv2
import numpy as np
import os
from datetime import datetime

# --- Configuration ---
# Update this path to your local image file
# image_path = r"C:\YJ\SP\LSP\Images\S_O_T_cosmetic_1.jpg"
# image_path = r"C:\YJ\SP\LSP\Images\S_O_T_cosmetic_10.jpg"
# image_path = r"C:\YJ\SP\LSP\rawData\Lens4_Lens_200X_Cosmetic.jpg"
# image_path = r"C:\YJ\SP\LSP\rawData\Lens5_Lens_200X_Cosmetic.jpg"
# image_path = r"C:\YJ\SP\LSP\rawData\Lens6_Lens_200X_Cosmetic.jpg"
# image_path = r"C:\YJ\SP\LSP\rawData\Lens11_cosmetic_200X_GFT JIG.jpg"
# image_path = r"C:\YJ\SP\LSP\rawData\Lens16_cosmetic_200X_GFT JIG.jpg"
image_path = r"C:\YJ\SP\LSP\rawData\Sabic_Sample_1_Cosmetics.jpg"
# image_path = r"C:\YJ\SP\LSP\rawData\Sabic_Sample_1_Cosmetics_3D_200X.jpg"

# Or use a relative path from script directory:
# image_path = "test_image.jpg"

# --- Load image with error handling ---
if not os.path.exists(image_path):
    raise FileNotFoundError(f"Image not found: {image_path}\nPlease update the path.")

img = cv2.imread(image_path)
if img is None:
    raise ValueError(f"Failed to load image from: {image_path}")

print(f"✓ Image loaded: {img.shape}")
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# --- Step 1: Improved Yellow Detection (based on color sampling) ---
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)

# --- Step 2: Define yellow range ---
lower_yellow = np.array([15, 60, 60])
upper_yellow = np.array([45, 255, 255])

# # Step 2: Define yellow range (optimized for both grey and black backgrounds)
# lower_yellow = np.array([15, 90, 90])   # H, S, V
# upper_yellow = np.array([38, 255, 255])

# Step 3: Create yellow mask
mask_yellow = cv2.inRange(hsv, lower_yellow, upper_yellow)

# Step 4: Remove shadow/halo (low saturation/value regions)
shadow_mask = cv2.inRange(hsv, np.array([0, 0, 0]), np.array([179, 80, 140]))
mask_no_shadow = cv2.bitwise_and(mask_yellow, cv2.bitwise_not(shadow_mask))

# Step 5: Reinforce with LAB b-channel (yellow has high b values)
b_channel = lab[:, :, 2]
b_mask = cv2.inRange(b_channel, 145, 255)
mask_combined = cv2.bitwise_and(mask_yellow, b_mask)

# Step 6: Morphological cleanup
kernel = np.ones((3,3), np.uint8)
mask_clean = cv2.morphologyEx(mask_combined, cv2.MORPH_CLOSE, kernel, iterations=2)
mask_clean = cv2.morphologyEx(mask_clean, cv2.MORPH_OPEN, kernel, iterations=1)

# --- Step 5: Find contours ---
contours, hierarchy = cv2.findContours(mask_clean, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

outer_contours = []
inner_contours = []

if hierarchy is not None:
    for i, h in enumerate(hierarchy[0]):
        parent = h[3]
        if parent == -1:
            outer_contours.append(contours[i])
        else:
            inner_contours.append(contours[i])

# --- Step 6: Visualize and fit rectangles ---
# Work with BGR image for saving
vis = img.copy()

def draw_rotated_rect(contour, color, label):
    """Fit and draw a rotated bounding box with size display."""
    rect = cv2.minAreaRect(contour)
    (cx, cy), (w, h), angle = rect
    box = cv2.boxPoints(rect)
    box = box.astype(int)
    cv2.drawContours(vis, [box], 0, color, 3)
    cv2.circle(vis, (int(cx), int(cy)), 4, (255,255,255), -1)
    cv2.putText(vis, f"{label}: {w:.1f}x{h:.1f}px", (int(cx - 80), int(cy - 10)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

# --- Outer contour rectangle (GREEN) ---
if outer_contours:
    largest_outer = max(outer_contours, key=cv2.contourArea)
    draw_rotated_rect(largest_outer, (0, 255, 0), "Outer")

# --- Inner contour rectangle (RED) ---
if inner_contours:
    largest_inner = max(inner_contours, key=cv2.contourArea)
    draw_rotated_rect(largest_inner, (0, 0, 255), "Inner")

# --- Step 7: Save results to history folder ---
# Create history folder if it doesn't exist
history_folder = "history"
os.makedirs(history_folder, exist_ok=True)

# Generate timestamp and output filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
input_filename = os.path.splitext(os.path.basename(image_path))[0]
output_filename = f"{input_filename}_annotated_{timestamp}.jpg"
output_path = os.path.join(history_folder, output_filename)

# Save annotated image
cv2.imwrite(output_path, vis)
print(f"✓ Annotated image saved: {output_path}")

# Save mask image
mask_filename = f"{input_filename}_mask_{timestamp}.jpg"
mask_path = os.path.join(history_folder, mask_filename)
cv2.imwrite(mask_path, mask_clean)
print(f"✓ Mask image saved: {mask_path}")

print(f"\n✓ Processing complete!")
print(f"  - Outer contours found: {len(outer_contours)}")
print(f"  - Inner contours found: {len(inner_contours)}")
