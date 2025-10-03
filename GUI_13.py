import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from ultralytics import YOLO
import datetime  # For timestamp saving

# === Load YOLO Models ===
model_lens = YOLO("200x_lens.pt")       # For lens/circles
model_rectangle = YOLO("outer_rect.pt")  # For rectangles (out - segmentation model)
# model_rectangle = YOLO("40x_rectt.pt")  # For rectangles (in/out)
model_defects = YOLO("defects.pt")      # For defect detection

# === Constants ===
OUT_RECT_LENGTH = 7.00
OUT_RECT_BREADTH = 4.90
IN_RECT_LENGTH = 5.60
IN_RECT_BREADTH = 2.40
RECT_TOL = 0.010

TARGET_LENS_DIAMETER = 0.240
LENS_TOL = 0.005

PIXEL_SCALES = {"40x": 380, "200x": 1940}

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

# === App Window Configuration ===
ctk.set_appearance_mode("light")  # Set light mode for better visibility
ctk.set_default_color_theme("blue")  # Use blue theme for professional look
app = ctk.CTk()

# Responsive window configuration
def get_screen_size():
    """Get screen dimensions for responsive sizing"""
    return app.winfo_screenwidth(), app.winfo_screenheight()

def set_responsive_geometry():
    """Set responsive window size based on screen dimensions"""
    screen_width, screen_height = get_screen_size()
    
    # Calculate responsive dimensions (80% of screen size, with constraints)
    window_width = min(max(int(screen_width * 0.8), 1200), 1600)  # Between 1200-1600px
    window_height = min(max(int(screen_height * 0.8), 800), 1000)  # Between 800-1000px
    
    # Center the window
    x = (screen_width - window_width) // 2
    y = (screen_height - window_height) // 2
    
    app.geometry(f"{window_width}x{window_height}+{x}+{y}")
    app.minsize(1000, 600)  # Minimum usable size

set_responsive_geometry()
app.title("Lens Quality Check")
app.configure(fg_color="#e0e0e0")  # Lighter grey background for main window

# --- Globals ---
uploaded_image = None
uploaded_image_path = None
annotated_image = None
measure_type = "rectangle"
image_preview_tk = None
annotated_image_tk = None
results_lines = []
tabs_created = False
tabs = None

zoom_level = 1.0
min_zoom = 0.1
max_zoom = 5.0
pan_start_x = 0
pan_start_y = 0
annotated_canvas = None
canvas_img_id = None

upload_tab_upload_btn = None
upload_tab_run_detection_btn = None
preview_img_label = None

# --- Responsive Sidebar ---
def get_responsive_sidebar_width():
    """Calculate responsive sidebar width based on window size"""
    window_width = app.winfo_width()
    if window_width < 1200:
        return 250  # Narrower sidebar for smaller screens
    elif window_width < 1400:
        return 280  # Standard sidebar
    else:
        return 320  # Wider sidebar for larger screens

# Initialize with responsive width
initial_sidebar_width = 280  # Default until window is realized
sidebar = ctk.CTkFrame(app, width=initial_sidebar_width, fg_color="#eaeaea", corner_radius=15)
sidebar.pack(side="left", fill="y", padx=(20, 10), pady=20)  # Added right padding for gap
sidebar.pack_propagate(False)  # Maintain fixed width

# Add visual separator/gap between sidebar and main frame
# separator = ctk.CTkFrame(app, width=2, fg_color="#d0d0d0", corner_radius=0)  # Light grey separator
# separator.pack(side="left", fill="y", pady=20)

# Responsive font sizing
def get_responsive_font_size(base_size, scale_factor=1.0):
    """Calculate responsive font size based on window dimensions"""
    window_width = app.winfo_width()
    if window_width < 1200:
        return int(base_size * 0.9 * scale_factor)  # Smaller fonts for smaller screens
    elif window_width > 1500:
        return int(base_size * 1.1 * scale_factor)  # Larger fonts for larger screens
    else:
        return int(base_size * scale_factor)

# Header with responsive font
def update_header_font():
    """Update header font size responsively"""
    font_size = get_responsive_font_size(22)
    project_label.configure(font=("Arial", font_size, "bold"))

project_label = ctk.CTkLabel(sidebar, text="Lens Quality Check", 
                           font=("Arial", 22, "bold"),
                           wraplength=200)
project_label.pack(pady=15)

def load_logo_image(path, max_w=200, max_h=100):
    try:
        img = Image.open(path)
        img.thumbnail((max_w, max_h))
        return ImageTk.PhotoImage(img)
    except Exception:
        return None

logo1_img = load_logo_image("Screenshot 2025-07-10 150120.png")
logo2_img = load_logo_image("sp_informal_logo_300.png")

if logo1_img:
    logo1_label = ctk.CTkLabel(sidebar, image=logo1_img, text="")
    logo1_label.pack(pady=(10,5))

collab_label = ctk.CTkLabel(sidebar, text="in collaboration with", font=("Arial", 14))
collab_label.pack(pady=(5,5))

if logo2_img:
    logo2_label = ctk.CTkLabel(sidebar, image=logo2_img, text="")
    logo2_label.pack(pady=(5,20))

mode_var = ctk.StringVar(value="measurement")
ctk.CTkLabel(sidebar, text="Mode:").pack(pady=(0,5))
# Mode selection dropdown with consistent styling
mode_dropdown = ctk.CTkOptionMenu(sidebar, 
                                variable=mode_var, 
                                values=["measurement", "defect"],
                                corner_radius=10)  # Consistent corner radius
mode_dropdown.pack(pady=(0,20))  # Standard vertical spacing

# Zoom level selection with consistent styling
zoom_var = ctk.StringVar(value="40x")
zoom_options = ["40x", "200x"]
zoom_dropdown = ctk.CTkOptionMenu(sidebar, 
                                variable=zoom_var, 
                                values=zoom_options, 
                                command=lambda _: update_measurement_type(),
                                corner_radius=10)  # Consistent corner radius
zoom_dropdown.pack(pady=(0,20))  # Standard vertical spacing

# Status Label
# Status label with standardized font and wrapping
status_label = ctk.CTkLabel(sidebar, 
                           text="Current Mode: Rectangle Measurement (40x)", 
                           wraplength=200,  # Standard sidebar text wrapping
                           justify="center", 
                           font=("Arial", 12),  # Standard status text size
                           corner_radius=10)  # Consistent corner radius
status_label.pack(pady=(10,20))  # Standard vertical spacing

def update_status_label(*args):
    mode = mode_var.get()
    zoom = zoom_var.get()
    if mode == "defect":
        status_text = "Current Mode:\nDefect Detection"
    else:  # measurement mode
        measure_type = "Rectangle" if zoom == "40x" else "Lens"
        status_text = f"Current Mode:\n{measure_type} Measurement ({zoom})"
    status_label.configure(text=status_text)

# Add trace to both variables to update status
mode_var.trace_add("write", update_status_label)
zoom_var.trace_add("write", update_status_label)

# Initialize status label
update_status_label()

main_frame = ctk.CTkFrame(app, fg_color="#eaeaea", corner_radius=15)
main_frame.pack(side="left", fill="both", expand=True, padx=(10, 20), pady=20)  # Left padding for gap

# Content container for better organization
content_frame = ctk.CTkFrame(main_frame, fg_color="#eaeaea")
content_frame.pack(fill="both", expand=True, padx=10, pady=10)

# Auto-update measurement type based on zoom level
def update_measurement_type():
    global measure_type
    if zoom_var.get() == "40x":
        measure_type = "rectangle"
    else:
        measure_type = "lens"

zoom_var.trace_add("write", lambda *args: update_measurement_type())

# === Upload & Preview Functions ===
def upload_image():
    global uploaded_image, uploaded_image_path, image_preview_tk, zoom_level
    path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
    if not path:
        return
    img = cv2.imread(path)
    if img is None:
        messagebox.showerror("Error", "Cannot load image.")
        return
    uploaded_image = img
    uploaded_image_path = path
    zoom_level = 1.0
    show_preview_image(img)
    if upload_tab_run_detection_btn:
        upload_tab_run_detection_btn.configure(state="normal")

def show_preview_image(img):
    global image_preview_tk
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    
    # Calculate aspect ratio for resizing
    aspect_ratio = img_pil.width / img_pil.height
    
    # Get responsive dimensions based on available space
    def get_available_preview_space():
        """Calculate available space for image preview"""
        window_width = app.winfo_width()
        window_height = app.winfo_height()
        
        # Account for sidebar and padding
        available_width = max(window_width - 400, 400)  # Subtract sidebar + padding
        available_height = max(window_height - 200, 300)  # Subtract header + controls
        
        return available_width, available_height
    
    available_width, available_height = get_available_preview_space()
    
    # Calculate target dimensions while maintaining aspect ratio
    target_width = min(available_width * 0.8, 800)  # Max 80% of available width
    target_height = int(target_width / aspect_ratio)
    
    # Ensure height doesn't exceed available space
    if target_height > available_height * 0.8:
        target_height = int(available_height * 0.8)
        target_width = int(target_height * aspect_ratio)
    
    # Ensure minimum usable size
    target_width = max(target_width, 300)
    target_height = max(target_height, 200)
    
    img_pil = img_pil.resize((int(target_width), int(target_height)), Image.LANCZOS)
    image_preview_tk = ctk.CTkImage(light_image=img_pil, size=(int(target_width), int(target_height)))
    
    if preview_img_label:
        preview_img_label.configure(image=image_preview_tk, text="")

# === Detection Function (Measurement & Defect) ===
def run_detection():
    global annotated_image, annotated_image_tk, results_lines, tabs_created, tabs, zoom_level

    if uploaded_image is None:
        messagebox.showwarning("Warning", "Please upload an image first.")
        return

    mode = mode_var.get()
    pixel_scale = PIXEL_SCALES.get(zoom_var.get(), 380)

    annotated = uploaded_image.copy()
    results_text = []

    # --- Measurement Logic ---
    if mode == "measurement":
        if measure_type == "lens":
            # Lens measurement logic (unchanged)
            res_lens = model_lens(uploaded_image)[0]
            circles = []
            for box, cls in zip(res_lens.boxes.xyxy.cpu().numpy(), res_lens.boxes.cls.cpu().numpy()):
                x1, y1, x2, y2 = map(int, box)
                label = res_lens.names[int(cls)].lower()
                if label != "lens":
                    continue
                w, h = x2 - x1, y2 - y1
                center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
                diameter_px = w
                diameter_mm = diameter_px / pixel_scale
                in_tol = abs(diameter_mm - TARGET_LENS_DIAMETER) <= LENS_TOL
                color = (0, 255, 0) if in_tol else (0, 0, 255)  # Green if in tol else Red
                cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
                circles.append((center_x, center_y, diameter_mm, in_tol))

            circles.sort(key=lambda c: c[0])
            for idx, (cx, cy, d_mm, in_tol) in enumerate(circles, start=1):
                label = f"Lens {idx}"
                # Find the top of the box for label placement
                # Find the corresponding box for this lens
                box = None
                for box_candidate, cls in zip(res_lens.boxes.xyxy.cpu().numpy(), res_lens.boxes.cls.cpu().numpy()):
                    x1, y1, x2, y2 = map(int, box_candidate)
                    label_candidate = res_lens.names[int(cls)].lower()
                    center_x, center_y = (x1 + x2)//2, (y1 + y2)//2
                    if label_candidate == "lens" and abs(center_x - cx) < 5 and abs(center_y - cy) < 5:
                        box = (x1, y1, x2, y2)
                        break
                if box:
                    label_x = int((box[0] + box[2]) // 2) - 30
                    label_y = int(box[1]) - 20
                else:
                    label_x = cx - 30
                    label_y = cy - 30
                cv2.putText(annotated, label, (label_x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,0), 5)
                # Annotate diameter below the box
                if box:
                    diam_x = int(box[0]) + 10  # Start from left edge, with small padding
                    diam_y = int(box[3]) + 80  # Lower, so it doesn't overlap with box edge
                else:
                    diam_x = cx - 60
                    diam_y = cy + 80
                diam_text = f"Dia: {d_mm:.3f}mm"
                cv2.putText(annotated, diam_text, (diam_x, diam_y), cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0,0,255) if not in_tol else (0,128,0), 4)
                results_text.append(f"{label} Diameter: {d_mm:.3f}mm {'OK' if in_tol else 'Out of Tolerance'}")

            # Calculate lens-to-lens distances and record in text output only (no image annotation)
            for i in range(len(circles) - 1):
                c1 = circles[i]
                c2 = circles[i+1]
                dist_mm = abs(c2[0] - c1[0]) / pixel_scale
                results_text.append(f"Center-to-center distance Lens {i+1} to Lens {i+2}: {dist_mm:.3f}mm")

            if len(circles) > 1:
                total_dist_mm = abs(circles[-1][0] - circles[0][0]) / pixel_scale
                results_text.append(f"Center-to-center distance Lens 1 to Lens {len(circles)}: {total_dist_mm:.3f}mm")
        
        # --- Rectangle Measurement Logic (Segmentation) ---
        elif measure_type == "rectangle":
            # Run detection and get results with confidence threshold
            res_rect = model_rectangle(uploaded_image, conf=0.7)[0]
            
            # # Debug information
            # print(f"Model output available fields: {dir(res_rect)}")
            # if hasattr(res_rect, 'masks') and res_rect.masks is not None:
            #     print(f"Number of masks: {len(res_rect.masks.data)}")
            # if hasattr(res_rect, 'boxes'):
            #     print(f"Number of boxes: {len(res_rect.boxes)}")
            #     print(f"Classes detected: {res_rect.boxes.cls.cpu().numpy()}")
            #     print(f"Available class names: {res_rect.names}")
            
            # Lists to store detected rectangles
            out_rects = []

                # Process masks if available
            if hasattr(res_rect, 'masks') and res_rect.masks is not None:
                # Get image dimensions
                img_height, img_width = uploaded_image.shape[:2]
                print(f"\nImage dimensions: {img_width}x{img_height}")
                
                # Create a combined mask at the image resolution
                combined_mask = np.zeros((img_height, img_width), dtype=np.uint8)
                
                # Process each mask and class from the segmentation results
                for i in range(len(res_rect.masks.data)):
                    # Get mask and ensure it's in the correct format
                    mask = res_rect.masks.data[i].cpu().numpy()
                    
                    # Get mask dimensions and resize if needed
                    mask_height, mask_width = mask.shape
                    if mask_height != img_height or mask_width != img_width:
                        mask = cv2.resize(mask, (img_width, img_height))
                    
                    # Get original image dimensions
                    img_height, img_width = uploaded_image.shape[:2]
                    mask_height, mask_width = mask.shape[:2]
                    
                    # Convert mask to proper binary image format with optimized threshold
                    # Scale the kernel sizes based on image dimensions
                    scale_factor = min(img_width, img_height) / 800  # baseline for scaling
                    
                    # Convert float mask to binary
                    mask = (mask > 0.3).astype("uint8") * 255
                    
                    # Scale kernel sizes based on image resolution
                    small_size = max(3, int(3 * scale_factor))
                    large_size = max(5, int(5 * scale_factor))
                    
                    # Ensure kernel sizes are odd
                    small_size = small_size + 1 if small_size % 2 == 0 else small_size
                    large_size = large_size + 1 if large_size % 2 == 0 else large_size
                    
                    kernel_small = np.ones((small_size, small_size), np.uint8)
                    kernel_large = np.ones((large_size, large_size), np.uint8)
                    
                    # Apply morphological operations for clean contours
                    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=1)
                    mask = cv2.dilate(mask, kernel_small, iterations=1)
                    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_large, iterations=1)
                    mask = cv2.erode(mask, kernel_small, iterations=1)
                    
                    # Update combined mask
                    combined_mask = cv2.bitwise_or(combined_mask, mask)
                    
                    # Apply threshold to ensure binary mask
                    _, mask_binary = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
                    
                    # Find contours in the binary mask
                    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    if not contours:
                        continue
                    
                    # Sort contours by area in descending order
                    contours = sorted(contours, key=cv2.contourArea, reverse=True)
                    
                    # Process the largest contour
                    contour = contours[0]
                    contour_area = cv2.contourArea(contour)
                    
                    # Use a slightly larger epsilon for stable rectangle detection
                    epsilon = 0.005 * cv2.arcLength(contour, True)
                    approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                    
                    # If the approximated contour has too few points, use a smaller epsilon
                    if len(approx_contour) < 10:
                        epsilon = 0.002 * cv2.arcLength(contour, True)
                        approx_contour = cv2.approxPolyDP(contour, epsilon, True)
                                        
                    # Get the minimum area rectangle from both contours and pick the better one
                    rect_orig = cv2.minAreaRect(contour)
                    rect_approx = cv2.minAreaRect(approx_contour)
                    
                    # Function to normalize rectangle measurements
                    def normalize_rect(rect):
                        (cx, cy), (w, h), angle = rect
                        if angle < -45:
                            angle += 90
                            w, h = h, w
                        return (cx, cy), (w, h), angle
                    
                    # Function to score rectangle quality
                    def score_rect(rect):
                        _, (w, h), _ = normalize_rect(rect)
                        w_mm, h_mm = w/pixel_scale, h/pixel_scale
                        length_mm, breadth_mm = max(w_mm, h_mm), min(w_mm, h_mm)
                        length_error = abs(length_mm - OUT_RECT_LENGTH)
                        breadth_error = abs(breadth_mm - OUT_RECT_BREADTH)
                        return length_error + breadth_error
                    
                    # Choose the better rectangle based on measurement error
                    rect = rect_orig if score_rect(rect_orig) < score_rect(rect_approx) else rect_approx
                    (cx, cy), (w, h), angle = normalize_rect(rect)
                    
                    # Get rectangle measurements
                    
                    # Get the pixel scale from the constants
                    pixel_scale = PIXEL_SCALES.get(zoom_var.get(), 120)  # 120 pixels/mm for 40x
                    
                    # Convert image dimensions back to original scale if they were resized
                    if mask_height != img_height or mask_width != img_width:
                        scale_factor_h = img_height / mask_height
                        scale_factor_w = img_width / mask_width
                        scale_factor = (scale_factor_h + scale_factor_w) / 2
                        w = w * scale_factor
                        h = h * scale_factor
                    
                    # Always use the actual width/height regardless of rotation
                    w_actual = max(w, h)  # Longer side
                    h_actual = min(w, h)  # Shorter side
                    length_mm = w_actual / pixel_scale
                    breadth_mm = h_actual / pixel_scale
                    

                    
                    # Get rectangle corners for drawing
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)
                    
                    # Get the minimum area rectangle and measurements
                    rect = cv2.minAreaRect(contour)
                    (cx, cy), (w, h), angle = rect
                    box = cv2.boxPoints(rect)
                    box = np.int32(box)

                    # Normalize dimensions and convert to mm
                    if angle < -45:
                        angle += 90
                        w, h = h, w
                    length_mm = max(w, h) / pixel_scale
                    breadth_mm = min(w, h) / pixel_scale

                    # Check if measurements are within tolerance
                    in_tol = (abs(length_mm - OUT_RECT_LENGTH) <= RECT_TOL and 
                             abs(breadth_mm - OUT_RECT_BREADTH) <= RECT_TOL)

                    # Draw the contour and rectangle
                    line_thickness = max(2, min(img_width, img_height) // 300)
                    cv2.drawContours(annotated, [contour], -1, (255, 0, 0), line_thickness)  # Contour in blue
                    color = (0, 255, 0) if in_tol else (0, 0, 255)  # Green if in tolerance, red if not
                    cv2.drawContours(annotated, [box], 0, color, line_thickness)

                    # Add measurement text
                    text = f"Out Rect: {length_mm:.3f}mm x {breadth_mm:.3f}mm"
                    font_scale = min(img_width, img_height) / 900  # Enlarged font size
                    text_x = int(min(box[:, 0]))  # Leftmost x coordinate
                    text_y = int(min(box[:, 1])) - 20  # Above the top of the rectangle
                    cv2.putText(annotated, text, (text_x, text_y),
                              cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)

                    # Add to results text
                    status = 'OK' if in_tol else 'Out of Tolerance'
                    results_text.append(f"Out Rect: {length_mm:.3f} x {breadth_mm:.3f}mm {status}")

                    # Save only the final annotated image (in the auto-save section later)
            
            # Function to find best matching rectangle
            def choose_best(rects, target_length, target_breadth):
                if not rects:
                    return None
                def dist(r): 
                    return abs(r[1] - target_length) + abs(r[2] - target_breadth)
                return sorted(rects, key=dist)[0]
            
            # No need for choose_best function anymore as we're selecting during processing
            # best_rect will be already set if we found a valid rectangle during mask processing

        # --- Rectangle Measurement Logic (Object Detection) ---
        # elif measure_type == "rectangle":
        #     res_rect = model_rectangle(uploaded_image)[0]
        #     out_rects = []
        #     in_rects = []
        #     for box, cls in zip(res_rect.boxes.xyxy.cpu().numpy(), res_rect.boxes.cls.cpu().numpy()):
        #         x1, y1, x2, y2 = map(int, box)
        #         label = res_rect.names[int(cls)].lower()
        #         w, h = x2-x1, y2-y1
        #         length_mm, breadth_mm = w/pixel_scale, h/pixel_scale
        #         if label=="out": out_rects.append((x1,y1,x2,y2,length_mm,breadth_mm))
        #         elif label=="in": in_rects.append((x1,y1,x2,y2,length_mm,breadth_mm))

        #     def choose_best(rects, target_length, target_breadth):
        #         if not rects: return None
        #         def dist(r): return abs(r[4]-target_length)+abs(r[5]-target_breadth)
        #         return sorted(rects, key=dist)[0]

        #     best_out = choose_best(out_rects, OUT_RECT_LENGTH, OUT_RECT_BREADTH)
        #     best_in = choose_best(in_rects, IN_RECT_LENGTH, IN_RECT_BREADTH)

        #     if best_out:
        #         x1,y1,x2,y2,length_mm,breadth_mm=best_out
        #         in_tol = abs(length_mm-OUT_RECT_LENGTH)<=RECT_TOL and abs(breadth_mm-OUT_RECT_BREADTH)<=RECT_TOL
        #         color = (0,255,0) if in_tol else (0,0,255)
        #         cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
        #         cv2.putText(annotated,f"Out Rect: {length_mm:.3f}mm x {breadth_mm:.3f}mm",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        #         results_text.append(f"Out Rect: {length_mm:.3f} x {breadth_mm:.3f} {'OK' if in_tol else 'Out of Tolerance'}")

        #     if best_in:
        #         x1,y1,x2,y2,length_mm,breadth_mm=best_in
        #         in_tol = abs(length_mm-IN_RECT_LENGTH)<=RECT_TOL and abs(breadth_mm-IN_RECT_BREADTH)<=RECT_TOL
        #         color = (0,255,0) if in_tol else (0,0,255)
        #         cv2.rectangle(annotated,(x1,y1),(x2,y2),color,2)
        #         cv2.putText(annotated,f"In Rect: {length_mm:.3f}mm x {breadth_mm:.3f}mm",(x1,y1-10),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,0,0),2)
        #         results_text.append(f"In Rect: {length_mm:.3f} x {breadth_mm:.3f} {'OK' if in_tol else 'Out of Tolerance'}")

        #     # Function to choose best rectangle
        #     def choose_best(rects, target_length, target_breadth):
        #         if not rects:
        #             return None
        #         def dist(r): return abs(r[1] - target_length) + abs(r[2] - target_breadth)
        #         return sorted(rects, key=dist)[0]


        #     # Pick best candidate
        #     best_out = choose_best(out_rects, OUT_RECT_LENGTH, OUT_RECT_BREADTH)
        #     best_in = choose_best(in_rects, IN_RECT_LENGTH, IN_RECT_BREADTH)


        #     # Check tolerance and annotate
        #     if best_out:
        #         box, length_mm, breadth_mm = best_out
        #         in_tol = abs(length_mm - OUT_RECT_LENGTH) <= RECT_TOL and abs(breadth_mm - OUT_RECT_BREADTH) <= RECT_TOL
        #         color = (0, 255, 0) if in_tol else (0, 0, 255)
        #         cv2.drawContours(annotated, [box], 0, color, 2)
        #         cv2.putText(annotated, f"Out Rect: {length_mm:.3f}mm x {breadth_mm:.3f}mm",
        #                     (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        #         results_text.append(f"Out Rect: {length_mm:.3f} x {breadth_mm:.3f} {'OK' if in_tol else 'Out of Tolerance'}")

        #     if best_in:
        #         box, length_mm, breadth_mm = best_in
        #         in_tol = abs(length_mm - IN_RECT_LENGTH) <= RECT_TOL and abs(breadth_mm - IN_RECT_BREADTH) <= RECT_TOL
        #         color = (0, 255, 0) if in_tol else (0, 0, 255)
        #         cv2.drawContours(annotated, [box], 0, color, 2)
        #         cv2.putText(annotated, f"In Rect: {length_mm:.3f}mm x {breadth_mm:.3f}mm",
        #                     (box[0][0], box[0][1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 2)
        #         results_text.append(f"In Rect: {length_mm:.3f} x {breadth_mm:.3f} {'OK' if in_tol else 'Out of Tolerance'}")

    # --- Defect Logic ---
    elif mode=="defect":
        res_def = model_defects(uploaded_image)[0]
        for box in res_def.boxes.xyxy.cpu().numpy():
            x1,y1,x2,y2=map(int,box)
            cv2.rectangle(annotated,(x1,y1),(x2,y2),(0,0,255),2)
            results_text.append(f"Defect Box: ({x1},{y1}) ({x2},{y2})")

    annotated_image = annotated
    results_lines = results_text

    # --- Auto-save annotated image ---
    if annotated_image is not None:
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(uploaded_image_path))[0]
        # Organize folders
        measurements_dir = os.path.join(HISTORY_DIR, "measurements")
        defects_dir = os.path.join(HISTORY_DIR, "defects")
        meas_40x_dir = os.path.join(measurements_dir, "40x")
        meas_200x_dir = os.path.join(measurements_dir, "200x")
        os.makedirs(measurements_dir, exist_ok=True)
        os.makedirs(defects_dir, exist_ok=True)
        os.makedirs(meas_40x_dir, exist_ok=True)
        os.makedirs(meas_200x_dir, exist_ok=True)

        if mode == "defect":
            save_path = os.path.join(defects_dir, f"{base_name}_annotated_{timestamp}.png")
            txt_path = os.path.join(defects_dir, f"{base_name}_annotated_{timestamp}.txt")
        elif mode == "measurement":
            if zoom_var.get() == "40x":
                save_path = os.path.join(meas_40x_dir, f"{base_name}_annotated_{timestamp}.png")
                txt_path = os.path.join(meas_40x_dir, f"{base_name}_annotated_{timestamp}.txt")
            else:
                save_path = os.path.join(meas_200x_dir, f"{base_name}_annotated_{timestamp}.png")
                txt_path = os.path.join(meas_200x_dir, f"{base_name}_annotated_{timestamp}.txt")
        else:
            save_path = os.path.join(HISTORY_DIR, f"{base_name}_annotated_{timestamp}.png")
            txt_path = os.path.join(HISTORY_DIR, f"{base_name}_annotated_{timestamp}.txt")
        cv2.imwrite(save_path, annotated_image)
        # Save results text output
        with open(txt_path, "w") as f:
            for line in results_lines:
                f.write(line + "\n")

    update_preview_tab()
    update_annotated_tab()
    update_results_tab()
def create_tabs():
    global tabs, preview_img_label, annotated_img_label, results_textbox
    global upload_tab_upload_btn, upload_tab_run_detection_btn
    global annotated_canvas, canvas_img_id

    # Responsive tab configuration
    def get_responsive_tab_padding():
        """Get responsive padding based on window size"""
        window_width = app.winfo_width()
        if window_width < 1200:
            return 10, 10  # Smaller padding for smaller screens
        elif window_width > 1500:
            return 30, 25  # Larger padding for larger screens
        else:
            return 20, 20  # Standard padding

    pad_x, pad_y = get_responsive_tab_padding()

    # Create tabview with responsive styling
    tab_container = ctk.CTkFrame(content_frame, fg_color="#eaeaea", corner_radius=10)
    tab_container.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)
    
    tabs = ctk.CTkTabview(tab_container, 
                         fg_color="#eaeaea",
                         segmented_button_fg_color="#e0e0e0",
                         segmented_button_selected_color="#3b8ed0",
                         segmented_button_selected_hover_color="#36719f",
                         height=50)  # Responsive tab height
    tabs.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)
    
    # Configure responsive tab button style
    responsive_font_size = get_responsive_font_size(14)
    tabs._segmented_button.configure(font=("Arial", responsive_font_size, "bold"))
    tabs._segmented_button.configure(corner_radius=10)
    tabs._segmented_button.grid_configure(padx=pad_x, pady=pad_y//2)
    
    # Create all tabs
    tabs.add("Upload New")
    tabs.add("Annotated Image")
    tabs.add("Results")
    
    # Configure each tab responsively
    for tab_name in ["Upload New", "Annotated Image", "Results"]:
        tab = tabs.tab(tab_name)
        tab.configure(fg_color="#eaeaea")
        # Add responsive padding inside each tab
        inner_frame = ctk.CTkFrame(tab, fg_color="#eaeaea")
        inner_frame.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)

    # Get Upload New tab
    upload_tab = tabs.tab("Upload New")
    upload_content = upload_tab.winfo_children()[0]

    # Responsive button sizing
    button_width = max(min(int(app.winfo_width() * 0.12), 200), 120)  # Slightly smaller for side-by-side
    button_font_size = get_responsive_font_size(14)

    # Create a frame to hold buttons side by side
    button_frame = ctk.CTkFrame(upload_content, fg_color="transparent")
    button_frame.pack(pady=(0, 10))

    # Upload button with responsive sizing
    upload_tab_upload_btn = ctk.CTkButton(button_frame, 
                                         text="Upload Image",
                                         command=lambda: upload_image(),
                                         width=button_width,
                                         height=40,
                                         corner_radius=10,
                                         font=("Arial", button_font_size))
    upload_tab_upload_btn.pack(side="left", padx=(0, 10))

    # Run Detection button with responsive sizing
    upload_tab_run_detection_btn = ctk.CTkButton(button_frame,
                                                text="Run Detection",
                                                command=lambda: run_detection(),
                                                width=button_width,
                                                height=40,
                                                corner_radius=10,
                                                font=("Arial", button_font_size))
    upload_tab_run_detection_btn.pack(side="left", padx=(10, 0))

    # Preview image container - responsive sizing
    preview_frame = ctk.CTkFrame(upload_content, fg_color="#eaeaea", corner_radius=10)
    preview_frame.pack(fill="both", expand=True, padx=pad_x, pady=(10, 0))

    # Responsive preview label - no fixed dimensions, adapts to container
    preview_img_label = ctk.CTkLabel(preview_frame, 
                                   text="No image uploaded", 
                                   fg_color="#eaeaea",
                                   corner_radius=10)
    preview_img_label.pack(expand=True, fill="both", padx=10, pady=10)

    # Get Annotated Image tab
    annotated_tab = tabs.tab("Annotated Image")
    annotated_content = annotated_tab.winfo_children()[0]

    # Responsive annotated image canvas - no fixed dimensions
    annotated_canvas = ctk.CTkCanvas(annotated_content, 
                                   bg="gray90", 
                                   highlightthickness=0)
    annotated_canvas.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)

    canvas_img_id = None

    annotated_canvas.bind("<MouseWheel>", on_mousewheel)
    annotated_canvas.bind("<Button-4>", on_mousewheel)
    annotated_canvas.bind("<Button-5>", on_mousewheel)
    annotated_canvas.bind("<ButtonPress-1>", on_pan_start)
    annotated_canvas.bind("<B1-Motion>", on_pan_move)

    # Get Results tab
    results_tab = tabs.tab("Results")
    results_content = results_tab.winfo_children()[0]
    
    # Responsive results textbox
    results_font_size = get_responsive_font_size(14)
    results_textbox = ctk.CTkTextbox(results_content, 
                                    wrap="word", 
                                    fg_color="white", 
                                    font=("Arial", results_font_size),
                                    corner_radius=10)
    results_textbox.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)
    results_textbox.configure(state="disabled")

    update_preview_tab()
    update_annotated_tab()
    update_results_tab()
    update_upload_tab_buttons_visibility()

def update_preview_tab():
    global image_preview_tk
    if uploaded_image is None:
        return
    
    # Use the responsive show_preview_image function
    show_preview_image(uploaded_image)

def update_annotated_tab():
    global annotated_image_tk, zoom_level, annotated_canvas, canvas_img_id
    if annotated_image is None:
        return
    img_rgb = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    new_size = (int(img_pil.width * zoom_level), int(img_pil.height * zoom_level))
    resized_img = img_pil.resize(new_size, Image.LANCZOS)
    annotated_image_tk = ctk.CTkImage(light_image=resized_img, size=new_size)
    annotated_canvas.delete("all")
    # For Canvas widget we still need to use PhotoImage
    canvas_image = ImageTk.PhotoImage(resized_img)
    canvas_img_id = annotated_canvas.create_image(0, 0, anchor="nw", image=canvas_image)
    annotated_canvas.image = canvas_image  # Keep a reference
    annotated_canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))

def update_results_tab():
    results_textbox.configure(state="normal")
    results_textbox.delete("0.0", "end")
    for line in results_lines:
        if "Out of Tolerance" in line:
            results_textbox.insert("end", line + "\n", ("red",))
        else:
            results_textbox.insert("end", line + "\n")
    results_textbox.tag_config("red", foreground="red")
    results_textbox.configure(state="disabled")

def update_upload_tab_buttons_visibility():
    if tabs is None:
        return

# === Additional Responsive Utilities ===
def refresh_responsive_elements():
    """Refresh all responsive elements when needed"""
    if tabs_created:
        # Update fonts
        update_header_font()
        
        # Update button sizes if they exist
        if upload_tab_upload_btn:
            button_width = max(min(int(app.winfo_width() * 0.12), 200), 120)  # Smaller for side-by-side
            button_font_size = get_responsive_font_size(14)
            upload_tab_upload_btn.configure(width=button_width, font=("Arial", button_font_size))
            upload_tab_run_detection_btn.configure(width=button_width, font=("Arial", button_font_size))
        
        # Update results textbox font if it exists
        if 'results_textbox' in globals():
            results_font_size = get_responsive_font_size(14)
            results_textbox.configure(font=("Arial", results_font_size))
        
        # Refresh image preview
        if uploaded_image is not None:
            show_preview_image(uploaded_image)


def on_mousewheel(event):
    global zoom_level
    if annotated_image is None:
        return
    if event.num == 4 or event.delta > 0:
        zoom_factor = 1.1
    elif event.num == 5 or event.delta < 0:
        zoom_factor = 0.9
    else:
        return
    new_zoom = zoom_level * zoom_factor
    if new_zoom < min_zoom:
        new_zoom = min_zoom
    elif new_zoom > max_zoom:
        new_zoom = max_zoom
    if abs(new_zoom - zoom_level) < 0.001:
        return
    zoom_level = new_zoom
    update_annotated_tab()

def on_pan_start(event):
    global pan_start_x, pan_start_y
    pan_start_x = event.x
    pan_start_y = event.y

def on_pan_move(event):
    global pan_start_x, pan_start_y, annotated_canvas, canvas_img_id
    if canvas_img_id is None:
        return
    dx = event.x - pan_start_x
    dy = event.y - pan_start_y
    annotated_canvas.move(canvas_img_id, dx, dy)
    pan_start_x = event.x
    pan_start_y = event.y

# Show tabbed interface by default on launch (after create_tabs is defined and event handlers are defined)
create_tabs()
tabs_created = True

# === Responsive Window Management ===
def on_window_resize(event=None):
    """Handle window resize events to update responsive elements"""
    if event and event.widget == app:  # Only handle main window resize
        # Update sidebar width
        new_sidebar_width = get_responsive_sidebar_width()
        sidebar.configure(width=new_sidebar_width)
        
        # Refresh all responsive elements
        app.after_idle(refresh_responsive_elements)

# Bind resize event
app.bind("<Configure>", on_window_resize)

# Initial responsive setup after window is realized
app.after(100, lambda: refresh_responsive_elements())

def on_mousewheel(event):
    global zoom_level
    if annotated_image is None:
        return
    if event.num == 4 or event.delta > 0:
        zoom_factor = 1.1
    elif event.num == 5 or event.delta < 0:
        zoom_factor = 0.9
    else:
        return
    new_zoom = zoom_level * zoom_factor
    if new_zoom < min_zoom:
        new_zoom = min_zoom
    elif new_zoom > max_zoom:
        new_zoom = max_zoom
    if abs(new_zoom - zoom_level) < 0.001:
        return
    zoom_level = new_zoom
    update_annotated_tab()

def on_pan_start(event):
    global pan_start_x, pan_start_y
    pan_start_x = event.x
    pan_start_y = event.y

def on_pan_move(event):
    global pan_start_x, pan_start_y, annotated_canvas, canvas_img_id
    if canvas_img_id is None:
        return
    dx = event.x - pan_start_x
    dy = event.y - pan_start_y
    annotated_canvas.move(canvas_img_id, dx, dy)
    pan_start_x = event.x
    pan_start_y = event.y

app.mainloop()