import cv2
import numpy as np
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
from ultralytics import YOLO
import datetime  # For timestamp saving
from sklearn.linear_model import RANSACRegressor
from scipy.stats import iqr

# === Load YOLO Models ===
# model_lens = YOLO("200x_lens.pt")       # For lens/circles
model_lens = YOLO("circle.pt")       # For lens/circles
# model_rectangle = YOLO("outer_rect.pt")  # For rectangles (out - segmentation model)
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

PIXEL_SCALES = {"40x": 387, "80x": 776, "200x": 1940}

# === Annotation Colors (BGR) ===
COLOR_OK = (0, 255, 0)           # Green – within tolerance
COLOR_FAIL = (0, 0, 255)        # Red – out of tolerance / defect
COLOR_OK_DARK = (0, 128, 0)     # Dark green – diameter text OK
COLOR_LABEL = (0, 0, 0)         # Black – lens label text
COLOR_OUTER_RECT = (0, 200, 0)  # Green – outer rectangle edges
COLOR_INNER_RECT = (200, 0, 180)  # Magenta – inner rectangle edges

# === Annotation Parameters Per Zoom / Mode ===
FONT_SCALE_200X = 2.0
THICKNESS_200X = 4
FONT_SCALE_40X = 0.4
THICKNESS_40X = 1
FONT_SCALE_RECT = 0.6
THICKNESS_RECT = 2
THICKNESS_RECT_POLY = 3
FONT_SCALE_DEFECT = 0.5
THICKNESS_DEFECT = 2

# === Overlay / Threshold ===
DEFECT_OVERLAY_ALPHA = 0.3
MASK_THRESHOLD = 0.5

# === Rectangle Detection Profiles for 40x ===
PROFILES = {
    "blue": {"outer": (4.0, 1.5, 2.0), "inner": (3.0, 1.5, 2.5), "confirm": 12, "deep_scan": 12},
    "dark_dark": {"outer": (30.0, 2.5, 2.0), "inner": (12.0, 4.0, 1.2), "confirm": 5, "deep_scan": 0},
    # "dark_dark": {"outer": (40.0, 5.0, 1.8), "inner": (12.0, 4.0, 1.2), "confirm": 5, "deep_scan": 0},
    "dark_light": {"outer": (3.0, 1.0, 1.5), "inner": (40.0, 12.0, 5.0), "confirm": 5, "deep_scan": 3},
    # "dark_light": {"outer": (40.0, 5.0, 1.5), "inner": (25.0, 2.0, 1.8), "confirm": 5, "deep_scan": 3},
    "grey_light": {"outer": (2.0, 1.2, 2.8), "inner": (1.2, 1.5, 2.0), "confirm": 5, "deep_scan": 3},
    "grey_dark": {"outer": (2.5, 1.2, 2.0), "inner": (2.0, 2.0, 2.0), "confirm": 5, "deep_scan": 6},
    "yellow_dark": {"outer": (4.0, 1.5, 2.0), "inner": (15.0, 2.5, 2.0), "confirm": 6, "deep_scan": 3},
    "yellow_light": {"outer": (4.0, 1.0, 2.0), "inner": (8.0, 2.5, 2.0), "confirm": 4, "deep_scan": 3}
}

def detect_profile(image):
    """Detect the image profile based on color characteristics for 40x rectangle detection."""
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Global Metrics
    avg_sat = np.mean(hsv[:, :, 1])
    avg_val = np.mean(hsv[:, :, 2])

    # Contrast Score: Standard Deviation of grayscale intensities
    contrast_score = np.std(gray)

    # 1. Check for Yellow First (Hue 15-35)
    yellow_mask = cv2.inRange(hsv, np.array([15, 50, 50]), np.array([35, 255, 255]))
    yellow_pct = np.count_nonzero(yellow_mask) / yellow_mask.size
    if yellow_pct > 0.05:
        if avg_val < 132 or avg_sat > 78:
            print(f"Yellow Dark - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
            return "yellow_dark"
        else:
            print(f"Yellow Light - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
            return "yellow_light"

    # 2. Identify the 'Blue' Profile (Strong Saturation)
    if avg_sat > 50:
        print(f"Blue Profile - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
        return "blue"

    # 3. Differentiate Dark Dark, Dark Light vs Grey/Yellow using Contrast and Avg Value
    if contrast_score >= 28.0:
        # Use Avg Value to distinguish dark_dark from dark_light based on common ranges
        # Dark_dark typical Avg_Val: 98-140
        # Dark_light typical Avg_Val: 197-202
        if avg_val > 170: 
            print(f"Dark Light Profile - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
            return "dark_light"
        else:
            print(f"Dark Dark Profile - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
            return "dark_dark"

    # 4. Differentiate Grey Light vs Grey Dark using Brightness
    if avg_val >= 160:
        print(f"Grey Light - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
        return "grey_light"
    else:
        print(f"Grey Dark - Avg Saturation: {avg_sat:.2f}, Avg Value: {avg_val:.2f}, Contrast Score: {contrast_score:.2f}")
        return "grey_dark"

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

def _load_logo_image(path, max_w=250, max_h=140):
    try:
        img = Image.open(path)
        img.thumbnail((max_w, max_h))
        return ctk.CTkImage(light_image=img, size=(img.width, img.height))
    except Exception:
        return None

# === Detection Function (Measurement & Defect) ===
def detect_rectangles_40x(image, pixel_scale):
    """
    Detect outer and inner rectangles in 40x images using edge detection and line fitting.
    Returns annotated image and measurement results.
    """
    # Detect profile
    active_case = detect_profile(image)
    p = PROFILES[active_case]
    
    # Preprocessing based on profile
    if "yellow" in active_case:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lower_yellow = np.array([15, 45, 45])
        upper_yellow = np.array([35, 255, 255])
        gray = cv2.inRange(hsv, lower_yellow, upper_yellow)
        kernel = np.ones((3,3), np.uint8)
        gray = cv2.morphologyEx(gray, cv2.MORPH_OPEN, kernel)
        gray = cv2.GaussianBlur(gray, (5, 5), 0)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    
    h, w = gray.shape
    output = image.copy()
    
    # Dynamic parameters
    c_size, c_mid = 15, 10
    corners = [gray[0:c_size, 0:c_size], gray[0:c_size, -c_size:],
               gray[-c_size:, 0:c_size], gray[-c_size:, -c_size:]]
    bg_avg_outer = np.median([np.mean(c) for c in corners])
    bg_std_outer = np.median([iqr(c) for c in corners])
    
    center_roi = gray[h//2-c_mid : h//2+c_mid, w//2-c_mid : w//2+c_mid]
    bg_avg_inner = np.mean(center_roi)
    bg_std_inner = iqr(center_roi)
    
    def calc_thresh(std, params_tuple):
        mini, sens, scale = params_tuple
        return max(mini, std * sens) * scale
    
    thresh_outer = calc_thresh(bg_std_outer, p["outer"])
    thresh_inner = calc_thresh(bg_std_inner, p["inner"])
    confirm_pix = p["confirm"]
    deep_scan_val = p["deep_scan"]
    
    # Scan zone definitions
    outer_v_range_top = np.concatenate([np.arange(int(w*0.05), int(w*0.25)), np.arange(int(w*0.75), int(w*0.95))])
    outer_v_range = np.concatenate([np.arange(int(w*0.05), int(w*0.45)), np.arange(int(w*0.55), int(w*0.95))])
    outer_h_range = np.concatenate([np.arange(int(h*0.05), int(h*0.45)), np.arange(int(h*0.55), int(h*0.95))])
    inner_v_range = np.arange(int(w*0.2), int(w*0.8))
    inner_h_range = np.arange(int(h*0.4), int(h*0.9))
    
    def get_raw_points(scan_range, thresh, axis='y', mode='inward', direction='low', ref_bg=128, dot_color=(255,0,0)):
        rv1, rv2 = [], []
        mid = (h // 2) if axis == 'x' else (w // 2)
        intensity_ceiling = 250
        
        for coord in scan_range:
            line_data = gray[:, coord].astype(float) if axis == 'y' else gray[coord, :].astype(float)
            
            if mode == 'inward':
                indices = range(len(line_data)-1-confirm_pix, mid, -1) if direction == 'high' else range(confirm_pix, mid)
            else:
                indices = range(mid + 25, len(line_data)-confirm_pix) if direction == 'high' else range(mid - 25, confirm_pix, -1)
            
            for i in indices:
                pixel_val = line_data[i]
                if abs(pixel_val - ref_bg) > thresh and pixel_val < intensity_ceiling:
                    if deep_scan_val > 0:
                        step = 1 if (mode == 'inward' and direction == 'low') or (mode == 'outward' and direction == 'high') else -1
                        win_indices = [i + (s * step) for s in range(deep_scan_val)]
                        win_indices = [idx for idx in win_indices if 0 <= idx < len(line_data)]
                        win_values = [abs(line_data[idx] - ref_bg) if line_data[idx] < intensity_ceiling else 0 for idx in win_indices]
                        peak_local_idx = np.argmax(win_values)
                        edge_idx = win_indices[peak_local_idx]
                    else:
                        edge_idx = i
                    
                    rv1.append(coord)
                    rv2.append(edge_idx)
                    px, py = (coord, edge_idx) if axis == 'y' else (edge_idx, coord)
                    cv2.circle(output, (px, py), 1, dot_color, -1)
                    break
        return [np.array(rv1), np.array(rv2)]
    
    def fit_line_ransac(points):
        if len(points[0]) < 5: return None
        indep = points[0].reshape(-1, 1)
        dep = points[1]
        try:
            ransac = RANSACRegressor(residual_threshold=3.0)
            ransac.fit(indep, dep)
            return [ransac.estimator_.coef_[0], ransac.estimator_.intercept_]
        except Exception:
            return None
    
    def draw_infinite_line(line, is_vertical, color):
        m, c = line
        if is_vertical:
            p1, p2 = (int(m*0 + c), 0), (int(m*h + c), h)
        else:
            p1, p2 = (0, int(m*0 + c)), (w, int(m*w + c))
        cv2.line(output, p1, p2, color, 1)
    
    def intersect(line_h, line_v):
        mh, ch = line_h
        mv, cv = line_v
        denom = (1.0 - mv * mh)
        if abs(denom) < 1e-7: denom = 1e-7
        x_c = (mv * ch + cv) / denom
        y_c = mh * x_c + ch
        return [int(x_c), int(y_c)]
    
    # Rectangle configurations
    rect_configs = [
        {
            'name': 'Outer',
            'color': COLOR_OUTER_RECT,
            'edges': {
                'top':    get_raw_points(outer_v_range_top, thresh_outer, 'y', 'inward', 'low',  bg_avg_outer, COLOR_OUTER_RECT),
                'bottom': get_raw_points(outer_v_range,     thresh_outer, 'y', 'inward', 'high', bg_avg_outer, COLOR_OUTER_RECT),
                'left':   get_raw_points(outer_h_range,     thresh_outer, 'x', 'inward', 'low',  bg_avg_outer, COLOR_OUTER_RECT),
                'right':  get_raw_points(outer_h_range,     thresh_outer, 'x', 'inward', 'high', bg_avg_outer, COLOR_OUTER_RECT)
            }
        },
        {
            'name': 'Inner',
            'color': COLOR_INNER_RECT,
            'edges': {
                'top':    get_raw_points(inner_v_range, thresh_inner, 'y', 'outward', 'low',  bg_avg_inner, COLOR_INNER_RECT),
                'bottom': get_raw_points(inner_v_range, thresh_inner, 'y', 'outward', 'high', bg_avg_inner, COLOR_INNER_RECT),
                'left':   get_raw_points(inner_h_range, thresh_inner, 'x', 'outward', 'low',  bg_avg_inner, COLOR_INNER_RECT),
                'right':  get_raw_points(inner_h_range, thresh_inner, 'x', 'outward', 'high', bg_avg_inner, COLOR_INNER_RECT)
            }
        }
    ]
    
    results = []
    
    for cfg in rect_configs:
        lines = {k: fit_line_ransac(cfg['edges'][k]) for k in cfg['edges']}
        
        if all(L is not None for L in lines.values()):
            for k in lines:
                draw_infinite_line(lines[k], k in ['left', 'right'], cfg['color'])
            
            pts = [
                intersect(lines['top'],    lines['left']),
                intersect(lines['top'],    lines['right']),
                intersect(lines['bottom'], lines['right']),
                intersect(lines['bottom'], lines['left'])
            ]
            
            box = np.array(pts, np.int32)
            
            width = np.linalg.norm(np.array(pts[0]) - np.array(pts[1]))
            height = np.linalg.norm(np.array(pts[0]) - np.array(pts[3]))
            
            # Convert to mm
            width_mm = width / pixel_scale
            height_mm = height / pixel_scale
            
            # Determine length and breadth (length is the longer side)
            length_mm = max(width_mm, height_mm)
            breadth_mm = min(width_mm, height_mm)
            
            # Check tolerance based on rectangle type
            if cfg['name'] == 'Outer':
                length_tol = abs(length_mm - OUT_RECT_LENGTH) <= RECT_TOL
                breadth_tol = abs(breadth_mm - OUT_RECT_BREADTH) <= RECT_TOL
            else:  # Inner
                length_tol = abs(length_mm - IN_RECT_LENGTH) <= RECT_TOL
                breadth_tol = abs(breadth_mm - IN_RECT_BREADTH) <= RECT_TOL
            
            in_tol = length_tol and breadth_tol
            
            # Use red if out of tolerance, green if OK
            color = COLOR_FAIL if not in_tol else COLOR_OK
            
            text_pos = (pts[0][0], pts[0][1] - 10 if cfg['name'] == 'Outer' else pts[0][1] + 30)
            label = f"{cfg['name']}: {length_mm:.3f}x{breadth_mm:.3f}mm"
            
            # Draw rectangle and text with conditional color
            cv2.polylines(output, [box.reshape((-1, 1, 2))], isClosed=True, color=color, thickness=THICKNESS_RECT_POLY)
            cv2.putText(output, label, text_pos, cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_RECT, color, THICKNESS_RECT)
            
            results.append({
                'name': cfg['name'],
                'length_mm': length_mm,
                'breadth_mm': breadth_mm,
                'in_tolerance': in_tol,
                'length_ok': length_tol,
                'breadth_ok': breadth_tol
            })
        else:
            results.append({
                'name': cfg['name'],
                'error': 'Could not detect rectangle boundaries'
            })
    
    return output, results

def detect_and_annotate_lenses(image, annotated, pixel_scale, label_filter, font_scale=FONT_SCALE_40X, thickness=THICKNESS_40X):
    """
    Detect lenses using YOLO model, annotate the image, and return results.
    
    Args:
        image: Original BGR image for model inference.
        annotated: Image to draw annotations on (may already have rectangle annotations).
        pixel_scale: Pixels per mm for the current zoom level.
        label_filter: List of class name strings to accept (e.g. ["lens", "circle"]).
        font_scale: OpenCV font scale for annotations.
        thickness: OpenCV text/line thickness for annotations.
    
    Returns:
        (annotated, results_text): Annotated image and list of result strings.
    """
    res_lens = model_lens(image)[0]
    results_text = []
    circles = []

    for box, cls in zip(res_lens.boxes.xyxy.cpu().numpy(), res_lens.boxes.cls.cpu().numpy()):
        x1, y1, x2, y2 = map(int, box)
        label = res_lens.names[int(cls)].lower()
        if label not in label_filter:
            continue
        w, h = x2 - x1, y2 - y1
        center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
        diameter_px = (w + h) / 2
        diameter_mm = diameter_px / pixel_scale
        in_tol = abs(diameter_mm - TARGET_LENS_DIAMETER) <= LENS_TOL
        color = COLOR_OK if in_tol else COLOR_FAIL
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        circles.append((center_x, center_y, diameter_mm, in_tol, (x1, y1, x2, y2)))

    circles.sort(key=lambda c: c[0])

    # Label offset scales relative to font_scale
    label_offset_x = int(30 * font_scale / FONT_SCALE_40X)
    label_offset_y = int(20 * font_scale / FONT_SCALE_40X)
    diam_offset_y = int(15 * font_scale / FONT_SCALE_40X)

    for idx, (cx, cy, d_mm, in_tol, (x1, y1, x2, y2)) in enumerate(circles, start=1):
        label_text = f"Lens {idx}"
        lx = int((x1 + x2) // 2) - label_offset_x
        ly = int(y1) - label_offset_y
        cv2.putText(annotated, label_text, (lx, ly),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, COLOR_LABEL, thickness)

        diam_x = int(x1) + 3
        diam_y = int(y2) + diam_offset_y
        diam_text = f"Dia: {d_mm:.3f}mm"
        diam_color = COLOR_FAIL if not in_tol else COLOR_OK_DARK
        cv2.putText(annotated, diam_text, (diam_x, diam_y),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, diam_color, thickness)
        results_text.append(f"Lens {idx} Diameter: {d_mm:.3f}mm {'OK' if in_tol else 'Out of Tolerance'}")

    # Lens-to-lens distances
    for i in range(len(circles) - 1):
        c1 = circles[i]
        c2 = circles[i + 1]
        dist_mm = abs(c2[0] - c1[0]) / pixel_scale
        results_text.append(f"Center-to-center distance Lens {i+1} to Lens {i+2}: {dist_mm:.3f}mm")

    if len(circles) > 1:
        total_dist_mm = abs(circles[-1][0] - circles[0][0]) / pixel_scale
        results_text.append(f"Center-to-center distance Lens 1 to Lens {len(circles)}: {total_dist_mm:.3f}mm")

    if not circles:
        results_text.append("No lenses detected.")

    return annotated, results_text

class LensQCApp:
    def __init__(self):
        ctk.set_appearance_mode("light")
        ctk.set_default_color_theme("blue")
        self.app = ctk.CTk()

        # --- State ---
        self.uploaded_image = None
        self.uploaded_image_path = None
        self.annotated_image = None
        self.measure_type = "rectangle"
        self.image_preview_tk = None
        self.annotated_image_tk = None
        self.results_lines = []

        self.tabs_created = False
        self.tabs = None
        self.results_textbox = None

        self.zoom_level = 1.0
        self.min_zoom = 0.1
        self.max_zoom = 5.0
        self.pan_start_x = 0
        self.pan_start_y = 0
        self.offset_x = 0
        self.offset_y = 0
        self.annotated_canvas = None
        self.canvas_img_id = None

        self.upload_tab_upload_btn = None
        self.upload_tab_run_detection_btn = None
        self.preview_img_label = None

        self._resize_after_id = None
        self._last_window_size = (0, 0)

        # --- Build UI ---
        self._set_responsive_geometry()
        self.app.title("Lens Quality Check")
        self.app.configure(fg_color="#e0e0e0")

        self._build_sidebar()
        self._build_main_frame()
        self._create_tabs()
        self.tabs_created = True

        self.app.bind("<Configure>", self._on_window_resize)
        self.app.after(100, self._refresh_responsive_elements)

    def run(self):
        self.app.mainloop()

    # === Window Geometry ===

    def _set_responsive_geometry(self):
        screen_width = self.app.winfo_screenwidth()
        screen_height = self.app.winfo_screenheight()
        window_width = min(max(int(screen_width * 0.8), 1200), 1600)
        window_height = min(max(int(screen_height * 0.8), 800), 1000)
        x = (screen_width - window_width) // 2
        y = (screen_height - window_height) // 2
        self.app.geometry(f"{window_width}x{window_height}+{x}+{y}")
        self.app.minsize(1000, 600)

    def _get_responsive_sidebar_width(self):
        window_width = self.app.winfo_width()
        if window_width < 1200:
            return 250
        elif window_width < 1400:
            return 280
        else:
            return 320

    def _get_responsive_font_size(self, base_size, scale_factor=1.0):
        window_width = self.app.winfo_width()
        if window_width < 1200:
            return int(base_size * 0.9 * scale_factor)
        elif window_width > 1500:
            return int(base_size * 1.1 * scale_factor)
        else:
            return int(base_size * scale_factor)

    # === Sidebar ===

    def _build_sidebar(self):
        self.sidebar = ctk.CTkFrame(self.app, width=280, fg_color="#eaeaea", corner_radius=15)
        self.sidebar.pack(side="left", fill="y", padx=(20, 10), pady=20)
        self.sidebar.pack_propagate(False)

        self.project_label = ctk.CTkLabel(self.sidebar, text="Lens Quality Check",
                                          font=("Arial", 26, "bold"), wraplength=1000)
        self.project_label.pack(pady=15)

        logo1_img = _load_logo_image("lsp-logo.png")
        logo2_img = _load_logo_image("sp-logo1.png")

        if logo1_img:
            self._logo1_img = logo1_img  # prevent garbage collection
            logo1_label = ctk.CTkLabel(self.sidebar, image=logo1_img, text="")
            logo1_label.pack(padx=20, pady=(16, 10))

        collab_label = ctk.CTkLabel(self.sidebar, text="in collaboration with", font=("Arial", 18))
        collab_label.pack(pady=(5, 5))

        if logo2_img:
            self._logo2_img = logo2_img
            logo2_label = ctk.CTkLabel(self.sidebar, image=logo2_img, text="")
            logo2_label.pack(padx=20, pady=(10, 28))

        # Mode dropdown
        self.mode_var = ctk.StringVar(value="measurement")
        mode_label = ctk.CTkLabel(self.sidebar, text="Mode:", font=("Arial", 18, "bold"))
        mode_label.pack(pady=(0, 5), padx=70, anchor="w")
        ctk.CTkOptionMenu(self.sidebar, variable=self.mode_var,
                          values=["measurement", "defect"],
                          width=200, height=40, font=("Arial", 18),
                          dropdown_font=("Arial", 17), corner_radius=10).pack(pady=(0, 20))

        # Zoom dropdown
        self.zoom_var = ctk.StringVar(value="40x")
        ctk.CTkOptionMenu(self.sidebar, variable=self.zoom_var,
                          values=["40x", "80x", "200x"],
                          command=lambda _: self._update_measurement_type(),
                          width=200, height=40, font=("Arial", 18),
                          dropdown_font=("Arial", 17), corner_radius=10).pack(pady=(0, 20))

        # Status label
        self.status_label = ctk.CTkLabel(self.sidebar,
                                         text="Current Mode: Rectangle Measurement (40x)",
                                         wraplength=200, justify="center",
                                         font=("Arial", 18), corner_radius=10)
        self.status_label.pack(pady=(10, 20))

        # Variable traces
        self.mode_var.trace_add("write", self._update_status_label)
        self.zoom_var.trace_add("write", self._update_status_label)
        self.zoom_var.trace_add("write", lambda *args: self._update_measurement_type())

        # Pixel scale entries
        self.pixel_scale_40x_var = ctk.StringVar(value=str(PIXEL_SCALES["40x"]))
        self.pixel_scale_80x_var = ctk.StringVar(value=str(PIXEL_SCALES["80x"]))
        self.pixel_scale_200x_var = ctk.StringVar(value=str(PIXEL_SCALES["200x"]))

        scale_label = ctk.CTkLabel(self.sidebar, text="Pixel to mm Scale:", font=("Arial", 18, "bold"))
        scale_label.pack(pady=(10, 5), padx=70, anchor="w")

        for label_text, var in [("40x:", self.pixel_scale_40x_var),
                                ("80x:", self.pixel_scale_80x_var),
                                ("200x:", self.pixel_scale_200x_var)]:
            row = ctk.CTkFrame(self.sidebar, fg_color="transparent")
            row.pack(pady=(0, 10), padx=0, fill="x")
            ctk.CTkLabel(row, text=label_text, font=("Arial", 18),
                         width=60, anchor="w").pack(side="left", padx=(65, 0))
            ctk.CTkEntry(row, textvariable=var, width=140, height=40,
                         font=("Arial", 18), corner_radius=10).pack(side="right", padx=(0, 60))

        ctk.CTkButton(self.sidebar, text="Confirm Scale",
                      command=self._update_pixel_scale_from_entry,
                      width=200, height=40, font=("Arial", 18),
                      corner_radius=10).pack(pady=(0, 20))

        self._update_status_label()

    # === Main Frame ===

    def _build_main_frame(self):
        self.main_frame = ctk.CTkFrame(self.app, fg_color="#eaeaea", corner_radius=15)
        self.main_frame.pack(side="left", fill="both", expand=True, padx=(10, 20), pady=20)

        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="#eaeaea")
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # === Status / Measurement Helpers ===

    def _update_status_label(self, *args):
        mode = self.mode_var.get()
        zoom = self.zoom_var.get()
        if mode == "defect":
            status_text = "Current Mode:\nDefect Detection"
        else:
            if zoom in ("40x", "80x"):
                mt = "Rectangle & Lens"
            else:
                mt = "Lens"
            status_text = f"Current Mode:\n{mt} Measurement ({zoom})"
        self.status_label.configure(text=status_text)

    def _update_measurement_type(self):
        if self.zoom_var.get() in ("40x", "80x"):
            self.measure_type = "rectangle"
        else:
            self.measure_type = "lens"

    def _update_pixel_scale_from_entry(self):
        try:
            val_40x = float(self.pixel_scale_40x_var.get())
            PIXEL_SCALES["40x"] = val_40x
            val_80x = float(self.pixel_scale_80x_var.get())
            PIXEL_SCALES["80x"] = val_80x
            val_200x = float(self.pixel_scale_200x_var.get())
            PIXEL_SCALES["200x"] = val_200x
            messagebox.showinfo("Success",
                                f"Pixel scales updated:\n40x: {val_40x}\n80x: {val_80x}\n200x: {val_200x}")
        except ValueError:
            messagebox.showerror("Error", "Please enter valid numeric values for pixel scales.")

    def _update_header_font(self):
        font_size = self._get_responsive_font_size(26)
        self.project_label.configure(font=("Arial", font_size, "bold"))

    # === Upload & Preview ===

    def upload_image(self):
        path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg *.jpeg *.png *.bmp")])
        if not path:
            return
        img = cv2.imread(path)
        if img is None:
            messagebox.showerror("Error", "Cannot load image.")
            return
        self.uploaded_image = img
        self.uploaded_image_path = path
        self.zoom_level = 1.0
        self._show_preview_image(img)
        if self.upload_tab_run_detection_btn:
            self.upload_tab_run_detection_btn.configure(state="normal")

    def _show_preview_image(self, img):
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        aspect_ratio = img_pil.width / img_pil.height

        window_width = self.app.winfo_width()
        window_height = self.app.winfo_height()
        available_width = max(window_width - 400, 400)
        available_height = max(window_height - 180, 500)

        scale_w = (available_width * 0.95) / img_pil.width
        scale_h = (available_height * 0.95) / img_pil.height
        scale = min(scale_w, scale_h)

        target_width = int(img_pil.width * scale)
        target_height = int(img_pil.height * scale)

        if target_width > available_width:
            target_width = int(available_width * 0.95)
            target_height = int(target_width / aspect_ratio)
        if target_height > available_height:
            target_height = int(available_height * 0.95)
            target_width = int(target_height * aspect_ratio)

        img_pil = img_pil.resize((target_width, target_height), Image.LANCZOS)
        self.image_preview_tk = ctk.CTkImage(light_image=img_pil, size=(target_width, target_height))

        if self.preview_img_label:
            self.preview_img_label.configure(image=self.image_preview_tk, text="")

    # === Detection ===

    def run_detection(self):
        if self.uploaded_image is None:
            messagebox.showwarning("Warning", "Please upload an image first.")
            return

        mode = self.mode_var.get()
        pixel_scale = PIXEL_SCALES.get(self.zoom_var.get(), 380)

        annotated = self.uploaded_image.copy()
        results_text = []

        # --- Measurement Logic ---
        if mode == "measurement":
            if self.measure_type == "lens":
                # 200x lens measurement
                annotated, lens_results = detect_and_annotate_lenses(
                    self.uploaded_image, annotated, pixel_scale,
                    label_filter=["lens", "circle"], font_scale=FONT_SCALE_200X, thickness=THICKNESS_200X)
                results_text.extend(lens_results)

            # --- Rectangle Measurement Logic ---
            elif self.measure_type == "rectangle":
                # Use edge detection method for rectangles
                annotated, rect_results = detect_rectangles_40x(self.uploaded_image, pixel_scale)

                results_text.append("=== Rectangle Measurements ===")
                for result in rect_results:
                    if 'error' in result:
                        results_text.append(f"{result['name']} Rectangle: {result['error']}")
                    else:
                        status = "OK" if result['in_tolerance'] else "Out of Tolerance"
                        results_text.append(f"{result['name']} Rectangle: {result['length_mm']:.3f}mm x {result['breadth_mm']:.3f}mm - {status}")
                        if not result['length_ok']:
                            results_text.append(f"  Length out of tolerance: {result['length_mm']:.3f}mm")
                        if not result['breadth_ok']:
                            results_text.append(f"  Breadth out of tolerance: {result['breadth_mm']:.3f}mm")

                # --- Also detect lenses using circle model ---
                results_text.append("")
                results_text.append("=== Lens Measurements ===")
                annotated, lens_results = detect_and_annotate_lenses(
                    self.uploaded_image, annotated, pixel_scale,
                    label_filter=["circle"], font_scale=FONT_SCALE_40X, thickness=THICKNESS_40X)
                results_text.extend(lens_results)

        # --- Defect Logic (Segmentation) ---
        elif mode == "defect":
            res_def = model_defects(self.uploaded_image)[0]
            img_height, img_width = self.uploaded_image.shape[:2]
            defect_count = 0

            for i, (box, cls, conf) in enumerate(zip(
                    res_def.boxes.xyxy.cpu().numpy(),
                    res_def.boxes.cls.cpu().numpy(),
                    res_def.boxes.conf.cpu().numpy())):
                x1, y1, x2, y2 = map(int, box)
                class_name = res_def.names[int(cls)]
                defect_count += 1

                # Draw segmentation mask if available
                if hasattr(res_def, 'masks') and res_def.masks is not None and i < len(res_def.masks.data):
                    mask = res_def.masks.data[i].cpu().numpy()
                    mask_h, mask_w = mask.shape
                    if mask_h != img_height or mask_w != img_width:
                        mask = cv2.resize(mask, (img_width, img_height))
                    binary_mask = (mask > MASK_THRESHOLD).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(binary_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                    # Semi-transparent red overlay for the mask region
                    overlay = annotated.copy()
                    cv2.fillPoly(overlay, contours, COLOR_FAIL)
                    cv2.addWeighted(overlay, DEFECT_OVERLAY_ALPHA, annotated, 1 - DEFECT_OVERLAY_ALPHA, 0, annotated)

                    # Draw contour outline
                    cv2.drawContours(annotated, contours, -1, COLOR_FAIL, THICKNESS_DEFECT)
                else:
                    # Fallback to bounding box if no mask
                    cv2.rectangle(annotated, (x1, y1), (x2, y2), COLOR_FAIL, THICKNESS_DEFECT)

                # Label with class name and confidence
                label = f"{class_name} {conf:.2f}"
                cv2.putText(annotated, label, (x1, max(20, y1 - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE_DEFECT, COLOR_FAIL, THICKNESS_DEFECT)
                results_text.append(f"Defect {defect_count}: {class_name} (conf={conf:.2f}) at ({x1},{y1}) ({x2},{y2})")

            if defect_count == 0:
                results_text.append("No defects detected.")
            else:
                results_text.insert(0, f"Total defects found: {defect_count}")

        self.annotated_image = annotated
        self.results_lines = results_text

        # --- Auto-save annotated image ---
        if self.annotated_image is not None:
            self._save_results(mode)

        self._update_preview_tab()
        self._update_annotated_tab()
        self._update_results_tab()

    _SAVE_DIRS = {
        ("measurement", "40x"):  "measurements/40x",
        ("measurement", "80x"):  "measurements/80x",
        ("measurement", "200x"): "measurements/200x",
        ("defect", None):        "defects",
    }

    def _save_results(self, mode):
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(self.uploaded_image_path))[0]

        zoom = self.zoom_var.get() if mode == "measurement" else None
        sub_dir = self._SAVE_DIRS.get((mode, zoom), "")
        save_dir = os.path.join(HISTORY_DIR, sub_dir)
        os.makedirs(save_dir, exist_ok=True)

        save_path = os.path.join(save_dir, f"{base_name}_annotated_{timestamp}.png")
        txt_path = os.path.join(save_dir, f"{base_name}_annotated_{timestamp}.txt")
        cv2.imwrite(save_path, self.annotated_image)
        with open(txt_path, "w") as f:
            for line in self.results_lines:
                f.write(line + "\n")

    # === Tabs ===

    def _create_tabs(self):
        window_width = self.app.winfo_width()
        if window_width < 1200:
            pad_x, pad_y = 10, 10
        elif window_width > 1500:
            pad_x, pad_y = 30, 25
        else:
            pad_x, pad_y = 20, 20

        tab_container = ctk.CTkFrame(self.content_frame, fg_color="#eaeaea", corner_radius=10)
        tab_container.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)

        self.tabs = ctk.CTkTabview(tab_container,
                                   fg_color="#eaeaea",
                                   segmented_button_fg_color="#e0e0e0",
                                   segmented_button_selected_color="#3b8ed0",
                                   segmented_button_selected_hover_color="#36719f",
                                   height=64)
        self.tabs.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)

        responsive_font_size = self._get_responsive_font_size(18)
        self.tabs._segmented_button.configure(font=("Arial", responsive_font_size, "bold"), height=48)
        self.tabs._segmented_button.configure(corner_radius=10)
        self.tabs._segmented_button.grid_configure(padx=pad_x, pady=pad_y)

        self.tabs.add("Upload New")
        self.tabs.add("Annotated Image")
        self.tabs.add("Results")

        self.tabs._segmented_button.grid_columnconfigure((0, 1, 2), weight=1)
        self.tabs._segmented_button.configure(
            font=("Arial", responsive_font_size, "bold"),
            height=48,
            corner_radius=10
        )

        for tab_name in ["Upload New", "Annotated Image", "Results"]:
            tab = self.tabs.tab(tab_name)
            tab.configure(fg_color="#eaeaea")
            inner_frame = ctk.CTkFrame(tab, fg_color="#eaeaea")
            inner_frame.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)

        # Upload tab
        upload_tab = self.tabs.tab("Upload New")
        upload_content = upload_tab.winfo_children()[0]

        button_width = max(min(int(self.app.winfo_width() * 0.12), 200), 120)
        button_font_size = self._get_responsive_font_size(18)

        button_frame = ctk.CTkFrame(upload_content, fg_color="transparent")
        button_frame.pack(pady=(0, 10))

        self.upload_tab_upload_btn = ctk.CTkButton(button_frame,
                                                   text="Upload Image",
                                                   command=self.upload_image,
                                                   width=button_width,
                                                   height=48,
                                                   corner_radius=10,
                                                   font=("Arial", button_font_size))
        self.upload_tab_upload_btn.pack(side="left", padx=(0, 10))

        self.upload_tab_run_detection_btn = ctk.CTkButton(button_frame,
                                                          text="Run Detection",
                                                          command=self.run_detection,
                                                          width=button_width,
                                                          height=48,
                                                          corner_radius=10,
                                                          font=("Arial", button_font_size))
        self.upload_tab_run_detection_btn.pack(side="left", padx=(10, 0))

        preview_frame = ctk.CTkFrame(upload_content, fg_color="#eaeaea", corner_radius=10)
        preview_frame.pack(fill="both", expand=True, padx=pad_x, pady=(10, 0))

        self.preview_img_label = ctk.CTkLabel(preview_frame,
                                              text="No image uploaded",
                                              fg_color="#eaeaea",
                                              corner_radius=10)
        self.preview_img_label.pack(expand=True, fill="both", padx=10, pady=10)

        # Annotated Image tab
        annotated_tab = self.tabs.tab("Annotated Image")
        annotated_content = annotated_tab.winfo_children()[0]

        self.annotated_canvas = ctk.CTkCanvas(annotated_content,
                                              bg="gray90",
                                              highlightthickness=0)
        self.annotated_canvas.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)
        self.canvas_img_id = None

        self.annotated_canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.annotated_canvas.bind("<Button-4>", self._on_mousewheel)
        self.annotated_canvas.bind("<Button-5>", self._on_mousewheel)
        self.annotated_canvas.bind("<ButtonPress-1>", self._on_pan_start)
        self.annotated_canvas.bind("<B1-Motion>", self._on_pan_move)

        # Results tab
        results_tab = self.tabs.tab("Results")
        results_content = results_tab.winfo_children()[0]

        results_font_size = self._get_responsive_font_size(14)
        self.results_textbox = ctk.CTkTextbox(results_content,
                                              wrap="word",
                                              fg_color="white",
                                              font=("Arial", results_font_size),
                                              corner_radius=10)
        self.results_textbox.pack(fill="both", expand=True, padx=pad_x, pady=pad_y)
        self.results_textbox.configure(state="disabled")

        self._update_preview_tab()
        self._update_annotated_tab()
        self._update_results_tab()

    # === Tab Updates ===

    def _update_preview_tab(self):
        if self.uploaded_image is None:
            return
        self._show_preview_image(self.uploaded_image)

    def _update_annotated_tab(self):
        if self.annotated_image is None:
            return
        img_rgb = cv2.cvtColor(self.annotated_image, cv2.COLOR_BGR2RGB)
        img_pil = Image.fromarray(img_rgb)
        new_size = (int(img_pil.width * self.zoom_level), int(img_pil.height * self.zoom_level))
        resized_img = img_pil.resize(new_size, Image.LANCZOS)
        self.annotated_image_tk = ctk.CTkImage(light_image=resized_img, size=new_size)
        self.annotated_canvas.delete("all")
        canvas_image = ImageTk.PhotoImage(resized_img)
        self.canvas_img_id = self.annotated_canvas.create_image(
            self.offset_x, self.offset_y, anchor="nw", image=canvas_image)
        self.annotated_canvas.image = canvas_image  # Keep a reference
        self.annotated_canvas.config(scrollregion=(0, 0, new_size[0], new_size[1]))

    def _update_results_tab(self):
        self.results_textbox.configure(state="normal")
        self.results_textbox.delete("0.0", "end")
        for line in self.results_lines:
            if "Out of Tolerance" in line:
                self.results_textbox.insert("end", line + "\n", ("red",))
            else:
                self.results_textbox.insert("end", line + "\n")
        self.results_textbox.tag_config("red", foreground="red")
        self.results_textbox.configure(state="disabled")

    # === Responsive ===

    def _refresh_responsive_elements(self):
        if not self.tabs_created:
            return

        self._update_header_font()

        if self.upload_tab_upload_btn:
            button_width = max(min(int(self.app.winfo_width() * 0.12), 200), 120)
            button_font_size = self._get_responsive_font_size(18)
            self.upload_tab_upload_btn.configure(width=button_width, font=("Arial", button_font_size))
            self.upload_tab_run_detection_btn.configure(width=button_width, font=("Arial", button_font_size))

        if self.results_textbox:
            results_font_size = self._get_responsive_font_size(14)
            self.results_textbox.configure(font=("Arial", results_font_size))

        if self.uploaded_image is not None:
            self._show_preview_image(self.uploaded_image)

    def _on_window_resize(self, event=None):
        if not event or event.widget != self.app:
            return
        new_size = (event.width, event.height)
        if new_size == self._last_window_size:
            return
        self._last_window_size = new_size
        if self._resize_after_id is not None:
            self.app.after_cancel(self._resize_after_id)
        self._resize_after_id = self.app.after(200, self._apply_resize)

    def _apply_resize(self):
        self._resize_after_id = None
        self.sidebar.configure(width=self._get_responsive_sidebar_width())
        self._refresh_responsive_elements()

    # === Zoom & Pan ===

    def _on_mousewheel(self, event):
        if self.annotated_image is None:
            return
        mouse_x = event.x
        mouse_y = event.y
        old_zoom = self.zoom_level
        if event.num == 4 or event.delta > 0:
            zoom_factor = 1.1
        elif event.num == 5 or event.delta < 0:
            zoom_factor = 0.9
        else:
            return
        new_zoom = self.zoom_level * zoom_factor
        if new_zoom < self.min_zoom:
            new_zoom = self.min_zoom
        elif new_zoom > self.max_zoom:
            new_zoom = self.max_zoom
        if abs(new_zoom - self.zoom_level) < 0.001:
            return
        self.offset_x = mouse_x - ((mouse_x - self.offset_x) * new_zoom / old_zoom)
        self.offset_y = mouse_y - ((mouse_y - self.offset_y) * new_zoom / old_zoom)
        self.zoom_level = new_zoom
        self._update_annotated_tab()

    def _on_pan_start(self, event):
        self.pan_start_x = event.x
        self.pan_start_y = event.y

    def _on_pan_move(self, event):
        if self.canvas_img_id is None:
            return
        dx = event.x - self.pan_start_x
        dy = event.y - self.pan_start_y
        self.offset_x += dx
        self.offset_y += dy
        self.annotated_canvas.move(self.canvas_img_id, dx, dy)
        self.pan_start_x = event.x
        self.pan_start_y = event.y


if __name__ == "__main__":
    app = LensQCApp()
    app.run()