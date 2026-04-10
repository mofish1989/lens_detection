import cv2
import numpy as np
from ultralytics import YOLO
from sklearn.linear_model import RANSACRegressor
from scipy.stats import iqr

from constants import (
    PROFILES, PIXEL_SCALES,
    OUT_RECT_LENGTH, OUT_RECT_BREADTH, IN_RECT_LENGTH, IN_RECT_BREADTH, RECT_TOL,
    TARGET_LENS_DIAMETER, LENS_TOL,
    COLOR_OK, COLOR_FAIL, COLOR_OK_DARK, COLOR_LABEL,
    COLOR_OUTER_RECT, COLOR_INNER_RECT,
    FONT_SCALE_200X, THICKNESS_200X, FONT_SCALE_40X, THICKNESS_40X,
    FONT_SCALE_RECT, THICKNESS_RECT, THICKNESS_RECT_POLY,
    FONT_SCALE_DEFECT, THICKNESS_DEFECT,
    DEFECT_OVERLAY_ALPHA, MASK_THRESHOLD,
)

# === Load YOLO Models ===
model_lens = YOLO("circle.pt")
model_defects = YOLO("defects.pt")


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
            return "yellow_dark"
        else:
            return "yellow_light"

    # 2. Identify the 'Blue' Profile (Strong Saturation)
    if avg_sat > 50:
        return "blue"

    # 3. Differentiate Dark Dark, Dark Light vs Grey/Yellow using Contrast and Avg Value
    if contrast_score >= 28.0:
        # Use Avg Value to distinguish dark_dark from dark_light based on common ranges
        # Dark_dark typical Avg_Val: 98-140
        # Dark_light typical Avg_Val: 197-202
        if avg_val > 170:
            return "dark_light"
        else:
            return "dark_dark"

    # 4. Differentiate Grey Light vs Grey Dark using Brightness
    if avg_val >= 160:
        return "grey_light"
    else:
        return "grey_dark"


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

    # Build outer background from non-black corner pixels to avoid black-border corruption
    all_corner_px = np.concatenate([c.flatten() for c in corners])
    non_black_px = all_corner_px[all_corner_px > 25]
    if len(non_black_px) >= 10:
        bg_avg_outer = float(np.median(non_black_px))
        bg_std_outer = float(np.std(non_black_px))
    else:
        # All corners are black — fall back to near-edge border strips
        border_px = max(25, int(min(h, w) * 0.03))
        strips = np.concatenate([
            gray[:border_px, :].flatten(),
            gray[-border_px:, :].flatten(),
            gray[:, :border_px].flatten(),
            gray[:, -border_px:].flatten()
        ])
        non_black_strips = strips[strips > 25]
        bg_avg_outer = float(np.median(non_black_strips)) if len(non_black_strips) >= 20 else 128.0
        bg_std_outer = float(np.std(non_black_strips)) if len(non_black_strips) >= 20 else 30.0
    bg_std_outer = float(np.clip(bg_std_outer, 0, 40))  # cap to prevent threshold explosion
    
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
        # axis='y' scans a column (indices are y, 0..h-1) → mid = h//2
        # axis='x' scans a row  (indices are x, 0..w-1) → mid = w//2
        mid = (w // 2) if axis == 'x' else (h // 2)
        intensity_ceiling = 250
        black_threshold = 28  # Reject black-border pixels as false edges
        edge_margin = max(confirm_pix, int(len(gray) * 0.02 if axis == 'y' else w * 0.02))
        min_grad = max(10.0, thresh * 0.30)

        def is_valid_candidate(i, step, line):
            pixel_val = line[i]
            if pixel_val <= black_threshold or pixel_val >= intensity_ceiling:
                return False

            diff = abs(pixel_val - ref_bg)
            if diff <= thresh:
                return False

            left = line[max(i - 1, 0)]
            right = line[min(i + 1, len(line) - 1)]
            if abs(right - left) < min_grad:
                return False

            # Require sustained deviation to suppress one-pixel noise and text specks.
            confirm_len = max(4, confirm_pix)
            win = [i + (k * step) for k in range(confirm_len)]
            win = [idx for idx in win if 0 <= idx < len(line)]
            if not win:
                return False
            strong = sum(1 for idx in win if abs(line[idx] - ref_bg) > (thresh * 0.70) and line[idx] > black_threshold)
            return strong >= max(3, int(0.60 * len(win)))
        
        for coord in scan_range:
            line_data = gray[:, coord].astype(float) if axis == 'y' else gray[coord, :].astype(float)
            
            if mode == 'inward':
                if direction == 'high':
                    # Scan from the edge inward, skip black boundary pixels
                    start_idx = len(line_data) - 1 - edge_margin
                    while start_idx > mid and line_data[start_idx] < black_threshold:
                        start_idx -= 1
                    indices = range(start_idx, mid, -1)
                else:  # direction == 'low'
                    # Scan from the edge inward, skip black boundary pixels
                    start_idx = edge_margin
                    while start_idx < mid and line_data[start_idx] < black_threshold:
                        start_idx += 1
                    indices = range(start_idx, mid)
            else:  # mode == 'outward'
                # Skip a proportional zone around the center so minor artifacts
                # near mid don't trigger (25 px was way too small for high-res images).
                outward_offset = max(25, int(len(line_data) * 0.10))
                if direction == 'high':
                    start_idx = mid + outward_offset
                    while start_idx < len(line_data) - edge_margin and line_data[start_idx] < black_threshold:
                        start_idx += 1
                    indices = range(start_idx, len(line_data) - edge_margin)
                else:  # direction == 'low'
                    start_idx = mid - outward_offset
                    while start_idx > edge_margin and line_data[start_idx] < black_threshold:
                        start_idx -= 1
                    indices = range(start_idx, edge_margin, -1)
            
            for i in indices:
                step = 1 if (mode == 'inward' and direction == 'low') or (mode == 'outward' and direction == 'high') else -1
                if is_valid_candidate(i, step, line_data):
                    if deep_scan_val > 0:
                        win_indices = [i + (s * step) for s in range(deep_scan_val)]
                        win_indices = [idx for idx in win_indices if 0 <= idx < len(line_data)]
                        win_values = [
                            abs(line_data[idx] - ref_bg)
                            if black_threshold < line_data[idx] < intensity_ceiling else 0
                            for idx in win_indices
                        ]
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


def detect_defects(image):
    """
    Detect defects using YOLO segmentation model, annotate the image, and return results.
    
    Args:
        image: Original BGR image.
    
    Returns:
        (annotated, results_text): Annotated image and list of result strings.
    """
    annotated = image.copy()
    results_text = []
    res_def = model_defects(image)[0]
    img_height, img_width = image.shape[:2]
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

    return annotated, results_text
