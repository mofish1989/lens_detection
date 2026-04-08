# === Constants ===
OUT_RECT_LENGTH = 7.00
OUT_RECT_BREADTH = 4.90
IN_RECT_LENGTH = 5.60
IN_RECT_BREADTH = 2.40
RECT_TOL = 0.010

TARGET_LENS_DIAMETER = 0.240
LENS_TOL = 0.005

PIXEL_SCALES = {"40x": 387, "80x": 780, "200x": 1940}

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
