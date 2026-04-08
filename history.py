import os
import datetime
import cv2

HISTORY_DIR = "history"
os.makedirs(HISTORY_DIR, exist_ok=True)

_SAVE_DIRS = {
    ("measurement", "40x"):  "measurements/40x",
    ("measurement", "80x"):  "measurements/80x",
    ("measurement", "200x"): "measurements/200x",
    ("defect", None):        "defects",
}


def save_results(annotated_image, results_lines, uploaded_image_path, mode, zoom):
    """Save annotated image and results text to the history directory."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_name = os.path.splitext(os.path.basename(uploaded_image_path))[0]

    zoom_key = zoom if mode == "measurement" else None
    sub_dir = _SAVE_DIRS.get((mode, zoom_key), "")
    save_dir = os.path.join(HISTORY_DIR, sub_dir)
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, f"{base_name}_annotated_{timestamp}.png")
    txt_path = os.path.join(save_dir, f"{base_name}_annotated_{timestamp}.txt")
    cv2.imwrite(save_path, annotated_image)
    with open(txt_path, "w") as f:
        for line in results_lines:
            f.write(line + "\n")
