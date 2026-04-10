# Lens Detection & Quality Control

This project is a Python-based application for automated lens inspection and quality control using computer vision and deep learning. It provides a modern GUI for image upload, annotation, measurement, and defect detection, leveraging YOLO models, edge detection with RANSAC line fitting, and OpenCV.

## Features

- **Lens Measurement (200x):** Detects circular lenses, measures diameters, and validates against tolerance.
- **Rectangle Measurement:** Detects outer and inner rectangular regions using edge detection and RANSAC line fitting, checks dimensions against standards. Works at all zoom levels (40x, 80x, 200x).
- **Automatic Profile Detection:** Automatically classifies images into one of 6 color profiles (blue, dark, grey_light, grey_dark, yellow_dark, yellow_light) and applies optimized detection parameters for each.
- **Defect Detection:** Identifies visual defects in lens samples.
- **Image Annotation:** Annotates detected objects and measurements directly on images.
- **Results Output:** Saves annotated images and measurement/defect results as text files in organized folders.
- **Configurable Pixel Scale:** Pixel-to-mm scale for both 40x, 80x and 200x can be adjusted at runtime via the sidebar.
- **Modern GUI:** Built with CustomTkinter, featuring responsive layout, tabbed views, zoom/pan, and real-time status updates.

## Folder Structure

- `app.py` — UI class, layout, event binding, tab management (entry point)
- `constants.py` — Measurement targets, tolerances, pixel scales, profiles, annotation colors/params
- `detection.py` — Detection algorithms: `detect_profile()`, `detect_rectangles()`, `detect_and_annotate_lenses()`, `detect_defects()`, YOLO model loading
- `history.py` — `save_results()`, history directory management, timestamped file writing
- `circle.pt`, `defects.pt` — YOLO model files
- `history/` — Output folder for annotated images and results
  - `measurements/40x/` — 40x measurement results
  - `measurements/80x/` — 80x measurement results
  - `measurements/200x/` — 200x measurement results
  - `defects/` — Defect detection results

## Setup & Installation

1. **Create Virtual Environment:**

   ```powershell
   python -m venv venv
   .\venv\Scripts\Activate.ps1
   ```

2. **Install Dependencies:**

   ```powershell
   pip install -r requirements.txt
   ```

3. **Run the Application:**
   ```powershell
   python app.py
   ```

## Usage

**Workflow:**

- Upload a microscope image.
- Select mode (measurement or defect) and zoom level (40x / 80x / 200x).
- Run detection to view annotated results and measurements.
- Annotated images and text results are auto-saved in the `history/` folder.

## Measurement Standards

- **Lens Diameter:** Target = 0.240 mm, Tolerance = ±0.005 mm
- **Rectangle:**
  - Outer: 7.00 mm × 4.90 mm, Tolerance = ±0.010 mm
  - Inner: 5.60 mm × 2.40 mm, Tolerance = ±0.010 mm
- **Pixel Scales (default):** 40x = 387 px/mm, 80x = 780 px/mm, 200x = 1940 px/mm (configurable at runtime)

## UI Conventions

- Responsive window: 80% of screen size (1200–1600 × 800–1000 px), light mode, blue theme
- Sidebar: 250–320 px width (responsive), project info, mode/zoom selectors, pixel scale configuration
- Tabs: Upload New, Annotated Image, Results
- Annotated image area: responsive canvas with zoom/pan
- Results tab: Shows measurement/defect results, color-coded for tolerance

## Output Organization

- Annotated images and results text files are saved with timestamped filenames in the appropriate subfolders under `history/`.
- Example: `history/measurements/200x/sample_annotated_20251003_153000.png` and `sample_annotated_20251003_153000.txt`

## Model Files

Place the YOLO `.pt` model files in the project root. Models used:

- `circle.pt` — Lens/circle detection
- `defects.pt` — Defect detection

## Error Handling

- Invalid image uploads and model inference errors are handled with user-friendly messages.
- Out-of-tolerance measurements are clearly indicated in both image and results text.

## License

This project is for research and educational use. Please contact the authors for commercial licensing.
