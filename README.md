# Lens Detection & Quality Control

This project is a Python-based application for automated lens inspection and quality control using computer vision and deep learning. It provides a modern GUI for image upload, annotation, measurement, and defect detection, leveraging YOLO models and OpenCV.

## Features

- **Lens Measurement (200x):** Detects circular lenses, measures diameters, and validates against tolerance.
- **Rectangle Measurement (40x):** Detects and measures rectangular regions, checks dimensions against standards.
- **Defect Detection:** Identifies visual defects in lens samples.
- **Image Annotation:** Annotates detected objects and measurements directly on images.
- **Results Output:** Saves annotated images and measurement/defect results as text files in organized folders.
- **Modern GUI:** Built with CustomTkinter, featuring tabbed views, zoom/pan, and real-time status updates.

## Folder Structure

- `GUI_13.py` — Main application code
- `200x_lens.pt`, `40x_rectt.pt`, `outer_rect.pt`, `defects.pt` — YOLO model files
- `history/` — Output folder for annotated images and results
  - `measurements/40x/` — Rectangle measurement results
  - `measurements/200x/` — Lens measurement results
  - `defects/` — Defect detection results

## Usage

1. **Install Requirements:**
   - Python 3.8+
   - Install dependencies:
     ```bash
     pip install customtkinter opencv-python pillow ultralytics
     ```
2. **Run the Application:**
   ```bash
   python GUI_13.py
   ```
3. **Workflow:**
   - Upload a microscope image.
   - Select mode (measurement or defect) and zoom level (40x or 200x).
   - Run detection to view annotated results and measurements.
   - Annotated images and text results are auto-saved in the `history/` folder.

## Measurement Standards

- **Lens Diameter (200x):** Target = 0.240 mm, Tolerance = ±0.005 mm
- **Rectangle (40x):**
  - Outer: 7.00 mm × 4.90 mm, Tolerance = ±0.010 mm
  - Inner: 5.60 mm × 2.40 mm, Tolerance = ±0.010 mm

## UI Conventions

- Main window: 1400×900 px, light mode, blue theme
- Sidebar: 280 px width, project info, mode/zoom selectors
- Tabs: Upload, Annotated Image, Results
- Annotated image area: 1200×800 px, zoom/pan enabled
- Results tab: Shows measurement/defect results, color-coded for tolerance

## Output Organization

- Annotated images and results text files are saved with timestamped filenames in the appropriate subfolders under `history/`.
- Example: `history/measurements/200x/sample_annotated_20251003_153000.png` and `sample_annotated_20251003_153000.txt`

## Model Files

Place the YOLO `.pt` model files in the project root. Models used:

- `200x_lens.pt` — Lens detection (200x)
- `40x_rectt.pt` — Rectangle detection (40x)
- `outer_rect.pt` — Rectangle segmentation
- `defects.pt` — Defect detection

## Error Handling

- Invalid image uploads and model inference errors are handled with user-friendly messages.
- Out-of-tolerance measurements are clearly indicated in both image and results text.

## License

This project is for research and educational use. Please contact the authors for commercial licensing.
