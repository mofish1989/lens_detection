# AI Agent Instructions for Lens Detection Project

## Project Overview

This is a Python-based quality control application for lens inspection using computer vision and deep learning. The application uses YOLO models for detecting and measuring lens dimensions and identifying defects in microscope images.

## Key Components

### YOLO Models

- `200x_lens.pt`: Detects and measures circular lenses at 200x magnification
- `40x_rectt.pt`: Detects rectangular regions at 40x magnification
- `outer_rect.pt`: Segmentation model for outer rectangle detection
- `defects.pt`: Detects visual defects in lens samples

### Constants & Measurement Standards

```python
OUT_RECT_LENGTH = 7.00  # mm
OUT_RECT_BREADTH = 4.90  # mm
IN_RECT_LENGTH = 5.60  # mm
IN_RECT_BREADTH = 2.40  # mm
RECT_TOL = 0.010  # mm tolerance

TARGET_LENS_DIAMETER = 0.240  # mm
LENS_TOL = 0.005  # mm tolerance

PIXEL_SCALES = {"40x": 380, "200x": 1940}  # pixels/mm at different magnifications
```

## Core Workflows

### Image Processing Pipeline

1. Image loading and preprocessing in BGR format (OpenCV)
2. Model inference based on selected mode:
   - Measurement mode:
     - At 200x: Lens diameter measurement
     - At 40x: Rectangle dimension measurement
   - Defect detection mode
3. Result annotation and measurement validation
4. Automatic result saving with timestamps

### User Interface Patterns

- Uses customtkinter for modern UI components
- Tabbed interface with Upload, Annotated Image, and Results views
- Zoom and pan functionality for detailed image inspection
- Real-time status updates and measurement display

## Project Conventions

### Image Handling

- Images are loaded and processed in BGR color space (OpenCV standard)
- Annotated images are automatically saved in the `history/` directory
- Naming pattern: `{base_name}_annotated_{timestamp}.png`

### Measurement Validation

- Measurements are checked against defined tolerances
- Results are color-coded:
  - Green: Within tolerance
  - Red: Out of tolerance

### Error Handling

- Failed image loads trigger user-friendly error messages
- Invalid measurements are clearly marked as out of tolerance
- Model inference errors are gracefully handled with user feedback

## Common Tasks

### Adding New Features

1. Update model constants if new measurements are needed
2. Add detection logic in the `run_detection()` function
3. Update the UI in relevant sections (preview, annotation, results)

### Model Updates

- YOLO model files (.pt) should be placed in the root directory
- Update `PIXEL_SCALES` if using new magnification levels
- Adjust detection confidence thresholds in model inference calls

### UI Modifications & Layout Standards

#### Component Layout

- Main window dimensions: 1400x900 pixels
- Sidebar width: 280 pixels
- Main frame: Expands to fill remaining space
- Use consistent padding: `padx=20, pady=20` for main sections
- Tab container: Use a white background frame with `corner_radius=10`, full width matching the main content frame, and padding `padx=20, pady=20`.
- Tabs: Height 50px, font size 14pt bold, increased button padding (`padx=30, pady=15`), and rounded corners for segmented buttons.
- Tab content: Place all tab content inside an inner frame with `padx=20, pady=20` for consistent spacing. Add extra padding (`padx=30, pady=20`) for main content areas inside each tab.

#### Font & Text Settings

- Headers: Arial 22pt bold (e.g., project title)
- Status text: Arial 12pt (status labels)
- Results text: Arial 25pt (measurement results)
- Use wraplength=200 for sidebar labels to prevent text overflow

#### Button & Control Sizing

- Large buttons (e.g., Upload): width=200, height=50, font=18pt
- Standard buttons: Default CustomTkinter sizing
- Preview image area: 1000x600 pixels
- Annotated view canvas: 1200x800 pixels

#### Visual Hierarchy

- Use CustomTkinter's built-in light mode and blue theme
- Maintain consistent spacing:
  - Between major sections: 20px
  - Between related controls: 10px
  - Tabs: Add extra space above and below tab container for visual separation
- Group related controls in CTkFrames
- Use corner_radius=10 for rounded container corners
