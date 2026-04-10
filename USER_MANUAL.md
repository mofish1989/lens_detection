# Lens Quality Check — User Manual

This guide walks you through installing and using the Lens Quality Check application from scratch. No programming experience is required.

---

## Table of Contents

1. [What You Need Before Starting](#1-what-you-need-before-starting)
2. [One-Time Installation](#2-one-time-installation)
3. [Starting the Application](#3-starting-the-application)
4. [Understanding the Interface](#4-understanding-the-interface)
5. [Inspecting an Image — Step by Step](#5-inspecting-an-image--step-by-step)
6. [Viewing Results](#6-viewing-results)
7. [Adjusting Pixel Scale](#7-adjusting-pixel-scale)
8. [Where Results Are Saved](#8-where-results-are-saved)
9. [Understanding Measurement Results](#9-understanding-measurement-results)
10. [Zooming and Panning on Annotated Images](#10-zooming-and-panning-on-annotated-images)
11. [Troubleshooting](#11-troubleshooting)

---

## 1. What You Need Before Starting

- A Windows computer
- Python 3.10 or newer installed on your computer
  - To check: open the Start menu, type `cmd`, press Enter, then type `python --version` and press Enter. You should see something like `Python 3.12.x`.
  - If Python is not installed, download it from https://www.python.org/downloads/ — during installation, **make sure to check the box that says "Add Python to PATH"**.
- The project folder provided to you (containing the application files and model files)

---

## 2. One-Time Installation

You only need to do this once, the first time you set up the application.

### Step 1 — Open the Command Prompt

1. Open **File Explorer** and navigate to the project folder (the folder containing `app.py`).
2. Click on the **address bar** at the top of the File Explorer window (where the folder path is shown).
3. Type `cmd` and press **Enter**. A black Command Prompt window will open, already pointed at the project folder.

### Step 2 — Create a Virtual Environment

In the Command Prompt window, type the following command and press **Enter**:

```
python -m venv venv
```

Wait a few seconds. This creates a private space for the application's dependencies so they don't interfere with anything else on your computer.

### Step 3 — Activate the Virtual Environment

Type the following command and press **Enter**:

```
venv\Scripts\activate
```

You should now see `(venv)` appear at the beginning of the line. This means the virtual environment is active.

### Step 4 — Install Dependencies

Type the following command and press **Enter**:

```
pip install -r requirements.txt
```

This will download and install all the software the application needs. It may take several minutes depending on your internet connection. Wait until the process finishes and you see the prompt again.

### Step 5 — Verify Installation

Type the following command and press **Enter**:

```
python -c "import customtkinter; import cv2; import ultralytics; print('All good!')"
```

If you see `All good!` printed, the installation was successful. You can now close the Command Prompt window.

---

## 3. Starting the Application

After the one-time installation, follow these steps each time you want to use the application.

### Step 1 — Open the Command Prompt in the Project Folder

1. Open **File Explorer** and navigate to the project folder.
2. Click the **address bar**, type `cmd`, and press **Enter**.

### Step 2 — Activate the Virtual Environment

```
venv\Scripts\activate
```

You should see `(venv)` at the beginning of the line.

### Step 3 — Launch the Application

```
python app.py
```

The application window will open. It may take a few seconds on the first launch while the models load.

---

## 4. Understanding the Interface

The application window has two main areas:

### Left Sidebar

| Item                  | Description                                                                                                                                             |
| --------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Mode** dropdown     | Choose between **measurement** (measure lens/rectangle dimensions) or **defect** (find visual defects).                                                 |
| **Zoom** dropdown     | Select the microscope magnification used to capture the image: **40x**, **80x**, or **200x**.                                                           |
| **Status display**    | Shows your current mode and zoom selection.                                                                                                             |
| **Pixel to mm Scale** | Advanced setting — shows the conversion factor from pixels to millimeters for each zoom level. Leave at the default values unless instructed otherwise. |

### Main Area (Tabs)

| Tab                 | Description                                                                                                                 |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------- |
| **Upload New**      | Upload a microscope image and start detection.                                                                              |
| **Annotated Image** | View the image with detection results drawn on top (rectangles, lens labels, defect highlights). You can zoom and pan here. |
| **Results**         | View measurement values and pass/fail status in text form.                                                                  |

---

## 5. Inspecting an Image — Step by Step

### Step 1 — Choose the Correct Settings

Before uploading, set the correct options in the sidebar:

- **Mode**: Select `measurement` to measure dimensions, or `defect` to check for visual defects.
- **Zoom**: Select the magnification that matches your microscope image (`40x`, `80x`, or `200x`).

> **Important:** The zoom setting must match the actual magnification of the image, otherwise measurements will be incorrect.

### Step 2 — Upload an Image

1. Click the **Upload New** tab (it should already be selected).
2. Click the **Upload Image** button.
3. A file browser window will appear. Navigate to and select your microscope image file (`.jpg`, `.jpeg`, `.png`, or `.bmp`).
4. Click **Open**. A preview of your image will appear.

### Step 3 — Run Detection

1. Click the **Run Detection** button (next to Upload Image).
2. Wait a moment while the application analyzes the image.
3. The application will automatically switch to the **Annotated Image** tab to show you the results.

---

## 6. Viewing Results

### Annotated Image Tab

After detection, the **Annotated Image** tab shows your original image with annotations drawn on top:

- **Green** outlines and text = measurements are within tolerance (PASS)
- **Red** outlines and text = measurements are out of tolerance (FAIL) or defects detected

### Results Tab

Click the **Results** tab to see a text summary of all measurements:

- For **measurement mode**, you will see:
  - Rectangle dimensions (length × breadth in mm) with pass/fail status
  - Lens diameters in mm with pass/fail status
  - Center-to-center distances between lenses
- For **defect mode**, you will see:
  - Number of defects found
  - Type and location of each defect

Results that are **out of tolerance** are displayed in **red text**.

---

## 7. Adjusting Pixel Scale

The pixel scale determines how the application converts pixels in the image to real-world millimeters. Default values are pre-configured and should work for most setups.

If your microscope calibration requires different values:

1. In the sidebar, locate the **Pixel to mm Scale** section.
2. Edit the number next to the zoom level you need to change (40x, 80x, or 200x).
3. Click the **Confirm Scale** button.
4. A confirmation message will appear showing the updated values.

> Only change these values if you have been given specific calibration numbers for your microscope.

---

## 8. Where Results Are Saved

Every time you run detection, the application automatically saves two files:

1. **Annotated image** (`.png`) — the image with all markings drawn on it
2. **Results text file** (`.txt`) — the text version of all measurements

These files are saved inside the `history` folder within the project folder, organized by type:

| What You Inspected  | Save Location                |
| ------------------- | ---------------------------- |
| Measurement at 40x  | `history/measurements/40x/`  |
| Measurement at 80x  | `history/measurements/80x/`  |
| Measurement at 200x | `history/measurements/200x/` |
| Defect detection    | `history/defects/`           |

File names include the original image name and a timestamp, for example:
`sample_annotated_20260410_143000.png`

---

## 9. Understanding Measurement Results

### Lens Measurements

| Term                 | Meaning                                                            |
| -------------------- | ------------------------------------------------------------------ |
| **Lens Diameter**    | The measured diameter of a detected circular lens, in millimeters. |
| **OK**               | The diameter is within the acceptable range (0.235 mm – 0.245 mm). |
| **Out of Tolerance** | The diameter is outside the acceptable range.                      |

### Rectangle Measurements (40x / 80x)

The application detects two rectangles in each image:

| Rectangle | Target Length | Target Breadth | Tolerance  |
| --------- | ------------- | -------------- | ---------- |
| **Outer** | 7.00 mm       | 4.90 mm        | ± 0.010 mm |
| **Inner** | 5.60 mm       | 2.40 mm        | ± 0.010 mm |

- **OK** = both length and breadth are within tolerance
- **Out of Tolerance** = one or both dimensions are outside the acceptable range

---

## 10. Zooming and Panning on Annotated Images

On the **Annotated Image** tab, you can inspect details more closely:

- **Zoom in**: Scroll your mouse wheel **up**
- **Zoom out**: Scroll your mouse wheel **down**
- **Pan (move around)**: Click and **hold the left mouse button**, then drag in any direction

---

## 11. Troubleshooting

| Problem                                                         | Solution                                                                                                                                                                           |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `python` is not recognized                                      | Python is not installed or not added to PATH. Reinstall Python and check "Add Python to PATH" during setup.                                                                        |
| `pip install` fails with network errors                         | Check your internet connection. If you are behind a corporate firewall, contact your IT department.                                                                                |
| The application window doesn't appear                           | Make sure you activated the virtual environment (`venv\Scripts\activate`) before running `python app.py`.                                                                          |
| "Cannot load image" error                                       | The selected file may be corrupted or in an unsupported format. Use `.jpg`, `.jpeg`, `.png`, or `.bmp` files.                                                                      |
| Measurements seem wrong                                         | Verify that the **Zoom** dropdown matches the actual magnification of your microscope image. Check that the pixel scale values are correct for your microscope setup.              |
| Application is slow on first run                                | The first detection run loads the AI models into memory, which can take a few seconds. Subsequent runs will be faster.                                                             |
| `venv\Scripts\activate` gives an error about execution policies | Open PowerShell as Administrator and run: `Set-ExecutionPolicy RemoteSigned -Scope CurrentUser` — then try again. Alternatively, use Command Prompt (`cmd`) instead of PowerShell. |
