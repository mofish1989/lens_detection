import cv2
import customtkinter as ctk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import os
import threading

from constants import (
    PIXEL_SCALES,
    FONT_SCALE_200X, THICKNESS_200X, FONT_SCALE_40X, THICKNESS_40X,
)
from detection import detect_rectangles, detect_and_annotate_lenses, detect_defects
from history import save_results

try:
    import ollama
    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False

# === Chat / Analysis Assistant ===
# Uses a local Ollama server (https://ollama.com). Install with `pip install ollama`,
# run `ollama serve`, and pull a model first, e.g. `ollama pull llama3.1`.
# Override the model/host via env vars if you like: OLLAMA_MODEL, OLLAMA_HOST.
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "gemma4:e4b")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
os.environ["OLLAMA_HOST"] = OLLAMA_HOST  # ensure ollama.chat() picks up the same host
CHAT_PANEL_WIDTH = 340


def _load_logo_image(path, max_w=250, max_h=140):
    try:
        img = Image.open(path)
        img.thumbnail((max_w, max_h))
        return ctk.CTkImage(light_image=img, size=(img.width, img.height))
    except Exception:
        return None


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

        # --- Chat / Analysis Assistant State ---
        self.chat_visible = True
        self.chat_messages = []  # conversation history sent to Ollama: [{"role": ..., "content": ...}]
        self.chat_frame = None
        self.chat_log = None
        self.chat_entry = None
        self.chat_send_btn = None
        self.ollama_ready = OLLAMA_AVAILABLE

        # --- Build UI ---
        self._set_responsive_geometry()
        self.app.title("Lens Quality Check")
        self.app.configure(fg_color="#e0e0e0")

        self._build_sidebar()
        self._build_main_frame()
        self._build_chat_panel()
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
        corner_radius=10).pack(pady=(0, 10))

        ctk.CTkButton(self.sidebar, text="Toggle Analysis Chat",
        command=self._toggle_chat_panel,
        width=200, height=40, font=("Arial", 18),
        corner_radius=10).pack(pady=(0, 20))

        self._update_status_label()

    # === Main Frame ===

    def _build_main_frame(self):
        self.main_frame = ctk.CTkFrame(self.app, fg_color="#eaeaea", corner_radius=15)
        self.main_frame.pack(side="left", fill="both", expand=True, padx=(10, 20), pady=20)

        self.content_frame = ctk.CTkFrame(self.main_frame, fg_color="#eaeaea")
        self.content_frame.pack(fill="both", expand=True, padx=10, pady=10)

    # === Analysis Chat Panel ===

    def _build_chat_panel(self):
        self.chat_frame = ctk.CTkFrame(self.app, width=CHAT_PANEL_WIDTH, fg_color="#eaeaea", corner_radius=15)
        self.chat_frame.pack(side="right", fill="y", padx=(0, 20), pady=20)
        self.chat_frame.pack_propagate(False)

        header = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        header.pack(fill="x", padx=15, pady=(15, 5))
        ctk.CTkLabel(header, text="Analysis Assistant", font=("Arial", 18, "bold")).pack(side="left")
        ctk.CTkButton(header, text="✕", width=28, height=28, corner_radius=8,
        command=self._toggle_chat_panel).pack(side="right")

        self.chat_log = ctk.CTkTextbox(self.chat_frame, wrap="word", fg_color="white",
            font=("Arial", 13), corner_radius=10, state="disabled")
        self.chat_log.pack(fill="both", expand=True, padx=15, pady=(5, 10))
        self.chat_log.tag_config("user", foreground="#1f6aa5")
        self.chat_log.tag_config("assistant", foreground="#222222")
        self.chat_log.tag_config("system", foreground="#888888")

        quick_row = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        quick_row.pack(fill="x", padx=15, pady=(0, 8))
        ctk.CTkButton(quick_row, text="Analyze current results", height=32, corner_radius=8,
            font=("Arial", 12), command=self._quick_analyze_results).pack(fill="x")

        input_row = ctk.CTkFrame(self.chat_frame, fg_color="transparent")
        input_row.pack(fill="x", padx=15, pady=(0, 15))

        self.chat_entry = ctk.CTkEntry(input_row, placeholder_text="Ask about the results...",
            height=40, font=("Arial", 13), corner_radius=10)
        self.chat_entry.pack(side="left", fill="x", expand=True, padx=(0, 8))
        self.chat_entry.bind("<Return>", lambda e: self._send_chat_message())

        self.chat_send_btn = ctk.CTkButton(input_row, text="Send", width=64, height=40,
            corner_radius=10, font=("Arial", 13),
            command=self._send_chat_message)
        self.chat_send_btn.pack(side="right")

        if not OLLAMA_AVAILABLE:
            self._append_chat_message("system", "The 'ollama' package isn't installed. Run: pip install ollama")
        elif not self.ollama_ready:
            self._append_chat_message("system", "Couldn't set up the Ollama client. Check OLLAMA_HOST.")
        else:
            self._append_chat_message("system",
                f"Ask me about your QC results, tolerances, or trends. (model: {OLLAMA_MODEL} via {OLLAMA_HOST})")

    def _toggle_chat_panel(self):
        if self.chat_visible:
            self.chat_frame.pack_forget()
        else:
            self.chat_frame.pack(side="right", fill="y", padx=(0, 20), pady=20)
        self.chat_visible = not self.chat_visible

    def _append_chat_message(self, role, text):
        self.chat_log.configure(state="normal")
        prefix = {"user": "You: ", "assistant": "Assistant: ", "system": ""}.get(role, "")
        self.chat_log.insert("end", f"{prefix}{text}\n\n", (role,))
        self.chat_log.configure(state="disabled")
        self.chat_log.see("end")

    def _build_results_context(self):
        """Summarize the current mode/zoom/results so the assistant can reason about them."""
        if not self.results_lines:
            return "No detection has been run yet."
        mode = self.mode_var.get()
        zoom = self.zoom_var.get()
        lines = "\n".join(self.results_lines)
        return f"Mode: {mode}\nZoom: {zoom}\n\nResults:\n{lines}"

    def _quick_analyze_results(self):
        if not self.results_lines:
            messagebox.showinfo("No results", "Run a detection first so there's something to analyze.")
            return
        self._send_chat_message(
            preset="Summarize these QC results, flag anything out of tolerance, and note any patterns worth attention.")

    def _send_chat_message(self, preset=None):
        user_text = preset if preset is not None else self.chat_entry.get().strip()
        if not user_text:
            return
        if not self.ollama_ready:
            messagebox.showwarning("Assistant unavailable",
                                   "Install the 'ollama' package (pip install ollama) and make sure "
                                   "the Ollama server is running (ollama serve) first.")
            return

        if preset is None:
            self.chat_entry.delete(0, "end")
        self._append_chat_message("user", user_text)
        self.chat_send_btn.configure(state="disabled", text="...")

        self.chat_messages.append({"role": "user", "content": user_text})
        history_copy = list(self.chat_messages)
        context = self._build_results_context()

        threading.Thread(target=self._call_chat_api, args=(history_copy, context), daemon=True).start()

    def _call_chat_api(self, history, context):
        try:
            system_prompt = (
                "You are a QC analysis assistant for a lens/rectangle inspection tool. "
                "Answer questions about the measurement and defect results below. "
                "Be concise and specific about which measurements are out of tolerance.\n\n"
                f"Current session data:\n{context}"
            )
            messages = [{"role": "system", "content": system_prompt}] + history
            response = ollama.chat(model=OLLAMA_MODEL, messages=messages)
            reply_text = response["message"]["content"]
        except Exception as e:
            reply_text = (
                f"Error contacting Ollama: {e}\n\n"
                f"Make sure the Ollama server is running (ollama serve) and the model is pulled "
                f"(ollama pull {OLLAMA_MODEL})."
            )

        self.app.after(0, lambda: self._handle_chat_response(reply_text))

    def _handle_chat_response(self, reply_text):
        self.chat_messages.append({"role": "assistant", "content": reply_text})
        self._append_chat_message("assistant", reply_text)
        self.chat_send_btn.configure(state="normal", text="Send")

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
                annotated, rect_results = detect_rectangles(self.uploaded_image, pixel_scale)

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
            annotated, defect_results = detect_defects(self.uploaded_image)
            results_text.extend(defect_results)

        self.annotated_image = annotated
        self.results_lines = results_text

        # --- Auto-save annotated image ---
        if self.annotated_image is not None:
            save_results(self.annotated_image, self.results_lines,
            self.uploaded_image_path, mode, self.zoom_var.get())

        self._update_preview_tab()
        self._update_annotated_tab()
        self._update_results_tab()

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