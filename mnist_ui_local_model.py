"""
MNIST paint + LOCAL model inference.

UI loads the trained Keras model (mnist_model.keras) directly and predicts in
process. The model is loaded ONCE at application startup; each "Evaluate" click
then only calls model.predict() -- fast, no network, no .env / token needed.

The preprocessing mirrors predict.py: the 784 pixel values are divided by 255
and reshaped to (1, 28, 28, 1) before being fed to the model.

Left:   a 28x28 drawing canvas you paint on with a soft brush.
Middle: the FULL 28x28 grid of pixels, each cell shaded by its grayscale value
        (0-255) and labelled with that value. The hovered cell is highlighted.
Right:  the model's prediction output (per-digit probabilities + predicted label).

Pixel values use the MNIST convention (ink = high value): a drawn digit is bright
ink on a black background, matching the training data and what the model expects.

Run with:  python mnist_ui_local_model.py
"""
import logging
# Configure application logging. Each entry is prefixed with a timestamp.
# Format: 2026-01-31 14:22:05,123 [INFO] message
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("mnist_ui_local_model")

log.info("Loading TensorFlow libraries")

import os
# Silence TensorFlow's informational/warning logs. These MUST be set before
# tensorflow is imported, since the C++ backend reads them at import time.
# (Same approach as predict.py.)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # 0=all, 1=no INFO, 2=no WARNING, 3=no ERROR
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"  # removes the oneDNN custom-operations notice
import threading
import tkinter as tk
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model

# Also quiet the Python-side tensorflow logger (e.g. the GPU-support warning).
tf.get_logger().setLevel(logging.ERROR)


# ----- configuration -------------------------------------------------------
MODEL_PATH = "mnist_model.keras"

GRID = 28                 # image is GRID x GRID pixels
CANVAS_PX = 420           # on-screen size of the paint area (15 px per cell)
CELL = CANVAS_PX // GRID  # screen pixels per image pixel
BRUSH_RADIUS = 1.4        # brush size in image-pixel units

GRID_PX = 448             # on-screen size of the 28x28 value cells (not incl. labels)
GCELL = GRID_PX // GRID   # screen pixels per grid cell (16)
GMARGIN = 22              # gutter (px) on the top/left for the 1..28 row/col labels
GRID_CANVAS_PX = GRID_PX + GMARGIN  # full canvas size including the label gutter

# colours for the dark theme
BG = "#20222e"
PANEL = "#2a2d3a"
GRID_LINE = "#3a3d4d"
ACCENT = "#5b63d3"
TEXT = "#c9ccd6"
TEXT_DIM = "#6f7488"
TEXT_STRONG = "#e8eaf0"
HIGHLIGHT = "#e0455e"


class PaintApp:
    def __init__(self, root):
        self.root = root
        root.title("MNIST Paint - local model inference")
        root.configure(bg=BG)

        # The model is loaded once, at startup (see load_model_at_startup()).
        self.model = None

        # pixel intensities, 0.0 (no ink) .. 1.0 (full ink), row-major [y][x]
        self.pixels = [[0.0] * GRID for _ in range(GRID)]
        self.hover = None  # (row, col) currently under the cursor

        # cache of canvas item ids for the value grid so we can update fast
        self._grid_rects = [[None] * GRID for _ in range(GRID)]
        self._grid_texts = [[None] * GRID for _ in range(GRID)]
        self._highlight_id = None

        self._build_body()

        self.redraw_canvas()
        self.build_value_grid()
        self.refresh_status()

    # ----- model loading (once, at startup) --------------------------------
    def load_model_at_startup(self):
        """Load the Keras model in a background thread so the window shows
        immediately. Evaluate stays disabled until loading finishes."""
        self.eval_btn.config(state="disabled")
        self.status.config(text="loading model...")
        self._set_output(f"Loading model from {MODEL_PATH} ...")
        threading.Thread(target=self._load_model_worker, daemon=True).start()

    def _load_model_worker(self):
        repo_root = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(repo_root, MODEL_PATH)
        log.info("Loading model from %s ...", model_path)  # (2) loading model
        if not os.path.isfile(model_path):
            self.root.after(0, self._on_model_error,
                            f"Model file not found:\n{model_path}")
            return
        try:
            model = load_model(model_path)
        except Exception as exc:
            self.root.after(0, self._on_model_error, str(exc))
            return
        self.root.after(0, self._on_model_loaded, model)

    def _on_model_loaded(self, model):
        self.model = model
        log.info("Model loaded successfully.")  # (2) loading model - done
        self.eval_btn.config(state="normal")
        self.status.config(text="model ready - draw a digit, then Evaluate")
        self._set_output("Model loaded. Draw a digit, then press Evaluate.")

    def _on_model_error(self, message):
        log.error("Model failed to load: %s", message)  # (2) loading model - failed
        self.eval_btn.config(state="disabled")
        self.status.config(text="model failed to load")
        self._set_output(f"Could not load model:\n{message}")

    # ----- layout ----------------------------------------------------------
    def _build_body(self):
        body = tk.Frame(self.root, bg=BG)
        body.pack(padx=16, pady=10)

        # --- left: paint canvas ---
        left = tk.Frame(body, bg=BG)
        left.grid(row=0, column=0, padx=(0, 20), sticky="n")

        self.canvas = tk.Canvas(left, width=CANVAS_PX, height=CANVAS_PX,
                                bg="white", highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<Button-1>", self.on_draw)
        self.canvas.bind("<B1-Motion>", self.on_draw)
        self.canvas.bind("<Button-3>", self.on_erase)
        self.canvas.bind("<B3-Motion>", self.on_erase)
        self.canvas.bind("<Motion>", self.on_hover)
        self.canvas.bind("<Leave>", self.on_leave)

        tk.Label(left, text="draw a digit here", bg=BG, fg=TEXT_DIM,
                 font=("Consolas", 11)).pack(pady=(8, 0))

        # --- middle: full 28x28 value grid ---
        middle = tk.Frame(body, bg=BG)
        middle.grid(row=0, column=1, padx=(0, 20), sticky="n")

        self.grid_canvas = tk.Canvas(middle, width=GRID_CANVAS_PX,
                                     height=GRID_CANVAS_PX,
                                     bg=PANEL, highlightthickness=0)
        self.grid_canvas.pack()
        # hovering over the value grid also updates the highlight/status
        self.grid_canvas.bind("<Motion>", self.on_grid_hover)
        self.grid_canvas.bind("<Leave>", self.on_leave)

        tk.Label(middle, text="all 28x28 pixel values", bg=BG, fg=TEXT_DIM,
                 font=("Consolas", 11)).pack(pady=(8, 0))

        # --- right: prediction output ---
        right = tk.Frame(body, bg=BG)
        right.grid(row=0, column=2, sticky="n")

        self.output = tk.Text(right, width=40, height=24, bg=PANEL,
                              fg=TEXT_STRONG, insertbackground=TEXT,
                              relief="flat", font=("Consolas", 11),
                              highlightthickness=1,
                              highlightbackground=GRID_LINE, padx=12, pady=10,
                              wrap="none")
        self.output.pack()
        self.output.configure(state="disabled")

        tk.Label(right, text="model predictions", bg=BG, fg=TEXT_DIM,
                 font=("Consolas", 11)).pack(pady=(8, 0))

        # --- bottom status ---
        self.status = tk.Label(self.root, text="loading model...",
                               bg=BG, fg=TEXT_STRONG,
                               font=("Consolas", 14, "bold"))
        self.status.pack(pady=(4, 12))

        # controls hint + clear/evaluate buttons
        ctrl = tk.Frame(self.root, bg=BG)
        ctrl.pack(pady=(0, 12))
        tk.Button(ctrl, text="Clear", command=self.clear,
                  bg=PANEL, fg=TEXT, activebackground=ACCENT,
                  activeforeground="white", relief="flat",
                  font=("Segoe UI", 10), padx=14, pady=4).pack(side="left")
        self.eval_btn = tk.Button(ctrl, text="Evaluate", command=self.evaluate,
                                  bg=PANEL, fg=TEXT, activebackground=ACCENT,
                                  activeforeground="white", relief="flat",
                                  font=("Segoe UI", 10), padx=14, pady=4)
        self.eval_btn.pack(side="left", padx=(8, 0))
        tk.Label(ctrl, text="  left-drag = draw    right-drag = erase",
                 bg=BG, fg=TEXT_DIM, font=("Segoe UI", 9)).pack(side="left")

    # ----- value helpers ----------------------------------------------------
    def value_at(self, row, col):
        """Grayscale value 0-255 shown to the user.

        Uses the MNIST convention (matching the training data and what the model
        expects): a drawn digit is bright ink on a black background, so
        value = ink * 255. 0 = black (no ink), 255 = white (full ink).
        """
        ink = self.pixels[row][col]          # 0..1, amount of ink drawn
        return int(round(ink * 255))         # ink -> high value

    def display_shade(self, value):
        """Hex colour for a given 0-255 grayscale value."""
        v = max(0, min(255, value))
        return f"#{v:02x}{v:02x}{v:02x}"

    # ----- drawing on the canvas -------------------------------------------
    def _paint(self, event, ink_target):
        cx = event.x / CELL
        cy = event.y / CELL
        r = BRUSH_RADIUS
        for y in range(GRID):
            for x in range(GRID):
                d2 = (x + 0.5 - cx) ** 2 + (y + 0.5 - cy) ** 2
                if d2 <= r * r:
                    fall = max(0.0, 1.0 - (d2 ** 0.5) / (r + 0.0001))
                    if ink_target == 1.0:
                        self.pixels[y][x] = min(1.0, self.pixels[y][x] + fall)
                    else:
                        self.pixels[y][x] = max(0.0, self.pixels[y][x] - fall * 1.5)
        self.redraw_canvas()
        self.update_value_grid()
        self.on_hover(event)

    def on_draw(self, event):
        self._paint(event, 1.0)

    def on_erase(self, event):
        self._paint(event, 0.0)

    def clear(self):
        log.info("Button clicked: Clear (canvas reset).")  # (3) clicking buttons
        self.pixels = [[0.0] * GRID for _ in range(GRID)]
        self.redraw_canvas()
        self.update_value_grid()
        if self.model is not None:
            self._set_output("draw a digit, then press Evaluate.")
        self.refresh_status()

    # ----- rendering: paint canvas -----------------------------------------
    def redraw_canvas(self):
        self.canvas.delete("all")
        for y in range(GRID):
            for x in range(GRID):
                val = self.value_at(y, x)
                self.canvas.create_rectangle(
                    x * CELL, y * CELL, (x + 1) * CELL, (y + 1) * CELL,
                    fill=self.display_shade(val), outline="")

    # ----- rendering: full value grid --------------------------------------
    def build_value_grid(self):
        """Create the 28x28 rectangles + labels once; later just recolour.

        The cells are offset by GMARGIN to leave a gutter on the top and left
        for the 1..28 column/row labels (1-based indexing shown to the user).
        """
        c = self.grid_canvas
        c.delete("all")
        show_text = GCELL >= 15  # only draw numbers if cells are big enough

        # 1..28 labels along the top (columns) and left (rows).
        for i in range(GRID):
            centre = GMARGIN + i * GCELL + GCELL // 2
            # column number above the grid
            c.create_text(centre, GMARGIN // 2, text=str(i + 1),
                          fill=TEXT_DIM, font=("Consolas", 6))
            # row number to the left of the grid
            c.create_text(GMARGIN // 2, centre, text=str(i + 1),
                          fill=TEXT_DIM, font=("Consolas", 6))

        for row in range(GRID):
            for col in range(GRID):
                x0 = GMARGIN + col * GCELL
                y0 = GMARGIN + row * GCELL
                x1, y1 = x0 + GCELL, y0 + GCELL
                val = self.value_at(row, col)
                rect = c.create_rectangle(x0, y0, x1, y1,
                                          fill=self.display_shade(val),
                                          outline=GRID_LINE)
                self._grid_rects[row][col] = rect
                if show_text:
                    fg = "#111111" if val > 140 else "#e8eaf0"
                    txt = c.create_text((x0 + x1) // 2, (y0 + y1) // 2,
                                        text=str(val), fill=fg,
                                        font=("Consolas", 6))
                    self._grid_texts[row][col] = txt
        self._highlight_id = c.create_rectangle(-10, -10, -10, -10,
                                                outline=HIGHLIGHT, width=2)

    def update_value_grid(self):
        """Recolour/relabel existing grid items to match current pixels."""
        c = self.grid_canvas
        for row in range(GRID):
            for col in range(GRID):
                val = self.value_at(row, col)
                c.itemconfig(self._grid_rects[row][col],
                             fill=self.display_shade(val))
                tid = self._grid_texts[row][col]
                if tid is not None:
                    fg = "#111111" if val > 140 else "#e8eaf0"
                    c.itemconfig(tid, text=str(val), fill=fg)

    def _move_highlight(self):
        c = self.grid_canvas
        if self._highlight_id is None:
            return
        if self.hover is None:
            c.coords(self._highlight_id, -10, -10, -10, -10)
            return
        row, col = self.hover
        x0 = GMARGIN + col * GCELL
        y0 = GMARGIN + row * GCELL
        c.coords(self._highlight_id, x0 + 1, y0 + 1,
                 x0 + GCELL - 1, y0 + GCELL - 1)
        c.tag_raise(self._highlight_id)

    # ----- hover handling ---------------------------------------------------
    def on_hover(self, event):
        col = int(event.x // CELL)
        row = int(event.y // CELL)
        self._set_hover(row, col)

    def on_grid_hover(self, event):
        # Subtract the label gutter before mapping to a cell index.
        col = int((event.x - GMARGIN) // GCELL)
        row = int((event.y - GMARGIN) // GCELL)
        self._set_hover(row, col)

    def _set_hover(self, row, col):
        if 0 <= row < GRID and 0 <= col < GRID:
            self.hover = (row, col)
            self._move_highlight()
            self.refresh_status()

    def on_leave(self, _event):
        self.hover = None
        self._move_highlight()

    def refresh_status(self):
        if self.hover is None:
            return
        row, col = self.hover
        val = self.value_at(row, col)
        # Show 1-based indexing to the user: internal [0][0] -> pixels[1][1].
        self.status.config(text=f"pixels[{row + 1}][{col + 1}] = {val}")

    # ----- output panel -----------------------------------------------------
    def _set_output(self, text):
        self.output.configure(state="normal")
        self.output.delete("1.0", "end")
        self.output.insert("1.0", text)
        self.output.configure(state="disabled")

    # ----- evaluate: run the local model -----------------------------------
    def _current_image(self):
        """Build the (1, 28, 28, 1) float array the model expects.

        The model was trained on MNIST, where a digit is bright ink (high
        value, ~255) on a black background (0) -- i.e. value = ink * 255.
        Values are normalized to [0, 1] and reshaped exactly like predict.py:
            X = (values / 255.0).reshape(-1, 28, 28, 1)
        """
        pixels = np.array(self.pixels, dtype="float32")  # (28, 28), ink 0..1
        # ink*255 then /255 == ink; we keep the explicit form to mirror predict.py.
        values = pixels * 255.0
        return (values / 255.0).reshape(-1, 28, 28, 1)

    def evaluate(self):
        log.info("Button clicked: Evaluate.")  # (3) clicking buttons
        if self.model is None:
            log.warning("Evaluate ignored: model is not loaded yet.")
            self._set_output("Model is not loaded yet.")
            return

        # Disable the button and run inference off the UI thread so the window
        # stays responsive (the very first predict() is slower).
        self.eval_btn.config(state="disabled")
        self.status.config(text="predicting...")
        self._set_output("Running local model...")
        image = self._current_image()
        # (4) input content to prediction - log the image shape, a short summary,
        # and ALL 784 pixel values (flattened, row-major) so the exact input is
        # captured. Values are rounded to 3 decimals to keep the line readable.
        non_zero = int(np.count_nonzero(image))
        flat = image.reshape(-1)
        # Normalized (0.0-1.0) input log - commented out; the raw 0-255 record below
        # covers the same pixels in a more human-readable form.
        # pixel_values = [round(float(v), 3) for v in flat]
        # log.info("Prediction input (normalized 0.0-1.0): shape=%s, non-zero pixels=%d, "
        #          "max=%.3f, pixels=%s",
        #          image.shape, non_zero, float(image.max()), pixel_values)
        # Extra record: the same pixels WITHOUT normalization, i.e. the raw
        # 0-255 grayscale values (image * 255, rounded to integers).
        raw_values = [int(round(float(v) * 255)) for v in flat]
        log.info("Prediction input (raw 0-255): pixels=%s", raw_values)
        threading.Thread(target=self._predict_worker, args=(image,),
                         daemon=True).start()

    def _predict_worker(self, image):
        try:
            predictions = self.model.predict(image, verbose=0)
            probs = predictions[0]
            predicted = int(np.argmax(probs))
        except Exception as exc:
            self.root.after(0, self._show_error, str(exc))
            return
        self.root.after(0, self._show_result, probs, predicted)

    def _show_error(self, message):
        log.error("Prediction failed: %s", message)  # (5) result of prediction (error)
        self._set_output(f"Prediction failed:\n{message}")
        self.status.config(text="prediction failed")
        self.eval_btn.config(state="normal")

    def _show_result(self, probs, predicted):
        self.eval_btn.config(state="normal")

        # (5) result of prediction - log the predicted digit and its confidence.
        confidence = float(probs[predicted])
        log.info("Prediction result: digit=%d, confidence=%.6f",
                 predicted, confidence)

        lines = ["Predictions:", ""]
        for digit, prob in enumerate(probs):
            lines.append(f"    digit {digit}: {prob:.6f}")
        lines.append("")
        lines.append(f"Predicted Label(digit): {predicted}")
        self._set_output("\n".join(lines))
        self.status.config(text=f"predicted digit: {predicted}")


def main():
    log.info("Application started.")  # (1) start of script
    root = tk.Tk()
    app = PaintApp(root)
    app._set_output("Loading model...")
    root.resizable(False, False)
    # Kick off model loading once the window is up.
    root.after(100, app.load_model_at_startup)
    root.mainloop()


if __name__ == "__main__":
    main()
