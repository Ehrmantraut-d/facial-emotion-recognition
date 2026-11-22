"""Polished Tkinter webcam viewer with on-device expression inference."""
from __future__ import annotations

import logging
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import cv2
from PIL import Image, ImageTk

from src.config import FRENCH_NAMES, MODEL_DIR
from src.ml.predictor import EmotionPredictor
from src.vision.detector import FaceDetector

LOG = logging.getLogger(__name__)


class EmotionApp:
    """Dark, card-based desktop interface for the FER2013 model."""

    BG = "#0b1220"
    PANEL = "#111c2e"
    PANEL_ALT = "#17243a"
    BORDER = "#24344d"
    TEXT = "#edf3ff"
    MUTED = "#91a2bb"
    ACCENT = "#57c7ff"
    GREEN = "#51d6a2"

    def __init__(self, root: tk.Tk, model_path: str | Path | None = None, camera_index: int = 0):
        self.root = root
        self.root.title("FER Studio | Expressions faciales")
        self.root.geometry("1120x760")
        self.root.minsize(980, 680)
        self.root.configure(bg=self.BG)
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._configure_style()

        path = Path(model_path) if model_path else MODEL_DIR / "selected_model.joblib"
        self.predictor = EmotionPredictor(path)
        self.detector = FaceDetector()
        self.camera_index = camera_index
        self.cap = None
        self.running = False
        self.photo = None
        self._last_frame_time = time.perf_counter()
        self._fps = 0.0

        self._build_layout()

    def _configure_style(self):
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=self.BG)
        style.configure("Panel.TFrame", background=self.PANEL, relief="flat")
        style.configure("AltPanel.TFrame", background=self.PANEL_ALT, relief="flat")
        style.configure("Title.TLabel", background=self.BG, foreground=self.TEXT,
                        font=("Segoe UI", 23, "bold"))
        style.configure("Subtitle.TLabel", background=self.BG, foreground=self.MUTED,
                        font=("Segoe UI", 10))
        style.configure("PanelTitle.TLabel", background=self.PANEL, foreground=self.TEXT,
                        font=("Segoe UI", 12, "bold"))
        style.configure("Small.TLabel", background=self.PANEL, foreground=self.MUTED,
                        font=("Segoe UI", 9))
        style.configure("Preview.TLabel", background="#08101d", foreground="#91a2bb",
                        font=("Segoe UI", 13), anchor="center", justify="center")
        style.configure("Prediction.TLabel", background=self.PANEL, foreground=self.ACCENT,
                        font=("Segoe UI", 25, "bold"))
        style.configure("Percent.TLabel", background=self.PANEL, foreground=self.GREEN,
                        font=("Segoe UI", 12, "bold"))
        style.configure("ScoreName.TLabel", background=self.PANEL, foreground=self.TEXT,
                        font=("Segoe UI", 9))
        style.configure("ScoreValue.TLabel", background=self.PANEL, foreground=self.MUTED,
                        font=("Segoe UI", 9))
        style.configure("Status.TLabel", background=self.BG, foreground=self.MUTED,
                        font=("Segoe UI", 9))
        style.configure("Primary.TButton", background="#1686c7", foreground="white",
                        font=("Segoe UI", 10, "bold"), borderwidth=0, padding=(18, 10))
        style.map("Primary.TButton", background=[("active", "#209bdc"), ("disabled", "#33465c")])
        style.configure("Secondary.TButton", background=self.PANEL_ALT, foreground=self.TEXT,
                        font=("Segoe UI", 10), borderwidth=0, padding=(16, 10))
        style.map("Secondary.TButton", background=[("active", "#233653"), ("disabled", "#17243a")])
        style.configure("Horizontal.TProgressbar", troughcolor="#223149",
                        background=self.ACCENT, bordercolor=self.PANEL, lightcolor=self.ACCENT,
                        darkcolor=self.ACCENT, thickness=8)

    def _build_layout(self):
        shell = ttk.Frame(self.root, style="App.TFrame", padding=(28, 22, 28, 16))
        shell.pack(fill="both", expand=True)
        shell.columnconfigure(0, weight=1)
        shell.rowconfigure(1, weight=1)

        header = ttk.Frame(shell, style="App.TFrame")
        header.grid(row=0, column=0, sticky="ew", pady=(0, 20))
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="FER Studio", style="Title.TLabel").grid(row=0, column=0, sticky="w")
        ttk.Label(header, text="Reconnaissance des expressions faciales · FER2013",
                  style="Subtitle.TLabel").grid(row=1, column=0, sticky="w", pady=(3, 0))
        ttk.Label(header, text="MODÈLE LOCAL  •  7 CLASSES", style="Subtitle.TLabel").grid(
            row=0, column=1, rowspan=2, sticky="e", padx=(16, 0))

        body = ttk.Frame(shell, style="App.TFrame")
        body.grid(row=1, column=0, sticky="nsew")
        body.columnconfigure(0, weight=7, uniform="col")
        body.columnconfigure(1, weight=3, uniform="col")
        body.rowconfigure(0, weight=1)

        left = ttk.Frame(body, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 14))
        left.rowconfigure(1, weight=1)
        left.columnconfigure(0, weight=1)
        ttk.Label(left, text="APERÇU CAMÉRA", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))
        preview = ttk.Frame(left, style="Panel.TFrame", padding=1)
        preview.grid(row=1, column=0, sticky="nsew")
        preview.rowconfigure(0, weight=1)
        preview.columnconfigure(0, weight=1)
        self.video = ttk.Label(preview, text="Caméra arrêtée\n\nDémarrez la caméra pour afficher l’aperçu.",
                               style="Preview.TLabel", anchor="center", justify="center")
        self.video.grid(row=0, column=0, sticky="nsew", padx=1, pady=1, ipady=80)

        controls = ttk.Frame(left, style="App.TFrame")
        controls.grid(row=2, column=0, sticky="ew", pady=(14, 0))
        self.start_btn = ttk.Button(controls, text="▶  Démarrer la caméra", style="Primary.TButton",
                                    command=self.start)
        self.start_btn.pack(side="left")
        self.stop_btn = ttk.Button(controls, text="■  Arrêter", style="Secondary.TButton",
                                   command=self.stop, state="disabled")
        self.stop_btn.pack(side="left", padx=(9, 0))
        ttk.Label(controls, text="Traitement local · aucune image enregistrée",
                  style="Subtitle.TLabel").pack(side="right", padx=(8, 0))

        right = ttk.Frame(body, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        right.rowconfigure(1, weight=1)
        right.columnconfigure(0, weight=1)
        ttk.Label(right, text="ANALYSE EN DIRECT", style="Subtitle.TLabel").grid(
            row=0, column=0, sticky="w", pady=(0, 8))

        prediction = ttk.Frame(right, style="Panel.TFrame", padding=20)
        prediction.grid(row=1, column=0, sticky="nsew")
        prediction.columnconfigure(0, weight=1)
        ttk.Label(prediction, text="Expression visible dominante", style="Small.TLabel").grid(
            row=0, column=0, sticky="w")
        self.prediction_title = ttk.Label(prediction, text="En attente", style="Prediction.TLabel")
        self.prediction_title.grid(row=1, column=0, sticky="w", pady=(8, 0))
        self.confidence = ttk.Label(prediction, text="—", style="Percent.TLabel")
        self.confidence.grid(row=2, column=0, sticky="w", pady=(2, 18))
        ttk.Separator(prediction).grid(row=3, column=0, sticky="ew", pady=(0, 16))
        ttk.Label(prediction, text="Scores par classe", style="PanelTitle.TLabel").grid(
            row=4, column=0, sticky="w", pady=(0, 12))

        self.score_bars = []
        self.score_labels = []
        for idx, label in enumerate(FRENCH_NAMES):
            row = ttk.Frame(prediction, style="Panel.TFrame")
            row.grid(row=5 + idx, column=0, sticky="ew", pady=5)
            row.columnconfigure(1, weight=1)
            ttk.Label(row, text=label, style="ScoreName.TLabel", width=12).grid(
                row=0, column=0, sticky="w", padx=(0, 8))
            bar = ttk.Progressbar(row, maximum=100, mode="determinate")
            bar.grid(row=0, column=1, sticky="ew")
            value = ttk.Label(row, text="0 %", style="ScoreValue.TLabel", width=5, anchor="e")
            value.grid(row=0, column=2, sticky="e", padx=(8, 0))
            self.score_bars.append(bar)
            self.score_labels.append(value)

        footer = ttk.Frame(shell, style="App.TFrame")
        footer.grid(row=2, column=0, sticky="ew", pady=(12, 0))
        footer.columnconfigure(0, weight=1)
        self.status = ttk.Label(footer, text="Prêt · le modèle est chargé", style="Status.TLabel")
        self.status.grid(row=0, column=0, sticky="w")
        self.fps_label = ttk.Label(footer, text="FPS  —", style="Status.TLabel")
        self.fps_label.grid(row=0, column=1, sticky="e")

    def _render_scores(self, scores: dict[int, float] | None):
        values = scores or {}
        for class_id, (bar, label) in enumerate(zip(self.score_bars, self.score_labels)):
            value = max(0.0, min(1.0, float(values.get(class_id, 0.0))))
            bar.configure(value=value * 100)
            label.configure(text=f"{value:.0%}")

    def start(self):
        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.cap.release()
            self.cap = None
            self.status.configure(text=f"Caméra {self.camera_index} indisponible")
            messagebox.showerror("Caméra indisponible", f"Impossible d'ouvrir la caméra {self.camera_index}.")
            return
        self.running = True
        self._last_frame_time = time.perf_counter()
        self.start_btn.configure(state="disabled")
        self.stop_btn.configure(state="normal")
        self.status.configure(text="Caméra active · analyse locale en cours")
        self._tick()

    def _tick(self):
        if not self.running or self.cap is None:
            return
        ok, frame = self.cap.read()
        if not ok or frame is None:
            self.status.configure(text="Image caméra invalide · nouvelle tentative…")
            self.root.after(80, self._tick)
            return

        now = time.perf_counter()
        delta = now - self._last_frame_time
        self._last_frame_time = now
        if delta > 0:
            instant_fps = 1.0 / delta
            self._fps = 0.9 * self._fps + 0.1 * instant_fps if self._fps else instant_fps
        self.fps_label.configure(text=f"FPS  {self._fps:.1f}")

        boxes = self.detector.detect(frame)
        if not boxes:
            self.status.configure(text="Aucun visage détecté")
            self.prediction_title.configure(text="—")
            self.confidence.configure(text="En attente d’un visage")
            self._render_scores(None)
        else:
            self.status.configure(text=f"{len(boxes)} visage(s) détecté(s) · scores indicatifs")
            for x, y, w, h in boxes:
                crop = frame[y:y + h, x:x + w]
                if crop.size == 0:
                    continue
                label, confidence, scores = self.predictor.predict_face(crop)
                color = (162, 214, 81)
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{FRENCH_NAMES[label]} {confidence:.0%}", (x, max(22, y - 8)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)
                self.prediction_title.configure(text=FRENCH_NAMES[label])
                self.confidence.configure(text=f"Confiance du modèle  ·  {confidence:.1%}")
                self._render_scores(scores)
                break

        cv2.putText(frame, f"{self._fps:.1f} FPS", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.62, (225, 235, 245), 2, cv2.LINE_AA)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image = Image.fromarray(rgb)
        image.thumbnail((900, 600))
        self.photo = ImageTk.PhotoImage(image)
        self.video.configure(image=self.photo, text="")
        self.root.after(15, self._tick)

    def stop(self):
        self.running = False
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.start_btn.configure(state="normal")
        self.stop_btn.configure(state="disabled")
        self.video.configure(image="", text="Caméra arrêtée\n\nDémarrez la caméra pour afficher l’aperçu.")
        self.photo = None
        if hasattr(self, "fps_label"):
            self.fps_label.configure(text="FPS  —")
        if hasattr(self, "status"):
            self.status.configure(text="Caméra arrêtée")
        if hasattr(self, "prediction_title"):
            self.prediction_title.configure(text="En attente")
            self.confidence.configure(text="—")
            self._render_scores(None)

    def close(self):
        self.stop()
        self.detector.close()
        self.root.destroy()


def main():
    logging.basicConfig(level=logging.INFO)
    root = tk.Tk()
    try:
        EmotionApp(root)
    except (FileNotFoundError, RuntimeError) as exc:
        messagebox.showerror("Modèle indisponible", str(exc))
        root.destroy()
        return
    root.mainloop()


if __name__ == "__main__":
    main()
