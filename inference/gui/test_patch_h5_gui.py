"""Minimal Tkinter GUI for testing PNG patches against PyTorch .pth models."""
from __future__ import annotations

import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import numpy as np
import torch
from PIL import Image, ImageTk

from inference.shared.patch_test_utils import (
    ACTIVATION_CHOICES,
    NORMALIZATION_CHOICES,
    PatchTestResult,
    build_result,
    load_class_names_file,
    load_pth_model,
    predict_patch,
    resolve_class_names,
)

SUPPORTED_ARCHES = ["resnet18", "resnet50"]


class PatchTesterGUI:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Patch .pth Tester")
        self.model = None
        self.patch_preview: Optional[ImageTk.PhotoImage] = None

        # Tk variables
        self.model_var = tk.StringVar()
        self.patch_var = tk.StringVar()
        self.class_var = tk.StringVar()
        self.image_size_var = tk.IntVar(value=224)
        self.topk_var = tk.IntVar(value=5)
        self.normalization_var = tk.StringVar(value="imagenet")
        self.activation_var = tk.StringVar(value="auto")
        self.arch_var = tk.StringVar(value="resnet50")
        self.num_classes_var = tk.IntVar(value=2)
        self.device_var = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")
        self.status_var = tk.StringVar(value="Load a .pth model to begin")

        self._build_layout()

    # ------------------------------------------------------------------ UI
    def _build_layout(self) -> None:
        main = ttk.Frame(self.root, padding=12)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)

        # Model row
        self._build_file_row(main, "Model (.pth)", self.model_var, 0, self._browse_model, self._load_model)
        # Patch row
        self._build_file_row(main, "Patch", self.patch_var, 1, self._browse_patch, None)
        # Class names row
        self._build_file_row(main, "Classes", self.class_var, 2, self._browse_classes, None)

        # Options
        options = ttk.LabelFrame(main, text="Inference Options", padding=8)
        options.grid(row=3, column=0, sticky="ew", pady=8)
        for col in range(4):
            options.columnconfigure(col, weight=1)

        ttk.Label(options, text="Image Size").grid(row=0, column=0, sticky="w")
        ttk.Spinbox(options, from_=32, to=2048, increment=16, textvariable=self.image_size_var, width=6).grid(row=0, column=1, sticky="w")
        ttk.Label(options, text="Top-K").grid(row=0, column=2, padx=(12, 0), sticky="w")
        ttk.Spinbox(options, from_=1, to=20, textvariable=self.topk_var, width=4).grid(row=0, column=3, sticky="w")

        ttk.Label(options, text="Architecture").grid(row=1, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(options, values=SUPPORTED_ARCHES, textvariable=self.arch_var, state="readonly", width=12).grid(
            row=1, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(options, text="Device").grid(row=1, column=2, padx=(12, 0), sticky="w", pady=(6, 0))
        ttk.Entry(options, textvariable=self.device_var, width=10).grid(row=1, column=3, sticky="w", pady=(6, 0))

        ttk.Label(options, text="Normalization").grid(row=2, column=0, sticky="w", pady=(6, 0))
        ttk.Combobox(options, values=NORMALIZATION_CHOICES, textvariable=self.normalization_var, state="readonly", width=12).grid(
            row=2, column=1, sticky="w", pady=(6, 0)
        )
        ttk.Label(options, text="Activation").grid(row=2, column=2, padx=(12, 0), sticky="w", pady=(6, 0))
        ttk.Combobox(options, values=ACTIVATION_CHOICES, textvariable=self.activation_var, state="readonly", width=12).grid(
            row=2, column=3, sticky="w", pady=(6, 0)
        )

        ttk.Label(options, text="# Classes").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Spinbox(options, from_=1, to=64, textvariable=self.num_classes_var, width=6).grid(
            row=3, column=1, sticky="w", pady=(6, 0)
        )

        # Buttons
        btn_row = ttk.Frame(main)
        btn_row.grid(row=4, column=0, sticky="ew")
        ttk.Button(btn_row, text="Run Test", command=self._run_test).grid(row=0, column=0, padx=(0, 8))
        ttk.Button(btn_row, text="Clear", command=self._clear_results).grid(row=0, column=1)

        # Results area
        results = ttk.Frame(main)
        results.grid(row=5, column=0, sticky="nsew", pady=(8, 0))
        main.rowconfigure(5, weight=1)
        results.columnconfigure(1, weight=1)

        self.preview_label = ttk.Label(results, text="Patch preview", relief="groove", anchor="center", width=32)
        self.preview_label.grid(row=0, column=0, sticky="n")

        self.tree = ttk.Treeview(results, columns=("class", "score"), show="headings", height=8)
        self.tree.heading("class", text="Class")
        self.tree.heading("score", text="Score")
        self.tree.column("class", width=200)
        self.tree.column("score", width=100, anchor="center")
        self.tree.grid(row=0, column=1, sticky="nsew", padx=(12, 0))

        scrollbar = ttk.Scrollbar(results, orient="vertical", command=self.tree.yview)
        self.tree.configure(yscrollcommand=scrollbar.set)
        scrollbar.grid(row=0, column=2, sticky="ns")

        ttk.Label(main, textvariable=self.status_var, anchor="w").grid(row=6, column=0, sticky="ew", pady=(8, 0))

    def _build_file_row(
        self,
        parent: ttk.Frame,
        label: str,
        variable: tk.StringVar,
        row: int,
        browse_callback,
        action_callback,
    ) -> None:
        frame = ttk.Frame(parent)
        frame.grid(row=row, column=0, sticky="ew", pady=2)
        frame.columnconfigure(1, weight=1)
        ttk.Label(frame, text=label).grid(row=0, column=0, sticky="w", padx=(0, 6))
        ttk.Entry(frame, textvariable=variable).grid(row=0, column=1, sticky="ew")
        ttk.Button(frame, text="Browse", command=browse_callback).grid(row=0, column=2, padx=(6, 0))
        if action_callback is not None:
            ttk.Button(frame, text="Load", command=action_callback).grid(row=0, column=3, padx=(6, 0))

    # ------------------------------------------------------------------ Event handlers
    def _browse_model(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("PyTorch model", "*.pth"), ("All files", "*.*")])
        if path:
            self.model_var.set(path)

    def _browse_patch(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.tif;*.tiff"), ("All files", "*.*")])
        if path:
            self.patch_var.set(path)
            self._update_preview(Path(path))

    def _browse_classes(self) -> None:
        path = filedialog.askopenfilename(filetypes=[("Class names", "*.json;*.txt;*.csv"), ("All files", "*.*")])
        if path:
            self.class_var.set(path)

    def _load_model(self) -> None:
        model_path = Path(self.model_var.get())
        if not model_path.exists():
            messagebox.showerror("Model missing", "Select a valid .pth file")
            return
        num_classes = self._determine_num_classes()
        if num_classes is None:
            return
        try:
            self.model = load_pth_model(
                self.arch_var.get(),
                num_classes,
                model_path,
                device=self._device_or_none(),
            )
        except Exception as err:  # pragma: no cover - UI feedback path
            messagebox.showerror("Model load failed", str(err))
            return
        self.status_var.set(f"Loaded model ({self.arch_var.get()} • {num_classes} classes)")

    def _update_preview(self, path: Path) -> None:
        try:
            image = Image.open(path).convert("RGB")
            image.thumbnail((256, 256))
            self.patch_preview = ImageTk.PhotoImage(image)
            self.preview_label.configure(image=self.patch_preview, text="")
        except Exception as err:  # pragma: no cover - UI feedback path
            self.preview_label.configure(text=f"Preview error: {err}")
            self.patch_preview = None

    def _clear_results(self) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        self.preview_label.configure(text="Patch preview", image="")
        self.patch_preview = None
        self.status_var.set("Cleared results")

    def _read_class_names(self, num_outputs: int) -> list[str]:
        path = self.class_var.get().strip()
        if not path:
            return resolve_class_names(None, num_outputs)
        try:
            names = load_class_names_file(Path(path))
            return resolve_class_names(names, num_outputs)
        except Exception as err:  # pragma: no cover - UI feedback path
            messagebox.showwarning("Class names", f"Failed to load class names: {err}")
            return resolve_class_names(None, num_outputs)

    def _determine_num_classes(self) -> Optional[int]:
        path = self.class_var.get().strip()
        if path:
            try:
                names = load_class_names_file(Path(path))
                self.num_classes_var.set(len(names))
                return len(names)
            except Exception as err:  # pragma: no cover - UI feedback path
                messagebox.showwarning("Class names", f"Failed to read class names: {err}")
                return None
        return max(1, int(self.num_classes_var.get()))

    def _device_or_none(self) -> Optional[str]:
        value = self.device_var.get().strip()
        return value or None

    def _run_test(self) -> None:
        if self.model is None:
            messagebox.showwarning("Model not loaded", "Please load a .pth model first")
            return
        patch_path = Path(self.patch_var.get())
        if not patch_path.exists():
            messagebox.showwarning("Patch missing", "Select a valid patch image")
            return
        try:
            probs = predict_patch(
                self.model,
                patch_path,
                image_size=self.image_size_var.get(),
                normalization=self.normalization_var.get(),
                activation=self.activation_var.get(),
                device=self._device_or_none(),
            )
        except Exception as err:  # pragma: no cover - UI feedback path
            messagebox.showerror("Inference error", str(err))
            return

        probs_vector = np.array(probs).flatten()
        class_names = self._read_class_names(len(probs_vector))
        result = build_result(probs_vector, class_names, self.topk_var.get())
        self._render_results(result)
        self.status_var.set(f"Inference complete: {patch_path.name}")

    def _render_results(self, result: PatchTestResult) -> None:
        for item in self.tree.get_children():
            self.tree.delete(item)
        for row in result.top_k:
            self.tree.insert("", "end", values=(row["class"], f"{row['score']:0.6f}"))


def launch_gui() -> None:
    root = tk.Tk()
    app = PatchTesterGUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_gui()
