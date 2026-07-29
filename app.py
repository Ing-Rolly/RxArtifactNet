import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import threading
import json
import os
import sys
import subprocess
import datetime
import time
from pathlib import Path

# Patch pour l'erreur 'usedforsecurity'
import hashlib
_original_md5 = hashlib.md5
def patched_md5(*args, **kwargs):
    kwargs.pop('usedforsecurity', None)
    return _original_md5(*args, **kwargs)
hashlib.md5 = patched_md5

# PIL
try:
    from PIL import Image, ImageTk
    PIL_AVAILABLE = True
except ImportError:
    PIL_AVAILABLE = False

# Chemins
BASE_DIR = Path(__file__).parent.resolve()
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"
OUTPUTS_DIR = BASE_DIR / "outputs"
HIST_DIR = BASE_DIR / "historique"

for d in (OUTPUTS_DIR, HIST_DIR):
    d.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE_DIR))

# Import des modules du projet
from knowledge_base import CLASS_NAMES, get_artifact_info
from pipeline import RxAnalysisPipeline
from diagnostic_engine import DiagnosticEngine
from predictive_engine import PredictiveEngine
from report_niveau2 import DiagnosticReport
from report_niveau3 import PredictiveReport
from component_knowledge import COMPONENTS

# Couleurs
C = {
    "bg": "#10161d", "bg2": "#1b2530", "bg3": "#29343f",
    "border": "#34495e", "blue": "#3ba7b0", "blue_lt": "#6fc4cb",
    "text": "#eceff1", "text2": "#a3b1bd", "text3": "#71808f",
    "green": "#5cb85c", "orange": "#f0ad4e", "red": "#d9534f",
    "yellow": "#f0ad4e", "purple": "#8e7cc3", "white": "#ffffff",
}
URGENCE_COLORS = {"critique": C["purple"], "haute": C["red"], "moyenne": C["yellow"],
                  "faible": C["green"], "aucune": C["blue"]}
HORIZON_COLORS = {"IMMÉDIAT": C["red"], "URGENT": C["orange"], "PLANIFIÉ": C["yellow"],
                  "SURVEILLANCE": C["blue"], "NORMAL": C["green"]}
ALERTE_COLORS = {"CRITIQUE": C["red"], "ROUGE": C["orange"], "ORANGE": C["yellow"],
                 "JAUNE": C["blue"], "VERT": C["green"]}

# --- Widgets personnalisés ---
class RoundedButton(tk.Canvas):
    def __init__(self, parent, text, command=None, bg=C["blue"], fg=C["white"],
                 hover_bg=None, width=160, height=38, font_size=10, **kwargs):
        super().__init__(parent, width=width, height=height,
                         bg=parent["bg"] if hasattr(parent, "__getitem__") else C["bg"],
                         highlightthickness=0, **kwargs)
        self.command = command
        self.bg = bg
        self.fg = fg
        self.hover_bg = hover_bg or self._lighten(bg)
        self.text = text
        self.font_size = font_size
        self._draw()
        self.bind("<Enter>", self._on_enter)
        self.bind("<Leave>", self._on_leave)
        self.bind("<Button-1>", self._on_click)

    def _lighten(self, hex_color):
        try:
            r = int(hex_color[1:3], 16)
            g = int(hex_color[3:5], 16)
            b = int(hex_color[5:7], 16)
            r = min(255, r + 30)
            g = min(255, g + 30)
            b = min(255, b + 30)
            return f"#{r:02x}{g:02x}{b:02x}"
        except Exception:
            return hex_color

    def _draw(self, bg=None):
        self.delete("all")
        bg = bg or self.bg
        w, h = int(self["width"]), int(self["height"])
        r = h // 2
        self.create_arc(0, 0, r*2, h, start=90, extent=180, fill=bg, outline=bg)
        self.create_arc(w-r*2, 0, w, h, start=-90, extent=180, fill=bg, outline=bg)
        self.create_rectangle(r, 0, w-r, h, fill=bg, outline=bg)
        self.create_text(w//2, h//2, text=self.text, fill=self.fg,
                         font=("Segoe UI", self.font_size, "bold"))

    def _on_enter(self, e): self._draw(self.hover_bg)
    def _on_leave(self, e): self._draw(self.bg)
    def _on_click(self, e):
        if self.command:
            self.command()
    def config_text(self, text):
        self.text = text
        self._draw()

class ScrollableFrame(tk.Frame):
    # On conserve cette classe pour les onglets (résultats), mais on ne l'utilise plus dans LeftPanel
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        self.canvas = tk.Canvas(self, bg=self["bg"], highlightthickness=0)
        self.scrollbar = ttk.Scrollbar(self, orient="vertical", command=self.canvas.yview)
        self.inner = tk.Frame(self.canvas, bg=self["bg"])
        self.inner.bind("<Configure>", self._on_configure)
        self.canvas.create_window((0, 0), window=self.inner, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)
        self.scrollbar.pack(side="right", fill="y")
        self.canvas.pack(side="left", fill="both", expand=True)
        self.canvas.bind("<MouseWheel>", self._on_mousewheel)
        self.inner.bind("<MouseWheel>", self._on_mousewheel)

    def _on_configure(self, e):
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))
    def _on_mousewheel(self, e):
        self.canvas.yview_scroll(int(-1 * (e.delta / 120)), "units")

class StatusBar(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["border"], height=28, **kwargs)
        self.pack_propagate(False)
        self._label = tk.Label(self, text="Prêt", bg=C["border"], fg=C["blue_lt"], font=("Segoe UI", 9))
        self._label.pack(side="left", padx=10)
        self._time_label = tk.Label(self, text="", bg=C["border"], fg=C["text3"], font=("Segoe UI", 8))
        self._time_label.pack(side="right", padx=10)
        self._update_time()

    def set(self, msg, color=None):
        self._label.config(text=msg, fg=color or C["blue_lt"])
    def _update_time(self):
        now = datetime.datetime.now().strftime("%H:%M:%S")
        self._time_label.config(text=now)
        self.after(1000, self._update_time)

class ProgressDialog(tk.Toplevel):
    def __init__(self, parent):
        super().__init__(parent)
        self.title("Analyse en cours...")
        self.configure(bg=C["bg2"])
        self.resizable(False, False)
        self.transient(parent)
        self.grab_set()
        w, h = 420, 180
        x = parent.winfo_rootx() + (parent.winfo_width() - w)//2
        y = parent.winfo_rooty() + (parent.winfo_height() - h)//2
        self.geometry(f"{w}x{h}+{x}+{y}")
        tk.Label(self, text="RxArtifactNet — Analyse en cours", bg=C["bg2"], fg=C["blue_lt"],
                 font=("Segoe UI", 11, "bold")).pack(pady=(20,8))
        self._msg = tk.Label(self, text="Initialisation...", bg=C["bg2"], fg=C["text2"],
                             font=("Segoe UI", 9), wraplength=380)
        self._msg.pack(pady=4)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Blue.Horizontal.TProgressbar", troughcolor=C["bg3"], background=C["blue"])
        self._bar = ttk.Progressbar(self, style="Blue.Horizontal.TProgressbar", length=360, mode="determinate", maximum=100)
        self._bar.pack(pady=10, padx=30)
        self._pct = tk.Label(self, text="0%", bg=C["bg2"], fg=C["text3"], font=("Segoe UI", 8))
        self._pct.pack()
    def update_progress(self, pct, msg):
        self._bar["value"] = pct
        self._msg.config(text=msg)
        self._pct.config(text=f"{pct}%")
        self.update()
    def close(self):
        self.grab_release()
        self.destroy()

# --- Panneau gauche (sans barres de défilement) ---
class LeftPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=C["bg2"], width=300, **kwargs)
        self.pack_propagate(False)
        self.app = app
        self._build()

    def _build(self):
        # Logo
        header = tk.Frame(self, bg=C["border"], pady=14)
        header.pack(fill="x")
        tk.Label(header, text="RxArtifactNet", bg=C["border"], fg=C["blue_lt"],
                 font=("Segoe UI", 15, "bold")).pack()
        tk.Label(header, text="Diagnostic Radiologique", bg=C["border"], fg=C["text3"],
                 font=("Segoe UI", 8, "italic")).pack()

        # Aperçu image (hauteur fixe)
        self._img_frame = tk.Frame(self, bg=C["bg"], relief="flat", bd=0, pady=2, height=220)
        self._img_frame.pack(fill="x", padx=12, pady=(12,4))
        self._img_frame.pack_propagate(False)
        self._img_border = tk.Frame(self._img_frame, bg=C["border"], pady=2, padx=2)
        self._img_border.pack(fill="both", expand=True)
        self._img_label = tk.Label(self._img_border,
                                   text="📂\n\nGlissez une image Rx ici\nou cliquez sur Ouvrir\n\nPNG · JPEG · DICOM",
                                   bg=C["bg"], fg=C["text3"], font=("Segoe UI", 9),
                                   justify="center")
        self._img_label.pack(fill="both", expand=True)
        self._img_label.bind("<Button-1>", lambda e: self.app.open_image())

        # Boutons Ouvrir/Effacer
        btn_row = tk.Frame(self, bg=C["bg2"])
        btn_row.pack(fill="x", padx=12, pady=4)
        self.btn_open = RoundedButton(btn_row, "📂 Ouvrir", command=self.app.open_image,
                                      bg=C["bg3"], fg=C["text"], hover_bg=C["border"],
                                      width=128, height=32, font_size=9)
        self.btn_open.pack(side="left")
        self.btn_clear = RoundedButton(btn_row, "✕ Effacer", command=self.app.clear_image,
                                       bg=C["bg3"], fg="#f87171", hover_bg="#450a0a",
                                       width=100, height=32, font_size=9)
        self.btn_clear.pack(side="right")

        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=12, pady=8)

        # --- Paramètres (sans scroll, radios en vertical) ---
        params_frame = tk.LabelFrame(self, text=" Paramètres ", bg=C["bg2"], fg=C["text3"],
                                     font=("Segoe UI", 8), bd=1, relief="solid",
                                     highlightbackground=C["border"])
        params_frame.pack(fill="x", padx=12, pady=4)

        # Champs de saisie
        fields = [("Appareil :", "device", "GE_XR656_Salle_A"),
                  ("Opérateur :", "operator", "")]
        self._vars = {}
        for label, key, default in fields:
            row = tk.Frame(params_frame, bg=C["bg2"])
            row.pack(fill="x", padx=8, pady=3)
            tk.Label(row, text=label, bg=C["bg2"], fg=C["text2"], font=("Segoe UI", 9), width=11, anchor="e").pack(side="left")
            var = tk.StringVar(value=default)
            self._vars[key] = var
            entry = tk.Entry(row, textvariable=var, bg=C["bg"], fg=C["text"], insertbackground=C["text"],
                             font=("Segoe UI", 9), relief="flat", bd=0,
                             highlightthickness=1, highlightbackground=C["border"], highlightcolor=C["blue"])
            entry.pack(side="left", fill="x", expand=True, padx=(4,0))

        # Niveau d'analyse (vertical)
        lvl_label = tk.Label(params_frame, text="Niveau :", bg=C["bg2"], fg=C["text2"], font=("Segoe UI", 9), anchor="w")
        lvl_label.pack(fill="x", padx=8, pady=(3,0))
        self._level_var = tk.StringVar(value="3")
        level_frame = tk.Frame(params_frame, bg=C["bg2"])
        level_frame.pack(fill="x", padx=8, pady=(0,8))
        for txt, val in [("N1 — Détection", "1"), ("N2 — +Diagnostic", "2"), ("N3 — +Prédiction", "3")]:
            rb = tk.Radiobutton(level_frame, text=txt, variable=self._level_var, value=val,
                                bg=C["bg2"], fg=C["text2"], selectcolor=C["bg"], activebackground=C["bg2"],
                                anchor="w")
            rb.pack(fill="x", pady=2)

        # Séparateur
        tk.Frame(self, bg=C["border"], height=1).pack(fill="x", padx=12, pady=8)

        # Bouton ANALYSER
        self.btn_analyze = RoundedButton(self, "🔬 ANALYSER", command=self.app.launch_analysis,
                                         bg=C["blue"], fg=C["white"], hover_bg="#3b82f6",
                                         width=250, height=40, font_size=13)
        self.btn_analyze.pack(pady=4)

        # Signature en bas
        tk.Label(self, text="TAMEZA ROLLY MIDELE\nUniv. de Dschang — 2026",
                 bg=C["bg2"], fg=C["text3"], font=("Segoe UI", 7, "italic"),
                 justify="center").pack(side="bottom", pady=8, fill="x")

    def set_image_preview(self, path):
        if not PIL_AVAILABLE:
            self._img_label.config(text=f"✓ {Path(path).name}", fg=C["green"])
        else:
            try:
                img = Image.open(path).convert("L")
                max_w = self._img_frame.winfo_width() - 20
                max_h = self._img_frame.winfo_height() - 20
                if max_w <= 0: max_w = 240
                if max_h <= 0: max_h = 180
                img.thumbnail((max_w, max_h), Image.LANCZOS)
                self._preview_img = ImageTk.PhotoImage(img)
                self._img_label.config(image=self._preview_img, text="", bg=C["bg"])
            except Exception:
                self._img_label.config(text=f"✓ {Path(path).name}", fg=C["green"])
        self._img_frame.update_idletasks()
        self._img_border.update_idletasks()

    def clear_preview(self):
        self._img_label.config(image="", text="📂\n\nGlissez une image Rx ici\nou cliquez sur Ouvrir\n\nPNG · JPEG · DICOM", fg=C["text3"])
        self._img_frame.update_idletasks()

    def get_params(self):
        return {"device": self._vars["device"].get().strip() or "Appareil_RX_01",
                "patient": "PAT_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
                "operator": self._vars["operator"].get().strip() or "Automatique",
                "level": int(self._level_var.get())}

# --- Panneau de droite (Visualisation + Rapports) ---
class RightPanel(tk.Frame):
    def __init__(self, parent, app, **kwargs):
        super().__init__(parent, bg=C["bg2"], width=360, **kwargs)
        self.pack_propagate(False)
        self.app = app
        self._build()

    def _build(self):
        # Zone de visualisation (hauteur réduite) — défilement horizontal
        # pour afficher une image par artefact détecté
        self.viz_frame = tk.LabelFrame(self, text=" Visualisation ", bg=C["bg2"], fg=C["text3"],
                                       font=("Segoe UI", 8), bd=1, relief="solid",
                                       highlightbackground=C["border"])
        self.viz_frame.pack(fill="both", expand=True, padx=8, pady=(8,4))

        self._canvas = tk.Canvas(self.viz_frame, bg=C["bg2"], highlightthickness=0)
        self._hscroll = ttk.Scrollbar(self.viz_frame, orient="horizontal",
                                      command=self._canvas.xview,
                                      style="Horizontal.TScrollbar")
        self._canvas.configure(xscrollcommand=self._hscroll.set)
        self._canvas.pack(side="top", fill="both", expand=True, padx=4, pady=(4,0))
        self._hscroll.pack(side="bottom", fill="x", padx=4, pady=(2,4))
        # Molette de la souris = défilement horizontal (plus pratique que la barre seule)
        self._canvas.bind("<MouseWheel>", lambda e: self._canvas.xview_scroll(int(-e.delta/60), "units"))

        self._fig_imgs = []  # garde les références PhotoImage (sinon garbage collect)
        self._placeholder_text = ("🖼️\n\nLa figure d'analyse\ns'affichera ici\n"
                                  "après l'analyse\n\n(une image par artefact détecté,\n"
                                  "défilement horizontal si plusieurs)")
        self._show_placeholder(self._placeholder_text, C["text3"])

        self._info = tk.Label(self.viz_frame, text="", bg=C["bg2"], fg=C["text3"],
                              font=("Segoe UI", 8), wraplength=320, justify="center")
        self._info.pack(side="bottom", pady=2)

        # Zone Rapports générés
        rpt_frame = tk.LabelFrame(self, text=" Rapports générés ", bg=C["bg2"], fg=C["text3"],
                                  font=("Segoe UI", 8), bd=1, relief="solid",
                                  highlightbackground=C["border"])
        rpt_frame.pack(fill="x", padx=8, pady=(4,8))
        self.btn_r1 = RoundedButton(rpt_frame, "📄 Rapport Niveau 1", command=lambda: self.app.open_report(1),
                                    bg=C["bg3"], fg=C["blue_lt"], hover_bg=C["border"],
                                    width=250, height=30, font_size=9)
        self.btn_r1.pack(pady=3)
        self.btn_r2 = RoundedButton(rpt_frame, "📊 Rapport Niveau 2", command=lambda: self.app.open_report(2),
                                    bg=C["bg3"], fg=C["blue_lt"], hover_bg=C["border"],
                                    width=250, height=30, font_size=9)
        self.btn_r2.pack(pady=3)
        self.btn_r3 = RoundedButton(rpt_frame, "📈 Rapport Niveau 3", command=lambda: self.app.open_report(3),
                                    bg=C["bg3"], fg=C["blue_lt"], hover_bg=C["border"],
                                    width=250, height=30, font_size=9)
        self.btn_r3.pack(pady=(3,8))

    def _show_placeholder(self, text, color):
        self._canvas.delete("all")
        self._canvas.create_text(150, 110, text=text, fill=color,
                                 font=("Segoe UI", 10, "italic"), justify="center")
        self._canvas.configure(scrollregion=(0, 0, 300, 220))

    def show_figures(self, fig_paths):
        """
        Affiche une image par artefact détecté, côte à côte, avec défilement
        horizontal si plusieurs images (barre de défilement + molette souris).
        """
        self._canvas.delete("all")
        self._fig_imgs = []

        if not fig_paths:
            self._show_placeholder("Visualisation indisponible", C["text3"])
            self._info.config(text="")
            return
        if not PIL_AVAILABLE:
            self._show_placeholder("PIL non disponible pour l'affichage", C["text3"])
            return

        target_h = max(200, self.viz_frame.winfo_height() - 40)
        x_offset = 6
        names = []
        for path in fig_paths:
            if not path or not Path(path).exists():
                continue
            try:
                img = Image.open(path)
                ratio = target_h / img.height
                w = max(1, int(img.width * ratio))
                img = img.resize((w, target_h), Image.LANCZOS)
                photo = ImageTk.PhotoImage(img)
                self._fig_imgs.append(photo)
                self._canvas.create_image(x_offset, 6, anchor="nw", image=photo)
                x_offset += w + 10
                names.append(Path(path).name)
            except Exception as e:
                print(f"Erreur affichage {path}: {e}")

        self._canvas.configure(scrollregion=(0, 0, max(x_offset, 300), target_h + 12))
        n = len(self._fig_imgs)
        if n == 0:
            self._show_placeholder("Aucune image valide à afficher", C["text3"])
            self._info.config(text="")
        elif n == 1:
            self._info.config(text=names[0])
        else:
            self._info.config(text=f"{n} artefact(s) — {' ◀ ▶ '} défilez horizontalement pour tout voir")

    def show_figure(self, fig_path):
        """Compatibilité ascendante : affiche une seule image."""
        self.show_figures([fig_path] if fig_path else [])

    def clear(self):
        self._show_placeholder(self._placeholder_text, C["text3"])
        self._info.config(text="")

# --- Panneau résultats (identique, sans modifications) ---
class ResultsPanel(tk.Frame):
    def __init__(self, parent, **kwargs):
        super().__init__(parent, bg=C["bg"], **kwargs)
        self._build()

    def _build(self):
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Dark.TNotebook", background=C["bg"], borderwidth=0)
        style.configure("Dark.TNotebook.Tab", background=C["bg2"], foreground=C["text3"],
                        padding=[14,6], font=("Segoe UI",9))
        style.map("Dark.TNotebook.Tab", background=[("selected", C["border"])], foreground=[("selected", C["blue_lt"])])

        self._notebook = ttk.Notebook(self, style="Dark.TNotebook")
        self._notebook.pack(fill="both", expand=True)

        self._tab_det = ScrollableFrame(self._notebook, bg=C["bg"])
        self._notebook.add(self._tab_det, text="🔍 Artefacts")
        self._det_placeholder = tk.Label(self._tab_det.inner,
            text="Aucune analyse effectuée.\nOuvrez une image et cliquez sur Analyser.",
            bg=C["bg"], fg=C["text3"], font=("Segoe UI",10,"italic"), justify="center")
        self._det_placeholder.pack(pady=60)

        self._tab_comp = tk.Frame(self._notebook, bg=C["bg"])
        self._notebook.add(self._tab_comp, text="⚙️ Composants")
        self._build_composants_table()

        self._tab_alerte = ScrollableFrame(self._notebook, bg=C["bg"])
        self._notebook.add(self._tab_alerte, text="🚨 Alerte")
        tk.Label(self._tab_alerte.inner, text="L'alerte globale s'affichera après l'analyse.",
                 bg=C["bg"], fg=C["text3"], font=("Segoe UI",10,"italic")).pack(pady=60)

        self._tab_log = tk.Frame(self._notebook, bg=C["bg"])
        self._notebook.add(self._tab_log, text="📋 Journal")
        self._build_log()

    def _build_composants_table(self):
        cols = ("Composant", "Sous-système", "Score", "Horizon", "P(30j)", "Tendance")
        style = ttk.Style()
        style.configure("Dark.Treeview", background=C["bg"], foreground=C["text"],
                        fieldbackground=C["bg"], borderwidth=0, font=("Segoe UI",9), rowheight=28)
        style.configure("Dark.Treeview.Heading", background=C["border"], foreground=C["blue_lt"],
                        font=("Segoe UI",9,"bold"), relief="flat")
        style.map("Dark.Treeview", background=[("selected", C["border"])], foreground=[("selected", C["blue_lt"])])
        scroll_y = ttk.Scrollbar(self._tab_comp, orient="vertical")
        scroll_x = ttk.Scrollbar(self._tab_comp, orient="horizontal")
        self._tree = ttk.Treeview(self._tab_comp, columns=cols, show="headings", style="Dark.Treeview",
                                  yscrollcommand=scroll_y.set, xscrollcommand=scroll_x.set)
        widths = [180,160,70,100,70,120]
        for col,w in zip(cols,widths):
            self._tree.heading(col, text=col)
            self._tree.column(col, width=w, minwidth=w, anchor="center")
        scroll_y.config(command=self._tree.yview)
        scroll_x.config(command=self._tree.xview)
        scroll_y.pack(side="right", fill="y")
        scroll_x.pack(side="bottom", fill="x")
        self._tree.pack(fill="both", expand=True)
        self._tree.tag_configure("even", background=C["bg"])
        self._tree.tag_configure("odd", background=C["bg2"])
        self._tree.tag_configure("crit", foreground=C["red"])
        self._tree.tag_configure("high", foreground=C["orange"])
        self._tree.tag_configure("med", foreground=C["yellow"])
        self._tree.tag_configure("ok", foreground=C["green"])

    def _build_log(self):
        scroll = ttk.Scrollbar(self._tab_log, orient="vertical")
        self._log_text = tk.Text(self._tab_log, bg="#020617", fg=C["text3"], font=("Consolas",8),
                                 relief="flat", bd=0, state="disabled", wrap="word",
                                 yscrollcommand=scroll.set, insertbackground=C["text"])
        scroll.config(command=self._log_text.yview)
        scroll.pack(side="right", fill="y")
        self._log_text.pack(fill="both", expand=True, padx=4, pady=4)

    def log(self, msg):
        self._log_text.config(state="normal")
        ts = datetime.datetime.now().strftime("%H:%M:%S")
        self._log_text.insert("end", f"[{ts}] {msg}\n")
        self._log_text.see("end")
        self._log_text.config(state="disabled")

    def clear_log(self):
        self._log_text.config(state="normal")
        self._log_text.delete("1.0", "end")
        self._log_text.config(state="disabled")

    def show_detections(self, detections):
        for w in self._tab_det.inner.winfo_children():
            w.destroy()
        if not detections:
            tk.Label(self._tab_det.inner, text="✅ Aucun artefact machine détecté\nAppareil conforme.",
                     bg=C["bg"], fg=C["green"], font=("Segoe UI",12,"bold"), justify="center").pack(pady=60)
            return
        for det in detections:
            self._build_artifact_card(self._tab_det.inner, det)
        self._notebook.select(0)

    def _build_artifact_card(self, parent, det):
        urgence = det.get("urgence","faible")
        color = URGENCE_COLORS.get(urgence, C["blue"])
        card = tk.Frame(parent, bg=C["bg2"], bd=0)
        card.pack(fill="x", padx=12, pady=5)
        tk.Frame(card, bg=color, width=5).pack(side="left", fill="y")
        content = tk.Frame(card, bg=C["bg2"])
        content.pack(side="left", fill="both", expand=True, padx=10, pady=8)
        top_row = tk.Frame(content, bg=C["bg2"])
        top_row.pack(fill="x")
        tk.Label(top_row, text=det.get("label",""), bg=C["bg2"], fg=C["text"], font=("Segoe UI",10,"bold")).pack(side="left")
        tk.Label(top_row, text=f"{det['confidence']:.1f}%", bg=C["bg2"], fg=color, font=("Segoe UI",10,"bold")).pack(side="right")
        comp = det.get("composant","—")
        tk.Label(content, text=f"⚙️ {comp}", bg=C["bg2"], fg=C["text2"], font=("Segoe UI",8), wraplength=400, anchor="w").pack(fill="x", pady=1)
        zone = det.get("zone","—")
        tk.Label(content, text=f"📍 {zone}", bg=C["bg2"], fg=C["text3"], font=("Segoe UI",8), anchor="w").pack(fill="x")
        actions = det.get("maintenance", [])
        if actions:
            tk.Label(content, text="Actions :", bg=C["bg2"], fg=C["blue_lt"], font=("Segoe UI",8,"bold")).pack(anchor="w", pady=(4,1))
            for act in actions[:3]:
                tk.Label(content, text=f"   → {act}", bg=C["bg2"], fg=C["blue_lt"], font=("Segoe UI",8), wraplength=400, anchor="w").pack(fill="x")
        tk.Frame(card, bg=C["border"], height=1).pack(fill="x", side="bottom")

    def show_composants(self, composants):
        for row in self._tree.get_children():
            self._tree.delete(row)
        for i,c in enumerate(composants):
            score = c.get("score_risque",0)
            hz = c.get("horizon","NORMAL")
            prob = c.get("prob_panne_30j",0)
            if score >= 85: tag = "crit"
            elif score >= 65: tag = "high"
            elif score >= 45: tag = "med"
            else: tag = "ok"
            row_tag = ("even" if i%2==0 else "odd", tag)
            self._tree.insert("", "end", tags=row_tag, values=(
                c.get("nom","—"), c.get("sous_systeme","—"),
                f"{score:.0f}/100", hz, f"{prob:.1f}%", c.get("tendance","—")))
        self._notebook.select(1)

    def show_alerte(self, alerte, score_global):
        for w in self._tab_alerte.inner.winfo_children():
            w.destroy()
        niveau = alerte.get("niveau","VERT")
        emoji = alerte.get("emoji","🟢")
        color = ALERTE_COLORS.get(niveau, C["green"])
        banner = tk.Frame(self._tab_alerte.inner, bg=color)
        banner.pack(fill="x", padx=16, pady=16)
        tk.Label(banner, text=f"{emoji}  ALERTE {niveau}", bg=color, fg=C["white"],
                 font=("Segoe UI",18,"bold"), justify="center").pack(pady=(14,4))
        tk.Label(banner, text=f"Score de risque global : {score_global:.1f} / 100",
                 bg=color, fg=C["white"], font=("Segoe UI",12)).pack(pady=4)
        action = alerte.get("delai_intervention","")
        tk.Label(banner, text=action, bg=color, fg="#fef9c3", font=("Segoe UI",10,"italic"), wraplength=500).pack(pady=(4,14))
        desc = alerte.get("description","")
        if desc:
            tk.Label(self._tab_alerte.inner, text=desc, bg=C["bg"], fg=C["text2"],
                     font=("Segoe UI",10), wraplength=500, justify="center").pack(pady=12)
        self._notebook.select(2)

# --- Application principale ---
class LoginDialog(tk.Toplevel):
    """
    Fenêtre de connexion — bloque l'accès à l'application tant que les
    identifiants d'un technicien valide n'ont pas été saisis.
    Identifiants stockés dans technicians.json (mots de passe hashés SHA-256,
    jamais en clair). Un compte par défaut (admin/admin123) est créé au
    premier lancement si le fichier n'existe pas encore.
    """
    CREDS_PATH = BASE_DIR / "technicians.json"

    def __init__(self, parent):
        super().__init__(parent)
        self.title("Connexion — RxArtifactNet")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.result = None  # nom du technicien si succès, None si annulé

        self._ensure_default_account()

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.transient(parent)
        self.grab_set()

        frame = tk.Frame(self, bg=C["bg"], padx=30, pady=25)
        frame.pack()

        tk.Label(frame, text="🔒 RxArtifactNet", font=("Segoe UI", 16, "bold"),
                 bg=C["bg"], fg=C["blue_lt"]).pack(pady=(0, 4))
        tk.Label(frame, text="Authentification technicien requise",
                 font=("Segoe UI", 9), bg=C["bg"], fg=C["text3"]).pack(pady=(0, 18))

        tk.Label(frame, text="Nom du technicien :", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 9), anchor="w").pack(fill="x")
        self._user_var = tk.StringVar()
        user_entry = tk.Entry(frame, textvariable=self._user_var, font=("Segoe UI", 10),
                              bg=C["bg2"], fg=C["text"], insertbackground=C["text"],
                              relief="flat")
        user_entry.pack(fill="x", pady=(2, 12), ipady=4)
        user_entry.focus_set()

        tk.Label(frame, text="Mot de passe :", bg=C["bg"], fg=C["text"],
                 font=("Segoe UI", 9), anchor="w").pack(fill="x")
        self._pwd_var = tk.StringVar()
        pwd_entry = tk.Entry(frame, textvariable=self._pwd_var, show="•",
                             font=("Segoe UI", 10), bg=C["bg2"], fg=C["text"],
                             insertbackground=C["text"], relief="flat")
        pwd_entry.pack(fill="x", pady=(2, 4), ipady=4)

        self._err_label = tk.Label(frame, text="", bg=C["bg"], fg=C["red"],
                                   font=("Segoe UI", 8))
        self._err_label.pack(fill="x", pady=(0, 10))

        btn = RoundedButton(frame, "Se connecter", command=self._try_login,
                            bg=C["blue"], fg=C["white"], hover_bg=C["blue_lt"],
                            width=260, height=36, font_size=10)
        btn.pack(pady=(4, 0))

        user_entry.bind("<Return>", lambda e: pwd_entry.focus_set())
        pwd_entry.bind("<Return>", lambda e: self._try_login())

        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.wait_window(self)

    def _ensure_default_account(self):
        if not self.CREDS_PATH.exists():
            default_hash = hashlib.sha256("admin123".encode()).hexdigest()
            data = {"admin": {"password_hash": default_hash, "role": "administrateur"}}
            with open(self.CREDS_PATH, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)

    def _try_login(self):
        name = self._user_var.get().strip()
        pwd = self._pwd_var.get()
        if not name or not pwd:
            self._err_label.config(text="Veuillez renseigner les deux champs.")
            return
        try:
            with open(self.CREDS_PATH, "r", encoding="utf-8") as f:
                creds = json.load(f)
        except Exception:
            creds = {}
        entry = creds.get(name)
        if entry and entry.get("password_hash") == hashlib.sha256(pwd.encode()).hexdigest():
            self.result = name
            self.destroy()
        else:
            self._err_label.config(text="Nom ou mot de passe incorrect.")

    def _on_cancel(self):
        self.result = None
        self.destroy()


class DeviceInfoDialog(tk.Toplevel):
    """
    Formulaire des caractéristiques de l'appareil, affiché juste après une
    connexion réussie. Bloque jusqu'à validation (Marque et Modèle requis).
    """
    def __init__(self, parent, technicien):
        super().__init__(parent)
        self.title("Caractéristiques de l'appareil — RxArtifactNet")
        self.configure(bg=C["bg"])
        self.resizable(False, False)
        self.result = None

        self.protocol("WM_DELETE_WINDOW", self._on_cancel)
        self.transient(parent)
        self.grab_set()

        frame = tk.Frame(self, bg=C["bg"], padx=30, pady=22)
        frame.pack()

        tk.Label(frame, text="🩻 Caractéristiques de l'appareil",
                 font=("Segoe UI", 13, "bold"), bg=C["bg"], fg=C["blue_lt"]).pack(pady=(0, 3))
        tk.Label(frame, text=f"Technicien connecté : {technicien}",
                 font=("Segoe UI", 8, "italic"), bg=C["bg"], fg=C["text3"]).pack(pady=(0, 16))

        self._vars = {}
        fields = [
            ("Marque :",                   "marque",       "GE Healthcare"),
            ("Modèle :",                   "modele",       "Discovery XR656"),
            ("Numéro de série :",          "numero_serie", ""),
            ("Localisation (salle) :",     "localisation", "Salle A"),
            ("Année de mise en service :", "annee",        ""),
        ]
        for label, key, default in fields:
            row = tk.Frame(frame, bg=C["bg"])
            row.pack(fill="x", pady=4)
            tk.Label(row, text=label, bg=C["bg"], fg=C["text2"], font=("Segoe UI", 9),
                     width=22, anchor="w").pack(side="left")
            var = tk.StringVar(value=default)
            entry = tk.Entry(row, textvariable=var, font=("Segoe UI", 9), bg=C["bg2"],
                             fg=C["text"], insertbackground=C["text"], relief="flat")
            entry.pack(side="left", fill="x", expand=True, ipady=3)
            self._vars[key] = var

        btn = RoundedButton(frame, "Valider et continuer", command=self._validate,
                            bg=C["blue"], fg=C["white"], hover_bg=C["blue_lt"],
                            width=280, height=36, font_size=10)
        btn.pack(pady=(18, 0))

        self.update_idletasks()
        w, h = self.winfo_reqwidth(), self.winfo_reqheight()
        sw, sh = self.winfo_screenwidth(), self.winfo_screenheight()
        self.geometry(f"{w}x{h}+{(sw-w)//2}+{(sh-h)//2}")

        self.wait_window(self)

    def _validate(self):
        marque = self._vars["marque"].get().strip()
        modele = self._vars["modele"].get().strip()
        if not marque or not modele:
            messagebox.showwarning("Champs requis", "Marque et modèle sont obligatoires.",
                                   parent=self)
            return
        self.result = {k: v.get().strip() for k, v in self._vars.items()}
        self.destroy()

    def _on_cancel(self):
        self.result = None
        self.destroy()


class RxArtifactNetApp:
    def __init__(self, root, technicien=None, device_info=None):
        self.root = root
        self.technicien = technicien or "Inconnu"
        self.device_info = device_info or {}
        self._image_path = None
        self._reports = {1: None, 2: None, 3: None}
        self._last_result = {}
        self._running = False
        self._setup_window()
        self._build_layout()
        self._setup_menu()
        self._apply_device_info()
        self._log(f"Connecté en tant que : {self.technicien}")
        self._log("Bienvenue dans RxArtifactNet v1.0")
        self._log("Ouvrez une image Rx thorax pour commencer.")

    def _apply_device_info(self):
        """Pré-remplit les champs Appareil/Opérateur avec les informations
        saisies lors de la connexion et de la fiche appareil."""
        if self.device_info:
            marque = self.device_info.get("marque", "")
            modele = self.device_info.get("modele", "")
            loc    = self.device_info.get("localisation", "")
            device_str = " ".join(x for x in [marque, modele, loc] if x)
            if device_str:
                self._left._vars["device"].set(device_str)
        if self.technicien:
            self._left._vars["operator"].set(self.technicien)

    def _setup_window(self):
        self.root.title(f"RxArtifactNet — Diagnostic Radiologique — {self.technicien}")
        self.root.configure(bg=C["bg"])
        self.root.minsize(1100, 700)
        sw = self.root.winfo_screenwidth()
        sh = self.root.winfo_screenheight()
        w, h = int(sw*0.85), int(sh*0.85)
        x, y = (sw-w)//2, (sh-h)//2
        self.root.geometry(f"{w}x{h}+{x}+{y}")
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _build_layout(self):
        self._status = StatusBar(self.root)
        self._status.pack(side="bottom", fill="x")
        main = tk.Frame(self.root, bg=C["bg"])
        main.pack(fill="both", expand=True)

        # Panneau gauche (300px)
        self._left = LeftPanel(main, app=self)
        self._left.pack(side="left", fill="y")

        # Séparateur
        tk.Frame(main, bg=C["border"], width=2).pack(side="left", fill="y")

        # Panneau droit (visualisation + rapports) - packé AVANT le centre
        # pour que sa largeur fixe soit respectée (sinon le centre, avec
        # expand=True, absorbe tout l'espace disponible en premier)
        self._right = RightPanel(main, app=self)
        self._right.pack(side="right", fill="y")

        # Séparateur
        tk.Frame(main, bg=C["border"], width=2).pack(side="right", fill="y")

        # Panneau central (résultats) - prend tout l'espace restant
        self._results = ResultsPanel(main)
        self._results.pack(side="left", fill="both", expand=True)

    def _setup_menu(self):
        menubar = tk.Menu(self.root, bg=C["bg2"], fg=C["text"], activebackground=C["border"], activeforeground=C["blue_lt"], relief="flat", bd=0)
        file_menu = tk.Menu(menubar, tearoff=0, bg=C["bg2"], fg=C["text"], activebackground=C["border"], activeforeground=C["blue_lt"])
        file_menu.add_command(label="Ouvrir une image...", accelerator="Ctrl+O", command=self.open_image)
        file_menu.add_separator()
        file_menu.add_command(label="Exporter résultats (JSON)", command=self.export_results)
        file_menu.add_separator()
        file_menu.add_command(label="Quitter", accelerator="Ctrl+Q", command=self._on_close)
        menubar.add_cascade(label="Fichier", menu=file_menu)
        ana_menu = tk.Menu(menubar, tearoff=0, bg=C["bg2"], fg=C["text"], activebackground=C["border"], activeforeground=C["blue_lt"])
        ana_menu.add_command(label="Lancer l'analyse", accelerator="F5", command=self.launch_analysis)
        menubar.add_cascade(label="Analyse", menu=ana_menu)
        rpt_menu = tk.Menu(menubar, tearoff=0, bg=C["bg2"], fg=C["text"], activebackground=C["border"], activeforeground=C["blue_lt"])
        for i in range(1,4):
            rpt_menu.add_command(label=f"Rapport Niveau {i}", command=lambda n=i: self.open_report(n))
        menubar.add_cascade(label="Rapports", menu=rpt_menu)
        help_menu = tk.Menu(menubar, tearoff=0, bg=C["bg2"], fg=C["text"], activebackground=C["border"], activeforeground=C["blue_lt"])
        help_menu.add_command(label="À propos", command=self._show_about)
        menubar.add_cascade(label="Aide", menu=help_menu)
        self.root.config(menu=menubar)
        self.root.bind("<Control-o>", lambda e: self.open_image())
        self.root.bind("<Control-q>", lambda e: self._on_close())
        self.root.bind("<F5>", lambda e: self.launch_analysis())

    def open_image(self):
        path = filedialog.askopenfilename(title="Ouvrir une image radiographique",
            filetypes=[("Images médicales", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff *.dcm *.dicom"), ("Tous", "*.*")])
        if path:
            self.load_image(path)

    def load_image(self, path):
        self._image_path = path
        self._left.set_image_preview(path)
        self._status.set(f"Image chargée : {Path(path).name}", C["green"])
        self._log(f"Image chargée : {path}")

    def clear_image(self):
        self._image_path = None
        self._left.clear_preview()
        self._right.clear()
        self._reports = {1: None, 2: None, 3: None}
        self._status.set("Image effacée")
        self._log("Image effacée.")

    def launch_analysis(self):
        if not self._image_path:
            messagebox.showwarning("Image manquante", "Veuillez d'abord ouvrir une image radiographique.")
            return
        if self._running:
            return
        params = self._left.get_params()
        self._running = True
        self._results.clear_log()
        self._log(f"Démarrage analyse Niveau {params['level']} ...")
        self._status.set("Analyse en cours...", C["yellow"])
        self._progress = ProgressDialog(self.root)
        t = threading.Thread(target=self._run_analysis, args=(params,), daemon=True)
        t.start()

    def _run_analysis(self, params):
        try:
            ts = time.time()
            level = params["level"]

            def prog(step, total, msg):
                pct = int(step/total*100) if total else 0
                self.root.after(0, self._progress.update_progress, pct, msg)
                self.root.after(0, self._log, f"  [{pct}%] {msg}")

            prog(1,10,"Chargement du modèle...")
            #weights = MODELS_DIR / "best_model.pkl"
            #weights = BASE_DIR / "trained_model_simple" / "best_model.pkl"
            weights = BASE_DIR / "trained_model_simple" / "model_weights.pkl"
            print("Poids trouvés :", weights.exists(), "->", weights)
            pipeline = RxAnalysisPipeline(weights_path=str(weights) if weights.exists() else None,
                                          output_dir=str(OUTPUTS_DIR), seg_threshold=0.30, cls_threshold=0.12)
            prog(3,10,"Prétraitement...")
            prog(5,10,"Inférence...")
            result_n1 = pipeline.analyze(image_path=self._image_path, patient_id=params["patient"],
                                         device_name=params["device"], operator=params["operator"],
                                         generate_report=True)
            result = {"success": True, "level": level, "n_artefacts": len(result_n1["detections"]),
                      "detections": self._format_detections(result_n1["detections"]),
                      "figure_path": result_n1.get("figure_path",""),
                      "figure_paths": result_n1.get("figure_paths",[]),
                      "report_path": result_n1.get("report_path",""),
                      "analysis_time": round(time.time()-ts,2)}

            if level >= 2:
                prog(7,10,"Diagnostic composant...")
                dev_id = params["device"].replace(" ","_")
                engine = DiagnosticEngine(device_id=dev_id, history_dir=str(HIST_DIR))
                diag = engine.diagnose(result_n1["detections"], save_history=True)
                meta = {"patient_id": params["patient"], "device": params["device"],
                        "date": datetime.date.today().strftime("%d/%m/%Y"), "operator": params["operator"]}
                r2_path = str(OUTPUTS_DIR / f"{params['patient']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_N2.pdf")
                DiagnosticReport(r2_path).build(diagnosis=diag, meta=meta, history_summary=engine.get_device_history_summary())
                result.update({"score_global": diag["score_global"], "horizon": diag["horizon_global"]["label"],
                               "horizon_action": diag["horizon_global"]["action"],
                               "composants": self._format_composants(diag), "report_n2_path": r2_path})

            if level >= 3:
                prog(9,10,"Prédiction temporelle...")
                pred_engine = PredictiveEngine(device_id=params["device"].replace(" ","_"), history_dir=str(HIST_DIR),
                                               alert_log=str(OUTPUTS_DIR / "alertes.json"))
                prediction = pred_engine.predict(current_diagnosis=diag, component_knowledge=COMPONENTS)
                r3_path = str(OUTPUTS_DIR / f"{params['patient']}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}_N3.pdf")
                from report_niveau3 import PredictiveReport
                PredictiveReport(r3_path).build(prediction=prediction, meta={"device": params["device"],
                                              "date": datetime.date.today().strftime("%d/%m/%Y")})
                result.update({"alerte_globale": prediction.get("alerte_globale", {}),
                               "n_alertes_actives": prediction.get("n_alertes_actives",0), "report_n3_path": r3_path})

            prog(10,10,"Terminé.")
            result["analysis_time"] = round(time.time()-ts,2)
        except Exception as e:
            import traceback
            result = {"success": False, "error": str(e), "traceback": traceback.format_exc()}
        self.root.after(0, self._on_analysis_done, result)

    def _on_analysis_done(self, result):
        self._running = False
        if hasattr(self,"_progress"): self._progress.close()
        if not result.get("success"):
            err = result.get("error","")
            tb = result.get("traceback","")
            messagebox.showerror("Erreur d'analyse", f"{err}\n\n{tb[:2000]}")
            self._status.set("Erreur : "+err[:60], C["red"])
            self._log(f"ERREUR : {err}")
            return
        self._last_result = result
        level = result.get("level",1)
        self._reports[1] = result.get("report_path","")
        self._reports[2] = result.get("report_n2_path","")
        self._reports[3] = result.get("report_n3_path","")
        self._results.show_detections(result.get("detections",[]))
        if level >= 2 and result.get("composants"):
            self._results.show_composants(result["composants"])
        if level >= 3 and result.get("alerte_globale"):
            self._results.show_alerte(result["alerte_globale"], result.get("score_global",0))
        if result.get("figure_paths"):
            self._right.show_figures(result["figure_paths"])
        elif result.get("figure_path"):
            self._right.show_figure(result["figure_path"])
        msg = f"✓ Analyse terminée — {result.get('n_artefacts',0)} artefact(s) — {result['analysis_time']:.2f}s — Niveau {level}"
        self._status.set(msg, C["green"])
        self._log(msg)

    def open_report(self, level):
        path = self._reports.get(level)
        if not path or not Path(path).exists():
            messagebox.showinfo("Rapport indisponible", f"Le rapport Niveau {level} n'a pas encore été généré.\nLancez d'abord une analyse de niveau {level}.")
            return
        if sys.platform == "win32": os.startfile(path)
        elif sys.platform == "darwin": subprocess.run(["open", path])
        else: subprocess.run(["xdg-open", path])

    def export_results(self):
        if not self._last_result:
            messagebox.showinfo("Aucun résultat", "Effectuez d'abord une analyse.")
            return
        path = filedialog.asksaveasfilename(title="Exporter les résultats", defaultextension=".json",
                                            filetypes=[("JSON","*.json")], initialfile="rxnet_resultats.json")
        if path:
            with open(path,"w",encoding="utf-8") as f:
                json.dump(self._last_result, f, ensure_ascii=False, indent=2)
            messagebox.showinfo("Exporté", f"Résultats exportés :\n{path}")

    def _log(self, msg):
        self._results.log(msg)

    def _format_detections(self, detections):
        try:
            from knowledge_base import get_artifact_info
            return [{"class": d["class"], "label": get_artifact_info(d["class"]).get("label",d["class"]),
                     "confidence": round(d["confidence"]*100,1), "urgence": get_artifact_info(d["class"]).get("urgence","—"),
                     "composant": get_artifact_info(d["class"]).get("composant","—"),
                     "zone": get_artifact_info(d["class"]).get("zone","—"),
                     "maintenance": get_artifact_info(d["class"]).get("maintenance",[])} for d in detections]
        except Exception:
            return detections

    def _format_composants(self, diag):
        return [{"nom": cd["composant_nom"], "sous_systeme": cd["sous_systeme"],
                 "score_risque": cd["score_risque"], "horizon": cd["horizon"]["label"],
                 "prob_panne_30j": cd["prob_panne_30j"], "tendance": cd["tendance"]}
                for cd in diag.get("component_diagnostics",[])]

    def _show_about(self):
        messagebox.showinfo("À propos", "RxArtifactNet v1.0\nDiagnostic automatisé des dysfonctionnements\nen radiographie thoracique\n\nDéveloppé par TAMEZA ROLLY MIDELE\nUniversité de Dschang — 2026")

    def _on_close(self):
        if self._running:
            if not messagebox.askokcancel("Quitter", "Une analyse est en cours. Voulez-vous quitter quand même ?"):
                return
        self.root.destroy()

def main():
    root = tk.Tk()

    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("Vertical.TScrollbar", background=C["bg3"], troughcolor=C["bg2"], borderwidth=0)
    style.configure("Horizontal.TScrollbar", background=C["bg3"], troughcolor=C["bg2"], borderwidth=0)

    # 1. Connexion technicien (obligatoire)
    login = LoginDialog(root)
    if not login.result:
        root.destroy()
        return
    technicien = login.result

    # 2. Fiche caractéristiques de l'appareil (obligatoire)
    device_dialog = DeviceInfoDialog(root, technicien)
    if not device_dialog.result:
        root.destroy()
        return
    device_info = device_dialog.result

    # 3. Interface principale
    app = RxArtifactNetApp(root, technicien=technicien, device_info=device_info)
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception:
        import traceback
        crash_log = BASE_DIR / "crash_log.txt"
        with open(crash_log, "w", encoding="utf-8") as f:
            f.write(f"Crash au démarrage — {datetime.datetime.now()}\n\n")
            f.write(traceback.format_exc())
        try:
            tk.messagebox.showerror(
                "Erreur au démarrage",
                f"L'application n'a pas pu démarrer.\n\n"
                f"Détails enregistrés dans :\n{crash_log}"
            )
        except Exception:
            pass  # même l'affichage de l'erreur a échoué — le fichier suffira
        raise