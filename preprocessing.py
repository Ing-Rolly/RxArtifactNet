"""
preprocessing.py
================
Pipeline de prétraitement — VERSION CORRIGÉE.
Simulation des 12 artefacts machine uniquement.
"""

import numpy as np
import cv2
from PIL import Image
import os
from typing import Tuple, Optional

from knowledge_base import CLASS_NAMES, NUM_CLASSES


def load_image(path: str) -> np.ndarray:
    ext = os.path.splitext(path)[1].lower()
    if ext in (".dcm", ".dicom"):
        try:
            import pydicom
            ds  = pydicom.dcmread(path)
            arr = ds.pixel_array.astype(np.float32)
            if hasattr(ds, "PhotometricInterpretation"):
                if ds.PhotometricInterpretation == "MONOCHROME1":
                    arr = arr.max() - arr
            arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-9) * 255
            return arr.astype(np.uint8)
        except ImportError:
            raise ImportError("pip install pydicom requis pour les fichiers DICOM")

    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        img = np.array(Image.open(path).convert("L"))
    if img is None:
        raise ValueError(f"Impossible de charger : {path}")
    return img


def apply_clahe(img: np.ndarray, clip: float = 2.0) -> np.ndarray:
    return cv2.createCLAHE(clipLimit=clip, tileGridSize=(8, 8)).apply(img)


def normalize(img: np.ndarray) -> np.ndarray:
    f = img.astype(np.float32)
    n = (f - f.mean()) / (f.std() + 1e-9)
    n = (n - n.min()) / (n.max() - n.min() + 1e-9)
    return n.astype(np.float32)


def resize(img: np.ndarray, target=(512, 512)) -> np.ndarray:
    h, w = img.shape[:2]
    th, tw = target
    scale  = min(tw / w, th / h)
    nh, nw = int(h * scale), int(w * scale)
    res    = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.zeros((th, tw), dtype=res.dtype)
    y0, x0 = (th - nh) // 2, (tw - nw) // 2
    canvas[y0:y0+nh, x0:x0+nw] = res
    return canvas


def to_tensor(img: np.ndarray) -> np.ndarray:
    return img[np.newaxis, np.newaxis, :, :]


# ══════════════════════════════════════════════════════════════════════════
# SIMULATEUR D'ARTEFACTS MACHINE
# ══════════════════════════════════════════════════════════════════════════

class MachineArtifactSimulator:
    """
    Simule les 12 artefacts machine sur une image Rx synthétique.
    Aucun artefact patient (pas de pli cutané, corps étranger, etc.)
    """

    def __init__(self, seed: int = 42):
        self.rng = np.random.default_rng(seed)

    def simulate(self, img: np.ndarray, artifact: str) -> np.ndarray:
        """Applique l'artefact machine sur l'image."""
        if artifact not in CLASS_NAMES:
            raise ValueError(f"Artefact inconnu : {artifact}. "
                             f"Classes valides : {CLASS_NAMES}")

        fn = getattr(self, f"_sim_{artifact}", None)
        if fn is None:
            return img
        return np.clip(fn(img.copy()), 0, 1).astype(np.float32)

    # ── Générateur HT ─────────────────────────────────────────────────────
    def _sim_overexposure(self, img):
        """Image trop noire — gain trop élevé."""
        return np.clip(img * self.rng.uniform(2.2, 3.5), 0, 1)

    def _sim_underexposure(self, img):
        """Image pâle, voile gris — signal insuffisant."""
        return np.clip(img * self.rng.uniform(0.15, 0.35), 0, 1)

    # ── Tube radiogène ────────────────────────────────────────────────────
    def _sim_focal_spot_defect(self, img):
        """Flou asymétrique directionnel — foyer usé."""
        kernel = np.zeros((15, 7))
        kernel[7, :] = 1.0 / 7
        return cv2.filter2D(img, -1, kernel)

    def _sim_anode_artifact(self, img):
        """Bandes de densité irrégulière — stator défaillant."""
        h, w = img.shape
        freq  = self.rng.integers(8, 16)
        for y in range(0, h, freq):
            intensity = self.rng.uniform(0.85, 0.95)
            img[y:y+2, :] *= intensity
        return img

    # ── Collimateur ───────────────────────────────────────────────────────
    def _sim_collimator_misalignment(self, img):
        """Zone sombre sur un bord — lame Pb mal alignée."""
        h, w  = img.shape
        side  = self.rng.integers(0, 4)  # 0=gauche 1=droite 2=haut 3=bas
        width = self.rng.integers(30, 80)
        if side == 0: img[:, :width]  *= 0.1
        elif side == 1: img[:, -width:] *= 0.1
        elif side == 2: img[:width, :]  *= 0.1
        else:           img[-width:, :] *= 0.1
        return img

    # ── Grille Bucky ──────────────────────────────────────────────────────
    def _sim_grid_lines(self, img):
        """Stries parallèles régulières — oscillation Bucky bloquée."""
        freq = self.rng.integers(16, 24)
        for x in range(0, img.shape[1], freq):
            img[:, x:x+2] *= self.rng.uniform(0.60, 0.80)
        return img

    def _sim_grid_cutoff(self, img):
        """Densification latérale progressive — grille décentrée."""
        h, w = img.shape
        for x in range(w):
            dist = min(x, w - 1 - x)
            fac  = max(0.3, dist / (w * 0.3))
            img[:, x] *= min(fac, 1.0)
        return img

    # ── Capteur DR ────────────────────────────────────────────────────────
    def _sim_dead_pixels(self, img):
        """Points noirs fixes — pixels TFT défaillants."""
        n_pts  = self.rng.integers(50, 200)
        n_lines = self.rng.integers(1, 4)
        h, w   = img.shape
        ys = self.rng.integers(0, h, n_pts)
        xs = self.rng.integers(0, w, n_pts)
        img[ys, xs] = 0.0
        for _ in range(n_lines):
            if self.rng.random() > 0.5:
                img[self.rng.integers(0, h), :] = 0.0
            else:
                img[:, self.rng.integers(0, w)] = 0.0
        return img

    def _sim_ghosting_dr(self, img):
        """Ombre du cliché précédent — recharge incomplète capteur."""
        ghost  = np.roll(img, shift=self.rng.integers(20, 60), axis=0)
        ghost  = np.roll(ghost, shift=self.rng.integers(10, 40), axis=1)
        return img * 0.80 + ghost * 0.20

    # ── Capteur CR ────────────────────────────────────────────────────────
    def _sim_cr_scratch(self, img):
        """Traits fins rectilignes — plaque IP rayée."""
        n = self.rng.integers(3, 10)
        for _ in range(n):
            x   = self.rng.integers(10, img.shape[1] - 10)
            val = self.rng.uniform(0.0, 0.3)
            img[:, x:x+1] = val
        return img

    def _sim_cr_fog(self, img):
        """Voile gris uniforme — lampe UV d'effacement défaillante."""
        voile = self.rng.uniform(0.12, 0.22)
        return img * (1 - voile) + voile

    # ── Console / PACS ────────────────────────────────────────────────────
    def _sim_processing_artifact(self, img):
        """Fenêtrage inadapté — LUT erronée."""
        choice = self.rng.integers(0, 3)
        if choice == 0:
            return np.clip(img * 1.8 - 0.3, 0, 1)   # surbrillance
        elif choice == 1:
            return 1.0 - img                          # image inversée
        else:
            # Blocs de compression
            bs = 32
            h, w = img.shape
            for i in range(0, h, bs):
                for j in range(0, w, bs):
                    block = img[i:i+bs, j:j+bs]
                    img[i:i+bs, j:j+bs] = block.mean()
            return img


# ══════════════════════════════════════════════════════════════════════════
# PRÉPROCESSEUR PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

class RxPreprocessor:

    def __init__(self, target_size=(512, 512), clahe_clip=2.0, augment=False, seed=42):
        self.target_size = target_size
        self.clahe_clip  = clahe_clip
        self.augment     = augment
        self.simulator   = MachineArtifactSimulator(seed)

    def process_file(self, path: str,
                     artifact: Optional[str] = None
                     ) -> Tuple[np.ndarray, np.ndarray]:
        raw = load_image(path)
        return self.process_array(raw, artifact)

    def process_array(self, img: np.ndarray,
                      artifact: Optional[str] = None
                      ) -> Tuple[np.ndarray, np.ndarray]:
        img   = resize(img, self.target_size)
        img   = apply_clahe(img, self.clahe_clip)
        img_n = normalize(img)
        if artifact:
            img_n = self.simulator.simulate(img_n, artifact)
        disp   = (img_n * 255).astype(np.uint8)
        tensor = to_tensor(img_n)
        return tensor, disp

    def create_synthetic_rx(self, artifact: Optional[str] = None,
                             seed: int = 0) -> Tuple[np.ndarray, np.ndarray]:
        """Génère un cliché Rx thorax synthétique réaliste."""
        rng = np.random.default_rng(seed)
        h, w = self.target_size
        img  = np.zeros((h, w), dtype=np.float32)

        # Médiastin
        cv2.ellipse(img, (w//2, h//2), (w//6, h//3), 0, 0, 360, 0.5, -1)
        # Poumons
        cv2.ellipse(img, (w//4, h//2), (w//5, h//3), 0, 0, 360, 0.15, -1)
        cv2.ellipse(img, (3*w//4, h//2), (w//5, h//3), 0, 0, 360, 0.15, -1)
        # Côtes
        for i in range(5):
            y = h//4 + i * h//10
            cv2.ellipse(img, (w//2, y), (w//3 + i*10, h//12), 0, 0, 180, 0.6, 2)
        # Clavicules
        cv2.ellipse(img, (w//3, h//5), (w//8, h//20), -20, 0, 180, 0.7, 3)
        cv2.ellipse(img, (2*w//3, h//5), (w//8, h//20), 20, 0, 180, 0.7, 3)
        # Cœur
        cv2.ellipse(img, (w//2 - w//12, h//2 + h//12),
                    (w//10, h//8), -15, 0, 360, 0.65, -1)
        # Bruit + flou
        img += rng.normal(0, 0.03, (h, w)).astype(np.float32)
        img  = cv2.GaussianBlur(np.clip(img, 0, 1), (5, 5), 1.0)

        # Appliquer l'artefact machine si demandé
        if artifact:
            img = self.simulator.simulate(img, artifact)

        img    = np.clip(img, 0, 1).astype(np.float32)
        disp   = (img * 255).astype(np.uint8)
        tensor = to_tensor(img)
        return tensor, disp