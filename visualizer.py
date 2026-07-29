"""
visualizer.py
=============
Visualisation des résultats de détection et segmentation d'artefacts.
Génère des overlays colorés, heatmaps et figures de synthèse.
"""

import numpy as np
import cv2
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
from typing import Dict, List, Tuple, Optional
import os

from knowledge_base import get_artifact_info, URGENCE_ORDRE


# ═══════════════════════════════════════════════════════════════════════════
# PALETTE DE COULEURS
# ═══════════════════════════════════════════════════════════════════════════

URGENCE_COLORS = {
    "critique": "#dc2626",   # rouge vif
    "haute":    "#ea580c",   # orange foncé
    "moyenne":  "#ca8a04",   # jaune-or
    "faible":   "#16a34a",   # vert
    "aucune":   "#2563eb",   # bleu
}

URGENCE_LABELS_FR = {
    "critique": "🔴 CRITIQUE",
    "haute":    "🟠 HAUTE",
    "moyenne":  "🟡 MOYENNE",
    "faible":   "🟢 FAIBLE",
    "aucune":   "🔵 AUCUNE",
}


# ═══════════════════════════════════════════════════════════════════════════
# OVERLAY DE SEGMENTATION
# ═══════════════════════════════════════════════════════════════════════════

def overlay_masks(image_gray: np.ndarray,
                  detections: List[Dict],
                  alpha: float = 0.4) -> np.ndarray:
    """
    Superpose les masques de segmentation colorés sur l'image en niveaux de gris.
    Retourne une image RGB uint8.
    """
    if image_gray.ndim == 2:
        rgb = cv2.cvtColor(image_gray, cv2.COLOR_GRAY2RGB)
    else:
        rgb = image_gray.copy()

    overlay = rgb.copy().astype(np.float32)

    for det in detections:
        info  = get_artifact_info(det["class"])
        color = info["couleur_viz"]          # (R, G, B)
        mask  = det["mask_bin"]              # (H, W) uint8

        # Colorer les pixels masqués
        for c, col in enumerate(color):
            overlay[:, :, c] = np.where(
                mask > 0,
                overlay[:, :, c] * (1 - alpha) + col * alpha,
                overlay[:, :, c]
            )

        # Contour du masque
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay.astype(np.uint8), contours, -1,
                         color[::-1], 2)  # BGR pour cv2

    return np.clip(overlay, 0, 255).astype(np.uint8)


def draw_bounding_boxes(image: np.ndarray,
                        detections: List[Dict]) -> np.ndarray:
    """
    Dessine les bounding boxes des régions contenant des artefacts.
    """
    out = image.copy()
    for det in detections:
        info  = get_artifact_info(det["class"])
        color = info["couleur_viz"]
        mask  = det["mask_bin"]
        label = info["label"]
        conf  = det["confidence"]

        # Trouver la bounding box du masque
        ys, xs = np.where(mask > 0)
        if len(ys) == 0:
            continue
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()), int(ys.max())

        bgr = (color[2], color[1], color[0])
        cv2.rectangle(out, (x1, y1), (x2, y2), bgr, 2)

        # Étiquette
        txt = f"{label[:20]} {conf*100:.0f}%"
        (tw, th), _ = cv2.getTextSize(txt, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
        cv2.rectangle(out, (x1, y1-th-6), (x1+tw+4, y1), bgr, -1)
        cv2.putText(out, txt, (x1+2, y1-3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255,255,255), 1,
                    cv2.LINE_AA)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# FIGURE DE SYNTHÈSE MATPLOTLIB
# ═══════════════════════════════════════════════════════════════════════════

def create_analysis_figure(image_display: np.ndarray,
                           detections: List[Dict],
                           full_seg_map: np.ndarray,
                           all_probs: np.ndarray,
                           class_names: List[str],
                           output_path: str,
                           patient_id: str = "PATIENT_001") -> List[str]:
    """
    Génère UNE image PAR artefact détecté (segmentation individuelle de
    chaque masque sur la radiographie d'entrée), plutôt qu'une seule image
    combinée. Retourne la liste des chemins générés, dans l'ordre de
    confiance décroissante des détections. S'il n'y a aucun artefact détecté,
    retourne une liste à un seul élément (image "aucun artefact").
    """
    base, ext = os.path.splitext(output_path)
    ext = ext or ".png"

    if not detections:
        path = f"{base}_no_artifact{ext}"
        _render_single_artifact(image_display, None, patient_id, path,
                                title="Aucun artefact machine détecté",
                                title_color="#22c55e")
        return [path]

    paths = []
    for i, det in enumerate(detections, start=1):
        info = get_artifact_info(det["class"])
        path = f"{base}_art{i}_{det['class']}{ext}"
        title = (f"{info['label']} — confiance {det['confidence']*100:.0f}% "
                 f"— urgence {info['urgence'].upper()}")
        _render_single_artifact(image_display, det, patient_id, path,
                                title=title,
                                title_color=URGENCE_COLORS.get(info["urgence"], "white"))
        paths.append(path)
    return paths


def _render_single_artifact(image_display: np.ndarray,
                            det: Optional[Dict],
                            patient_id: str,
                            output_path: str,
                            title: str,
                            title_color: str) -> str:
    """Génère une figure centrée sur UN seul artefact (ou l'image seule si det=None)."""
    fig, ax = plt.subplots(figsize=(7, 7), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    if det is not None:
        seg_img = overlay_masks(image_display, [det], alpha=0.45)
        seg_img = draw_bounding_boxes(seg_img, [det])
        ax.imshow(seg_img)
    else:
        ax.imshow(image_display, cmap="gray")
        ax.text(0.5, 0.06, "✓ Aucun artefact détecté — appareil conforme",
                transform=ax.transAxes, ha="center", va="bottom",
                color="#22c55e", fontsize=10, fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.4", facecolor="#0f172a",
                          edgecolor="#22c55e", linewidth=1.2))

    fig.suptitle(f"RxArtifactNet — {patient_id}", fontsize=12, color="white",
                 fontweight="bold", y=0.98)
    ax.set_title(title, color=title_color, fontsize=10, pad=10)
    ax.axis("off")

    plt.savefig(output_path, dpi=150, bbox_inches="tight", facecolor="#0f172a")
    plt.close(fig)
    return output_path


# ═══════════════════════════════════════════════════════════════════════════
# EXPORT IMAGE OVERLAY SEUL
# ═══════════════════════════════════════════════════════════════════════════

def export_overlay_image(image_display: np.ndarray,
                         detections: List[Dict],
                         output_path: str) -> str:
    """Exporte uniquement l'image avec overlays (sans matplotlib)."""
    out = overlay_masks(image_display, detections)
    out = draw_bounding_boxes(out, detections)
    cv2.imwrite(output_path, cv2.cvtColor(out, cv2.COLOR_RGB2BGR))
    return output_path
