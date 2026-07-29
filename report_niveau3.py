"""
report_niveau3.py
=================
Rapport PDF niveau 3 — Prédiction temporelle et alertes.
Intègre les niveaux 1, 2 et 3 dans un rapport unifié.
"""

import os, datetime
import numpy as np
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether
)

# ── Palette ──────────────────────────────────────────────────────────────
C_DARK   = colors.HexColor("#0f172a")
C_NAVY   = colors.HexColor("#1e3a5f")
C_BLUE   = colors.HexColor("#2563eb")
C_BLT    = colors.HexColor("#dbeafe")
C_BLP    = colors.HexColor("#eff6ff")
C_RED    = colors.HexColor("#dc2626")
C_REDL   = colors.HexColor("#fee2e2")
C_ORANGE = colors.HexColor("#ea580c")
C_ORANGL = colors.HexColor("#ffedd5")
C_YELL   = colors.HexColor("#ca8a04")
C_YELLL  = colors.HexColor("#fef9c3")
C_GREEN  = colors.HexColor("#16a34a")
C_GREENL = colors.HexColor("#dcfce7")
C_GREY   = colors.HexColor("#64748b")
C_GREYL  = colors.HexColor("#f1f5f9")
C_WHITE  = colors.white

ALERT_COLOR = {
    "CRITIQUE": (C_RED,    C_REDL),
    "ROUGE":    (C_ORANGE, C_ORANGL),
    "ORANGE":   (C_YELL,   C_YELLL),
    "JAUNE":    (C_BLUE,   C_BLP),
    "VERT":     (C_GREEN,  C_GREENL),
}
ALERT_HEX = {
    "CRITIQUE": "#dc2626",
    "ROUGE":    "#ea580c",
    "ORANGE":   "#ca8a04",
    "JAUNE":    "#2563eb",
    "VERT":     "#16a34a",
}


# ══════════════════════════════════════════════════════════════════════════
# GÉNÉRATION DES GRAPHIQUES
# ══════════════════════════════════════════════════════════════════════════

def plot_degradation_curves(predictions: List[dict],
                             output_path: str) -> str:
    """
    Génère les courbes de dégradation Weibull pour chaque composant.
    """
    n = min(len(predictions), 4)
    if n == 0:
        return None

    fig, axes = plt.subplots(1, n, figsize=(5 * n, 5),
                              facecolor="#0f172a")
    if n == 1:
        axes = [axes]

    fig.suptitle("Courbes de dégradation Weibull — Probabilité de panne",
                 color="white", fontsize=13, fontweight="bold", y=1.02)

    alert_colors_hex = {
        "CRITIQUE": "#ef4444", "ROUGE": "#f97316",
        "ORANGE": "#eab308", "JAUNE": "#3b82f6", "VERT": "#22c55e"
    }

    for ax, pred in zip(axes, predictions[:n]):
        ax.set_facecolor("#1e293b")
        curve = pred["weibull"].get("curve", [])
        if not curve:
            ax.text(0.5, 0.5, "Données\ninsuffisantes",
                    transform=ax.transAxes, ha="center", va="center",
                    color="white", fontsize=10)
            continue

        jours  = [p["jour"]       for p in curve]
        prob   = [p["prob_panne"] for p in curve]
        survie = [p["survie"]     for p in curve]

        niveau = pred["alerte"]["niveau"]
        col    = alert_colors_hex.get(niveau, "#64748b")

        ax.fill_between(jours, prob, alpha=0.15, color=col)
        ax.plot(jours, prob,   color=col,      lw=2.5, label="P(panne)")
        ax.plot(jours, survie, color="#64748b", lw=1.5,
                linestyle="--", label="P(survie)", alpha=0.7)

        # Seuils
        ax.axhline(85, color="#ef4444", linestyle=":", lw=1,
                   alpha=0.6, label="Seuil critique")
        ax.axhline(65, color="#f97316", linestyle=":", lw=1,
                   alpha=0.6, label="Seuil urgent")

        # Fenêtre d'intervention
        fw = pred.get("fenetre_intervention", {})
        if fw.get("debut") and fw.get("fin"):
            today = datetime.date.today()
            try:
                d1 = (datetime.date.fromisoformat(fw["debut"]) - today).days
                d2 = (datetime.date.fromisoformat(fw["fin"])   - today).days
                d1, d2 = max(d1, 0), max(d2, 0)
                if d1 < max(jours) and d2 <= max(jours):
                    ax.axvspan(d1, d2, alpha=0.20, color="#22c55e",
                               label="Fenêtre optimale")
            except Exception:
                pass

        ax.set_title(
            pred["composant_nom"][:28],
            color="white", fontsize=9, pad=6
        )
        ax.set_xlabel("Jours", color="white", fontsize=8)
        ax.set_ylabel("Probabilité (%)", color="white", fontsize=8)
        ax.set_ylim(0, 105)
        ax.tick_params(colors="white", labelsize=7)
        ax.spines[:].set_color("#334155")
        ax.legend(facecolor="#0f172a", labelcolor="white",
                  fontsize=6, loc="upper left")

        # Badge niveau d'alerte
        ax.text(0.98, 0.05, niveau,
                transform=ax.transAxes, ha="right", va="bottom",
                fontsize=8, fontweight="bold", color=col,
                bbox=dict(boxstyle="round,pad=0.3",
                          facecolor="#0f172a", edgecolor=col, linewidth=1))

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close()
    return output_path


def plot_timeline(predictions: List[dict],
                  device_id: str,
                  output_path: str) -> str:
    """
    Génère une timeline horizontale des interventions recommandées.
    """
    fig, ax = plt.subplots(figsize=(14, 5), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    today = datetime.date.today()
    horizon = 365
    ax.set_xlim(0, horizon)
    ax.set_ylim(-1, len(predictions) + 1)

    # Axe temporel
    for d in [0, 30, 60, 90, 180, 270, 365]:
        ax.axvline(d, color="#334155", lw=0.5, alpha=0.5)
        date_lbl = (today + datetime.timedelta(days=d)).strftime("%d/%m")
        ax.text(d, len(predictions) + 0.5, date_lbl,
                ha="center", va="bottom", color="#94a3b8",
                fontsize=7)

    # Ligne "aujourd'hui"
    ax.axvline(0, color="#f59e0b", lw=2, zorder=5)
    ax.text(2, len(predictions) + 0.5, "Aujourd'hui",
            color="#f59e0b", fontsize=8, fontweight="bold")

    alert_colors_hex = {
        "CRITIQUE": "#ef4444", "ROUGE": "#f97316",
        "ORANGE": "#eab308", "JAUNE": "#3b82f6", "VERT": "#22c55e"
    }

    for i, pred in enumerate(predictions):
        y     = i
        col   = alert_colors_hex.get(pred["alerte"]["niveau"], "#64748b")
        nom   = pred["composant_nom"][:30]
        fw    = pred.get("fenetre_intervention", {})

        # Barre de vie restante
        jp = fw.get("jours_restants")
        if jp and jp > 0:
            jp_clipped = min(jp, horizon)
            ax.barh(y, jp_clipped, left=0, height=0.35,
                    color=col, alpha=0.15, zorder=2)
            # Date de panne estimée
            ax.axvline(min(jp, horizon), color=col,
                       lw=1.5, linestyle="--", alpha=0.7)
            ax.text(min(jp, horizon) + 2, y,
                    f"J+{jp}", color=col, va="center",
                    fontsize=7, fontweight="bold")

        # Fenêtre d'intervention optimale
        try:
            d1 = (datetime.date.fromisoformat(fw["debut"]) - today).days
            d2 = (datetime.date.fromisoformat(fw["fin"])   - today).days
            d1 = max(0, min(d1, horizon))
            d2 = max(0, min(d2, horizon))
            if d2 > d1:
                ax.barh(y, d2 - d1, left=d1, height=0.35,
                        color="#22c55e", alpha=0.6, zorder=3,
                        label="Fenêtre optimale" if i == 0 else "")
        except Exception:
            pass

        # Label composant
        ax.text(-5, y, nom, ha="right", va="center",
                color="white", fontsize=8)

        # Pastille d'alerte
        niveau = pred["alerte"]["niveau"]
        ax.scatter([0], [y], s=80, color=col, zorder=6)

    # Légende
    patches = [
        mpatches.Patch(color="#22c55e", alpha=0.7,
                       label="Fenêtre d'intervention optimale"),
        mpatches.Patch(color="#94a3b8", alpha=0.3,
                       label="Durée de vie résiduelle"),
    ]
    ax.legend(handles=patches, facecolor="#0f172a",
              labelcolor="white", fontsize=8, loc="lower right")

    ax.set_title(
        f"Timeline des interventions — {device_id}",
        color="white", fontsize=12, fontweight="bold", pad=10
    )
    ax.set_xlabel("Jours depuis aujourd'hui", color="white", fontsize=9)
    ax.set_yticks([])
    ax.spines[:].set_color("#334155")
    ax.tick_params(colors="white")

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close()
    return output_path


def plot_score_history(history: List[dict],
                       output_path: str) -> str:
    """Courbe d'évolution du score global dans le temps."""
    if len(history) < 2:
        return None

    dates  = [e["date"][:10] for e in history]
    scores = [e["score_global"] for e in history]

    fig, ax = plt.subplots(figsize=(12, 4), facecolor="#0f172a")
    ax.set_facecolor("#1e293b")

    x = np.arange(len(scores))
    ax.fill_between(x, scores, alpha=0.15, color="#3b82f6")
    ax.plot(x, scores, color="#3b82f6", lw=2.5, marker="o",
            markersize=5, label="Score global")

    # Zones de risque
    ax.axhspan(85, 100, alpha=0.08, color="#ef4444")
    ax.axhspan(65,  85, alpha=0.08, color="#f97316")
    ax.axhspan(45,  65, alpha=0.08, color="#eab308")
    ax.axhspan(20,  45, alpha=0.08, color="#3b82f6")

    # Labels zones
    for y, label, col in [
        (92, "IMMÉDIAT",     "#ef4444"),
        (75, "URGENT",       "#f97316"),
        (55, "PLANIFIÉ",     "#eab308"),
        (32, "SURVEILLANCE", "#3b82f6"),
        (10, "NORMAL",       "#22c55e"),
    ]:
        ax.text(len(scores) - 0.5, y, label, ha="right", va="center",
                color=col, fontsize=7, fontweight="bold", alpha=0.7)

    # Régression
    if len(scores) >= 3:
        coeffs = np.polyfit(x, scores, min(2, len(scores)-1))
        x_ext  = np.linspace(0, len(scores) + 4, 100)
        y_ext  = np.polyval(coeffs, x_ext)
        y_ext  = np.clip(y_ext, 0, 100)
        ax.plot(x_ext, y_ext, color="#f59e0b", lw=1.5,
                linestyle="--", alpha=0.7, label="Tendance")
        # Zone de projection
        ax.axvspan(len(scores) - 1, len(scores) + 4,
                   alpha=0.08, color="#f59e0b")
        ax.text(len(scores) + 0.5,
                float(np.polyval(coeffs, len(scores) + 2)),
                "Projection", color="#f59e0b", fontsize=7, va="bottom")

    ax.set_xticks(x)
    ax.set_xticklabels(dates, rotation=30, ha="right",
                       fontsize=7, color="white")
    ax.set_ylabel("Score de risque", color="white", fontsize=9)
    ax.set_ylim(0, 105)
    ax.set_title("Évolution du score de risque global",
                 color="white", fontsize=11, fontweight="bold")
    ax.tick_params(colors="white")
    ax.spines[:].set_color("#334155")
    ax.legend(facecolor="#0f172a", labelcolor="white", fontsize=8)

    plt.tight_layout()
    plt.savefig(output_path, dpi=130, bbox_inches="tight",
                facecolor="#0f172a")
    plt.close()
    return output_path


# ══════════════════════════════════════════════════════════════════════════
# RAPPORT PDF NIVEAU 3
# ══════════════════════════════════════════════════════════════════════════

class PredictiveReport:

    MARGIN = 1.8 * cm
    W      = A4[0] - 2 * 1.8 * cm

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.st = getSampleStyleSheet()
        self._styles()
        self.story = []

    def _styles(self):
        def S(n, **kw):
            p = kw.pop("parent", "Normal")
            return ParagraphStyle(n, parent=self.st[p], **kw)
        self.sTitle  = S("t",  parent="Title", fontSize=18, textColor=C_WHITE,
                         alignment=TA_CENTER, fontName="Helvetica-Bold", leading=22)
        self.sSub    = S("sb", fontSize=9,  textColor=C_BLT,  alignment=TA_CENTER)
        self.sMeta   = S("mt", fontSize=8,  textColor=C_BLT,  alignment=TA_CENTER)
        self.sSect   = S("sc", fontSize=11, textColor=C_WHITE,
                         fontName="Helvetica-Bold", leading=14)
        self.sH2     = S("h2", fontSize=10, textColor=C_NAVY,
                         fontName="Helvetica-Bold", spaceBefore=5, spaceAfter=3)
        self.sBody   = S("bd", fontSize=8.5, textColor=colors.HexColor("#1e293b"),
                         leading=13)
        self.sBullet = S("bl", fontSize=8.5, textColor=colors.HexColor("#1e293b"),
                         leading=13, leftIndent=12, spaceAfter=2)
        self.sNote   = S("nt", fontSize=8,  textColor=colors.HexColor("#475569"),
                         leading=12, leftIndent=8)
        self.sTH     = S("TH", fontSize=8,  textColor=C_WHITE,
                         fontName="Helvetica-Bold", alignment=TA_CENTER)
        self.sTD     = S("TD", fontSize=8,  textColor=colors.HexColor("#1e293b"),
                         leading=11)
        self.sTDc    = S("Tc", fontSize=8,  textColor=colors.HexColor("#1e293b"),
                         leading=11, alignment=TA_CENTER)

    def _sect(self, text, color=None):
        c = color or C_NAVY
        t = Table([[Paragraph(text, self.sSect)]], colWidths=[self.W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), c),
            ("TOPPADDING",    (0,0),(-1,-1), 8),
            ("BOTTOMPADDING", (0,0),(-1,-1), 8),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
        ]))
        return t

    def _box(self, text, bg=None, border=None):
        bg = bg or C_BLP; border = border or C_BLUE
        t = Table([[Paragraph(text, self.sNote)]], colWidths=[self.W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), bg),
            ("LINEABOVE",     (0,0),(-1,0),  2, border),
            ("TOPPADDING",    (0,0),(-1,-1), 7),
            ("BOTTOMPADDING", (0,0),(-1,-1), 7),
            ("LEFTPADDING",   (0,0),(-1,-1), 10),
        ]))
        return t

    def _img(self, path, h_ratio=0.45):
        if path and os.path.exists(path):
            return RLImage(path, width=self.W, height=self.W * h_ratio)
        return Spacer(1, 0.1*cm)

    def _garde(self, meta, prediction):
        s = self.story
        ag  = prediction.get("alerte_globale")
        niv = ag["niveau"] if ag else "VERT"
        c_fg, c_bg = ALERT_COLOR.get(niv, (C_GREY, C_GREYL))
        cv = Table([
            [Paragraph("MAINTENANCE PRÉDICTIVE — NIVEAU 3", self.sSub)],
            [Paragraph("Prédiction Temporelle de Panne", self.sTitle)],
            [Paragraph("Modèle de Weibull · Régression temporelle · Fenêtre d'intervention",
                       self.sSub)],
            [Spacer(1, 0.3*cm)],
            [HRFlowable(width="100%", thickness=1,
                        color=colors.HexColor("#93c5fd"))],
            [Spacer(1, 0.2*cm)],
            [Paragraph(
                f"Appareil : <b>{meta.get('device','N/A')}</b>  |  "
                f"Date : <b>{meta.get('date','')}</b>  |  "
                f"Alerte : <b>{niv}</b>",
                self.sMeta)],
        ], colWidths=[self.W])
        cv.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), C_DARK),
            ("TOPPADDING",    (0,0),(-1,-1), 14),
            ("BOTTOMPADDING", (0,0),(-1,-1), 14),
            ("LEFTPADDING",   (0,0),(-1,-1), 18),
            ("RIGHTPADDING",  (0,0),(-1,-1), 18),
        ]))
        s.append(cv)
        s.append(Spacer(1, 0.5*cm))

    def _resume_alertes(self, prediction):
        s = self.story
        s.append(self._sect("1.  Résumé des Alertes"))
        s.append(Spacer(1, 0.3*cm))

        ag  = prediction.get("alerte_globale")
        if not ag:
            s.append(self._box("Aucune alerte active — appareil conforme.",
                                bg=C_GREENL, border=C_GREEN))
            return

        niv  = ag["niveau"]
        c_fg, c_bg = ALERT_COLOR.get(niv, (C_GREY, C_GREYL))

        # Bandeau alerte globale
        bt = Table([[
            Paragraph(f"{ag['emoji']}  ALERTE {niv}",
                      ParagraphStyle("al", parent=self.sTH, fontSize=14,
                                     textColor=c_fg)),
            Paragraph(ag["delai_intervention"],
                      ParagraphStyle("ali", parent=self.sNote,
                                     textColor=c_fg, alignment=TA_RIGHT)),
        ]], colWidths=[8*cm, self.W - 8*cm])
        bt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), c_bg),
            ("LINEABOVE",     (0,0),(-1,0),  3, c_fg),
            ("TOPPADDING",    (0,0),(-1,-1), 12),
            ("BOTTOMPADDING", (0,0),(-1,-1), 12),
            ("LEFTPADDING",   (0,0),(-1,-1), 14),
            ("RIGHTPADDING",  (0,0),(-1,-1), 14),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        s.append(bt)
        s.append(Spacer(1, 0.3*cm))

        # Tableau synthèse par composant
        rows = []
        for pred in prediction["predictions"]:
            a   = pred["alerte"]
            fw  = pred.get("fenetre_intervention", {})
            jp  = fw.get("jours_restants", "—")
            fd  = fw.get("debut", "—")
            reg = pred.get("regression", {})
            rows.append([
                pred["composant_nom"][:28],
                a["niveau"],
                f"{pred['score_actuel']:.0f}/100",
                f"J+{jp}" if isinstance(jp, int) else "—",
                fd[:10] if fd != "—" else "—",
                f"{pred['weibull'].get('prob_panne_30j',0):.0f}%",
            ])
        t = Table(
            [[Paragraph(h, self.sTH) for h in
              ["Composant","Alerte","Score","Panne est.",
               "Début interv.","P(panne/30j)"]]] +
            [[Paragraph(str(c), self.sTDc if i in (1,2,3,5) else self.sTD)
              for i, c in enumerate(r)] for r in rows],
            colWidths=[4.5*cm, 2.0*cm, 1.8*cm, 2.0*cm, 2.8*cm, 2.3*cm],
            repeatRows=1
        )
        ts = TableStyle([
            ("BACKGROUND",    (0,0),(-1,0),  C_NAVY),
            ("ROWBACKGROUNDS",(0,1),(-1,-1), [C_WHITE, C_GREYL]),
            ("GRID",          (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
            ("TOPPADDING",    (0,0),(-1,-1), 5),
            ("BOTTOMPADDING", (0,0),(-1,-1), 5),
            ("LEFTPADDING",   (0,0),(-1,-1), 6),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ("LINEBELOW",     (0,0),(-1,0),  1.5, C_BLUE),
        ])
        for i, pred in enumerate(prediction["predictions"], 1):
            niv_i = pred["alerte"]["niveau"]
            c_fi, c_bi = ALERT_COLOR.get(niv_i, (C_GREY, C_GREYL))
            ts.add("BACKGROUND", (1,i),(1,i), c_bi)
            ts.add("TEXTCOLOR",  (1,i),(1,i), c_fi)
            ts.add("FONTNAME",   (1,i),(1,i), "Helvetica-Bold")
        t.setStyle(ts)
        s.append(t)

    def _section_courbes(self, fig_degradation, fig_timeline, fig_history):
        s = self.story
        s.append(PageBreak())
        s.append(self._sect("2.  Courbes de Dégradation & Timeline"))
        s.append(Spacer(1, 0.3*cm))

        if fig_degradation:
            s.append(Paragraph("Probabilité de panne par composant (Weibull) :",
                                self.sH2))
            s.append(self._img(fig_degradation, 0.38))
            s.append(Paragraph(
                "Chaque courbe montre la probabilité de panne cumulée. "
                "La zone verte indique la fenêtre d'intervention optimale.",
                self.sNote))
            s.append(Spacer(1, 0.3*cm))

        if fig_timeline:
            s.append(Paragraph("Timeline des interventions recommandées :", self.sH2))
            s.append(self._img(fig_timeline, 0.28))
            s.append(Paragraph(
                "Barres vertes = fenêtres d'intervention optimales. "
                "Lignes pointillées = date de panne estimée.",
                self.sNote))
            s.append(Spacer(1, 0.3*cm))

        if fig_history:
            s.append(Paragraph("Évolution historique du score de risque :", self.sH2))
            s.append(self._img(fig_history, 0.26))
            s.append(Paragraph(
                "Zone jaune = projection future. "
                "Zones colorées = niveaux d'alerte.",
                self.sNote))

    def _section_predictions(self, prediction):
        s = self.story
        s.append(PageBreak())
        s.append(self._sect("3.  Prédictions Détaillées par Composant"))

        for i, pred in enumerate(prediction["predictions"], 1):
            s.append(Spacer(1, 0.4*cm))
            niv    = pred["alerte"]["niveau"]
            c_fg, c_bg = ALERT_COLOR.get(niv, (C_GREY, C_GREYL))
            reg    = pred.get("regression", {})
            weib   = pred.get("weibull",    {})
            fw     = pred.get("fenetre_intervention", {})

            # En-tête
            ht = Table([[
                Paragraph(f"#{i} — {pred['composant_nom']}",
                          ParagraphStyle("ch", parent=self.sSect,
                                         fontSize=10, textColor=C_WHITE)),
                Paragraph(
                    f"{pred['alerte']['emoji']}  {niv}  |  "
                    f"Score {pred['score_actuel']:.0f}/100",
                    ParagraphStyle("ch2", parent=self.sSect, fontSize=9,
                                   textColor=c_bg, alignment=TA_RIGHT)),
            ]], colWidths=[self.W*0.62, self.W*0.38])
            ht.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), c_fg),
                ("TOPPADDING",    (0,0),(-1,-1), 7),
                ("BOTTOMPADDING", (0,0),(-1,-1), 7),
                ("LEFTPADDING",   (0,0),(0,-1),  10),
                ("RIGHTPADDING",  (-1,0),(-1,-1),10),
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ]))
            s.append(KeepTogether([ht]))
            s.append(Spacer(1, 0.15*cm))

            # Tableau prédictions
            jp  = fw.get("jours_restants", "—")
            deb = fw.get("debut", "—")
            fin = fw.get("fin",   "—")
            msg = fw.get("message", "—")

            rows = [
                ["Sous-système",         pred["sous_systeme"]],
                ["Alerte",               f"{pred['alerte']['emoji']} {niv} — {pred['alerte']['description']}"],
                ["Délai intervention",    pred["alerte"]["delai_intervention"]],
                ["Panne estimée",         f"J+{jp}" if isinstance(jp,int) else "Indéterminée"],
                ["Fenêtre optimale",      f"{deb} → {fin}"],
                ["Conseil",              msg],
                ["P(panne) 30 jours",    f"{weib.get('prob_panne_30j',0):.1f} % (Weibull)"],
                ["P(panne) 90 jours",    f"{weib.get('prob_panne_90j',0):.1f} % (Weibull)"],
                ["Survie actuelle",       f"{weib.get('survival_now',0):.1f} %"],
                ["Score prédit J+30",     f"{reg.get('score_j30','—')}/100" if reg.get('score_j30') else "—"],
                ["Score prédit J+90",     f"{reg.get('score_j90','—')}/100" if reg.get('score_j90') else "—"],
                ["Confiance prédiction",  f"{reg.get('confiance_pct',0):.0f} % (R²={reg.get('r2',0):.2f})"],
            ]
            pt = Table(rows, colWidths=[4.5*cm, self.W - 4.5*cm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",     (0,0),(0,-1), c_bg),
                ("TEXTCOLOR",      (0,0),(0,-1), c_fg),
                ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0),(-1,-1), 8.5),
                ("ROWBACKGROUNDS", (1,0),(-1,-1), [C_WHITE, C_GREYL]),
                ("GRID",           (0,0),(-1,-1), 0.3,
                 colors.HexColor("#e2e8f0")),
                ("TOPPADDING",     (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
                ("LEFTPADDING",    (0,0),(-1,-1), 8),
                ("VALIGN",         (0,0),(-1,-1), "TOP"),
            ]))
            s.append(pt)

            # Actions prioritaires
            if pred.get("actions_prioritaires"):
                s.append(Spacer(1, 0.15*cm))
                s.append(Paragraph("Actions prioritaires :", self.sH2))
                for a in pred["actions_prioritaires"]:
                    s.append(Paragraph(f"▸  {a}", self.sBullet))

    def _footer(self):
        s = self.story
        s.append(Spacer(1, 0.4*cm))
        s.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#94a3b8")))
        s.append(Spacer(1, 0.2*cm))
        s.append(self._box(
            "Modèles utilisés : Distribution de Weibull (fiabilité industrielle IEC 61649) "
            "+ Régression polynomiale sur l'historique des scores. "
            "Les prédictions sont des estimations probabilistes — "
            "toute décision d'intervention doit être validée par un ingénieur biomédical.",
            bg=C_YELLL, border=C_YELL
        ))

    def build(self, prediction: dict, meta: dict,
              fig_degradation: str = None,
              fig_timeline:    str = None,
              fig_history:     str = None) -> str:

        doc = SimpleDocTemplate(
            self.output_path, pagesize=A4,
            leftMargin=self.MARGIN, rightMargin=self.MARGIN,
            topMargin=1.8*cm, bottomMargin=1.8*cm,
        )
        self._garde(meta, prediction)
        self._resume_alertes(prediction)
        self._section_courbes(fig_degradation, fig_timeline, fig_history)
        self._section_predictions(prediction)
        self._footer()
        doc.build(self.story)
        return self.output_path
