"""
report_niveau2.py
=================
Rapport PDF niveau 2 — Diagnostic composant et maintenance prédictive.
S'appuie sur la sortie de DiagnosticEngine pour produire un rapport complet.
"""

import os
import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.units import cm
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, KeepTogether
)

# ── Palette ──────────────────────────────────────────────────────────────
C_DARK    = colors.HexColor("#0f172a")
C_NAVY    = colors.HexColor("#1e3a5f")
C_BLUE    = colors.HexColor("#2563eb")
C_BLUE_LT = colors.HexColor("#dbeafe")
C_BLUE_PL = colors.HexColor("#eff6ff")
C_RED     = colors.HexColor("#dc2626")
C_RED_LT  = colors.HexColor("#fee2e2")
C_ORANGE  = colors.HexColor("#ea580c")
C_ORANGE_L= colors.HexColor("#ffedd5")
C_YELLOW  = colors.HexColor("#ca8a04")
C_YELLOW_L= colors.HexColor("#fef9c3")
C_GREEN   = colors.HexColor("#16a34a")
C_GREEN_L = colors.HexColor("#dcfce7")
C_PURPLE  = colors.HexColor("#7c3aed")
C_PURPLE_L= colors.HexColor("#f5f3ff")
C_GREY    = colors.HexColor("#64748b")
C_GREY_LT = colors.HexColor("#f1f5f9")
C_WHITE   = colors.white

HORIZON_COLORS = {
    "IMMÉDIAT":     (C_RED,    C_RED_LT),
    "URGENT":       (C_ORANGE, C_ORANGE_L),
    "PLANIFIÉ":     (C_YELLOW, C_YELLOW_L),
    "SURVEILLANCE": (C_BLUE,   C_BLUE_PL),
    "NORMAL":       (C_GREEN,  C_GREEN_L),
}
TENDANCE_FR = {
    "aggravation_rapide":  "↑↑ Aggravation rapide",
    "aggravation":         "↑  Aggravation",
    "stable":              "→  Stable",
    "amelioration":        "↓  Amélioration",
    "amelioration_rapide": "↓↓ Amélioration rapide",
    "insuffisant":         "—  Données insuffisantes",
}


def _bar_chart(score: float, width_cm: float = 8.0) -> Table:
    """Barre de progression colorée pour le score de risque."""
    pct     = min(max(score, 0), 100) / 100
    w_total = width_cm * cm
    w_fill  = w_total * pct
    w_empty = w_total - w_fill

    if score >= 85:   bar_color = C_RED
    elif score >= 65: bar_color = C_ORANGE
    elif score >= 45: bar_color = C_YELLOW
    elif score >= 20: bar_color = C_BLUE
    else:             bar_color = C_GREEN

    filled  = Table([[""]], colWidths=[w_fill])
    filled.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), bar_color),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))
    empty   = Table([[""]], colWidths=[max(w_empty, 0.01)])
    empty.setStyle(TableStyle([
        ("BACKGROUND",    (0,0), (-1,-1), C_GREY_LT),
        ("TOPPADDING",    (0,0), (-1,-1), 5),
        ("BOTTOMPADDING", (0,0), (-1,-1), 5),
    ]))

    bar = Table([[filled, empty]], colWidths=[w_fill, max(w_empty, 0.01)])
    bar.setStyle(TableStyle([
        ("TOPPADDING",    (0,0), (-1,-1), 0),
        ("BOTTOMPADDING", (0,0), (-1,-1), 0),
        ("LEFTPADDING",   (0,0), (-1,-1), 0),
        ("RIGHTPADDING",  (0,0), (-1,-1), 0),
    ]))
    return bar


class DiagnosticReport:

    MARGIN    = 1.8 * cm
    W         = A4[0] - 2 * 1.8 * cm

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.st = getSampleStyleSheet()
        self._styles()
        self.story = []

    def _styles(self):
        def S(n, **kw):
            p = kw.pop("parent", "Normal")
            return ParagraphStyle(n, parent=self.st[p], **kw)

        self.sTitle  = S("sT",  parent="Title", fontSize=18, textColor=C_WHITE,
                         alignment=TA_CENTER, fontName="Helvetica-Bold",
                         spaceAfter=2, leading=22)
        self.sSub    = S("sSb", fontSize=9,  textColor=C_BLUE_LT,
                         alignment=TA_CENTER, spaceAfter=2)
        self.sMeta   = S("sMt", fontSize=8,  textColor=C_BLUE_LT,
                         alignment=TA_CENTER)
        self.sSect   = S("sSc", fontSize=11, textColor=C_WHITE,
                         fontName="Helvetica-Bold", leading=14)
        self.sH2     = S("sH2", fontSize=10, textColor=C_NAVY,
                         fontName="Helvetica-Bold", spaceBefore=6, spaceAfter=3)
        self.sBody   = S("sBd", fontSize=8.5,
                         textColor=colors.HexColor("#1e293b"), leading=13)
        self.sBullet = S("sBl", fontSize=8.5,
                         textColor=colors.HexColor("#1e293b"),
                         leading=13, leftIndent=12, spaceAfter=2)
        self.sNote   = S("sNt", fontSize=8,
                         textColor=colors.HexColor("#475569"),
                         leading=12, leftIndent=8)
        self.sTH     = S("sTH", fontSize=8,  textColor=C_WHITE,
                         fontName="Helvetica-Bold", alignment=TA_CENTER)
        self.sTD     = S("sTD", fontSize=8,
                         textColor=colors.HexColor("#1e293b"), leading=11)
        self.sTDc    = S("sTc", fontSize=8,
                         textColor=colors.HexColor("#1e293b"),
                         leading=11, alignment=TA_CENTER)

    def _sect(self, text, color=None):
        c = color or C_NAVY
        t = Table([[Paragraph(text, self.sSect)]],
                  colWidths=[self.W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), c),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ]))
        return t

    def _box(self, text, bg=None, border=None):
        bg = bg or C_BLUE_PL; border = border or C_BLUE
        t = Table([[Paragraph(text, self.sNote)]], colWidths=[self.W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), bg),
            ("LINEABOVE",     (0,0), (-1,0),  2, border),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        return t

    def _make_table(self, headers, rows, widths, hcolor=None):
        hc   = hcolor or C_NAVY
        data = [[Paragraph(h, self.sTH) for h in headers]]
        for row in rows:
            data.append([
                Paragraph(str(c), self.sTDc if i == 0 else self.sTD)
                for i, c in enumerate(row)
            ])
        t = Table(data, colWidths=widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(-1,0),  hc),
            ("ROWBACKGROUNDS",  (0,1),(-1,-1), [C_WHITE, C_GREY_LT]),
            ("GRID",           (0,0),(-1,-1), 0.4, colors.HexColor("#cbd5e1")),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 6),
            ("VALIGN",         (0,0),(-1,-1), "TOP"),
            ("LINEBELOW",      (0,0),(-1,0),  1.5, C_BLUE),
        ]))
        return t

    # ── PAGE DE GARDE ─────────────────────────────────────────────────────
    def _garde(self, diag: dict, meta: dict):
        s = self.story
        cv = Table([
            [Paragraph("DIAGNOSTIC MAINTENANCE PRÉDICTIVE", self.sSub)],
            [Paragraph("Radiographie Thoracique — Niveau 2", self.sTitle)],
            [Paragraph("Analyse composant · Score de risque · Horizon d'intervention", self.sSub)],
            [Spacer(1, 0.3*cm)],
            [HRFlowable(width="100%", thickness=1,
                        color=colors.HexColor("#93c5fd"))],
            [Spacer(1, 0.2*cm)],
            [Paragraph(
                f"Appareil : <b>{meta.get('device','N/A')}</b>  |  "
                f"Patient : <b>{meta.get('patient_id','N/A')}</b>  |  "
                f"Date : <b>{meta.get('date', datetime.date.today().strftime('%d/%m/%Y'))}</b>",
                self.sMeta)],
            [Paragraph(
                f"Opérateur : {meta.get('operator','Automatique')}  |  "
                f"RxArtifactNet v2.0 — Niveaux 1 & 2",
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

    # ── RÉSUMÉ EXÉCUTIF ────────────────────────────────────────────────────
    def _resume(self, diag: dict):
        s = self.story
        s.append(self._sect("1.  Résumé Exécutif — Diagnostic Appareil"))
        s.append(Spacer(1, 0.3*cm))

        hz   = diag["horizon_global"]
        c_fg, c_bg = HORIZON_COLORS.get(hz["label"], (C_GREY, C_GREY_LT))
        score = diag["score_global"]

        # Bandeau score global
        score_data = [[
            Paragraph(f"Score de risque global", self.sNote),
            Paragraph(f"<b>{score:.0f} / 100</b>",
                      ParagraphStyle("sc", parent=self.sTH, fontSize=16,
                                     textColor=c_fg)),
            Paragraph(f"<b>{hz['label']}</b>",
                      ParagraphStyle("hz", parent=self.sTH, fontSize=11,
                                     textColor=c_fg, alignment=TA_RIGHT)),
        ]]
        bt = Table(score_data,
                   colWidths=[5*cm, 4*cm, self.W-9*cm])
        bt.setStyle(TableStyle([
            ("BACKGROUND",    (0,0),(-1,-1), c_bg),
            ("LINEABOVE",     (0,0),(-1,0),  2, c_fg),
            ("TOPPADDING",    (0,0),(-1,-1), 10),
            ("BOTTOMPADDING", (0,0),(-1,-1), 10),
            ("LEFTPADDING",   (0,0),(-1,-1), 12),
            ("RIGHTPADDING",  (0,0),(-1,-1), 12),
            ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
        ]))
        s.append(bt)
        s.append(Spacer(1, 0.2*cm))

        # Barre de progression
        s.append(Paragraph("Niveau de risque :", self.sNote))
        s.append(Spacer(1, 0.1*cm))
        s.append(_bar_chart(score, width_cm=self.W / cm))
        s.append(Spacer(1, 0.3*cm))

        # Action recommandée
        s.append(self._box(
            f"Action recommandée : <b>{hz['action']}</b>",
            bg=c_bg, border=c_fg
        ))
        s.append(Spacer(1, 0.3*cm))

        # Tableau de synthèse composants
        if diag["component_diagnostics"]:
            s.append(Paragraph("Composants à risque identifiés :", self.sH2))
            rows = []
            for cd in diag["component_diagnostics"]:
                c_h, c_b = HORIZON_COLORS.get(
                    cd["horizon"]["label"], (C_GREY, C_GREY_LT)
                )
                rows.append([
                    cd["composant_nom"][:30],
                    cd["sous_systeme"][:22],
                    f"{cd['score_risque']:.0f}/100",
                    cd["horizon"]["label"],
                    f"{cd['prob_panne_30j']:.0f}%",
                    TENDANCE_FR.get(cd["tendance"], cd["tendance"])[:20],
                ])
            t = self._make_table(
                ["Composant", "Sous-système", "Score",
                 "Horizon", "Panne/30j", "Tendance"],
                rows,
                [4.5*cm, 3.5*cm, 1.8*cm, 2.2*cm, 1.8*cm, 3.6*cm]
            )
            # Coloriser horizon
            for i, cd in enumerate(diag["component_diagnostics"], 1):
                c_h, c_b = HORIZON_COLORS.get(
                    cd["horizon"]["label"], (C_GREY, C_GREY_LT)
                )
                t.setStyle(TableStyle([
                    ("BACKGROUND", (3,i),(3,i), c_b),
                    ("TEXTCOLOR",  (3,i),(3,i), c_h),
                    ("FONTNAME",   (3,i),(3,i), "Helvetica-Bold"),
                ]))
            s.append(t)

    # ── DIAGNOSTIC PAR COMPOSANT ───────────────────────────────────────────
    def _detail_composants(self, diag: dict):
        s = self.story
        s.append(PageBreak())
        s.append(self._sect("2.  Diagnostic Détaillé par Composant"))

        for i, cd in enumerate(diag["component_diagnostics"], 1):
            s.append(Spacer(1, 0.4*cm))
            c_fg, c_bg = HORIZON_COLORS.get(
                cd["horizon"]["label"], (C_GREY, C_GREY_LT)
            )

            # En-tête composant
            ht = Table([[
                Paragraph(
                    f"Composant #{i} — {cd['composant_nom']}",
                    ParagraphStyle("ch", parent=self.sSect, fontSize=10,
                                   textColor=C_WHITE)
                ),
                Paragraph(
                    f"Score {cd['score_risque']:.0f}/100  |  {cd['horizon']['label']}",
                    ParagraphStyle("ch2", parent=self.sSect, fontSize=9,
                                   textColor=c_bg, alignment=TA_RIGHT)
                ),
            ]], colWidths=[self.W*0.65, self.W*0.35])
            ht.setStyle(TableStyle([
                ("BACKGROUND",    (0,0),(-1,-1), c_fg),
                ("TOPPADDING",    (0,0),(-1,-1), 7),
                ("BOTTOMPADDING", (0,0),(-1,-1), 7),
                ("LEFTPADDING",   (0,0),(0,-1),  10),
                ("RIGHTPADDING",  (-1,0),(-1,-1),10),
                ("VALIGN",        (0,0),(-1,-1), "MIDDLE"),
            ]))
            s.append(KeepTogether([ht]))

            # Barre de score
            s.append(Spacer(1, 0.15*cm))
            s.append(_bar_chart(cd["score_risque"], self.W / cm))
            s.append(Spacer(1, 0.15*cm))

            # Propriétés
            tendance_txt = TENDANCE_FR.get(cd["tendance"], cd["tendance"])
            age_txt = (f"{cd['age_ans']:.1f} ans / "
                       f"{cd['duree_vie_ans']} ans (durée de vie)"
                       if cd.get("age_ans") and cd.get("duree_vie_ans")
                       else "Non renseigné")

            prop_rows = [
                ["Sous-système",         cd["sous_systeme"]],
                ["Zone physique",         cd["zone"]],
                ["Score de risque",       f"{cd['score_risque']:.1f} / 100"],
                ["Probabilité panne 30j", f"{cd['prob_panne_30j']:.0f} %"],
                ["Probabilité panne 90j", f"{cd['prob_panne_90j']:.0f} %"],
                ["Tendance historique",   tendance_txt],
                ["Nb apparitions (hist)", str(cd["n_apparitions"])],
                ["Âge du composant",      age_txt],
                ["Coût remplacement",     cd["cout_remplacement"]],
            ]
            pt = Table(prop_rows,
                       colWidths=[4.0*cm, self.W - 4.0*cm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",     (0,0),(0,-1), c_bg),
                ("TEXTCOLOR",      (0,0),(0,-1), c_fg),
                ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
                ("FONTSIZE",       (0,0),(-1,-1), 8.5),
                ("ROWBACKGROUNDS", (1,0),(-1,-1), [C_WHITE, C_GREY_LT]),
                ("GRID",           (0,0),(-1,-1), 0.3,
                 colors.HexColor("#e2e8f0")),
                ("TOPPADDING",     (0,0),(-1,-1), 5),
                ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
                ("LEFTPADDING",    (0,0),(-1,-1), 8),
                ("VALIGN",         (0,0),(-1,-1), "TOP"),
            ]))
            s.append(pt)

            # Actions
            s.append(Spacer(1, 0.2*cm))
            s.append(Paragraph("Actions correctives :", self.sH2))
            for action in cd["actions"]:
                s.append(Paragraph(f"▸  {action}", self.sBullet))

    # ── HISTORIQUE APPAREIL ────────────────────────────────────────────────
    def _historique(self, history_summary: dict):
        s = self.story
        s.append(PageBreak())
        s.append(self._sect("3.  Historique de l'Appareil",
                             color=colors.HexColor("#166534")))
        s.append(Spacer(1, 0.3*cm))

        if history_summary.get("n_analyses", 0) == 0:
            s.append(self._box(
                "Aucun historique disponible pour cet appareil. "
                "Cet enregistrement sera la première entrée.",
                bg=C_YELLOW_L, border=C_YELLOW
            ))
            return

        hs = history_summary
        rows = [
            ["Nombre d'analyses",     str(hs["n_analyses"])],
            ["Première analyse",      hs["premiere_date"]],
            ["Dernière analyse",      hs["derniere_date"]],
            ["Score moyen",           f"{hs['score_moyen']:.1f} / 100"],
            ["Score maximum",         f"{hs['score_max']:.1f} / 100"],
            ["Score actuel",          f"{hs['score_actuel']:.1f} / 100"],
            ["Tendance globale",      TENDANCE_FR.get(hs["tendance_globale"],
                                                       hs["tendance_globale"])],
        ]
        pt = Table(rows, colWidths=[5*cm, self.W - 5*cm])
        pt.setStyle(TableStyle([
            ("BACKGROUND",     (0,0),(0,-1), C_BLUE_PL),
            ("TEXTCOLOR",      (0,0),(0,-1), C_BLUE),
            ("FONTNAME",       (0,0),(0,-1), "Helvetica-Bold"),
            ("FONTSIZE",       (0,0),(-1,-1), 8.5),
            ("ROWBACKGROUNDS", (1,0),(-1,-1), [C_WHITE, C_GREY_LT]),
            ("GRID",           (0,0),(-1,-1), 0.3,
             colors.HexColor("#e2e8f0")),
            ("TOPPADDING",     (0,0),(-1,-1), 5),
            ("BOTTOMPADDING",  (0,0),(-1,-1), 5),
            ("LEFTPADDING",    (0,0),(-1,-1), 8),
        ]))
        s.append(pt)

        if hs.get("top_artefacts"):
            s.append(Spacer(1, 0.3*cm))
            s.append(Paragraph("Artefacts les plus fréquents :", self.sH2))
            for art, count in hs["top_artefacts"]:
                s.append(Paragraph(
                    f"▸  {art}  —  {count} occurrence(s)", self.sBullet
                ))

    # ── FOOTER ────────────────────────────────────────────────────────────
    def _footer(self):
        s = self.story
        s.append(Spacer(1, 0.4*cm))
        s.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#94a3b8")))
        s.append(Spacer(1, 0.2*cm))
        s.append(self._box(
            "Avertissement · Les scores de risque sont calculés automatiquement "
            "par RxArtifactNet v2.0. Toute décision d'arrêt ou d'intervention "
            "sur l'appareil doit être validée par un ingénieur biomédical ou "
            "technicien de maintenance qualifié.",
            bg=C_YELLOW_L, border=C_YELLOW
        ))

    # ── POINT D'ENTRÉE ────────────────────────────────────────────────────
    def build(self, diagnosis: dict, meta: dict,
              history_summary: dict = None) -> str:

        doc = SimpleDocTemplate(
            self.output_path, pagesize=A4,
            leftMargin=self.MARGIN, rightMargin=self.MARGIN,
            topMargin=1.8*cm, bottomMargin=1.8*cm,
            title="Rapport Diagnostic Maintenance — Rx Thorax",
            author="RxArtifactNet v2.0"
        )

        self._garde(diagnosis, meta)
        self._resume(diagnosis)
        self._detail_composants(diagnosis)
        if history_summary:
            self._historique(history_summary)
        self._footer()

        doc.build(self.story)
        return self.output_path
