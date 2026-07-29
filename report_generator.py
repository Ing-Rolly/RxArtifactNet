"""
report_generator.py
===================
Génère le rapport PDF de maintenance à partir des résultats RxArtifactNet.
Format professionnel avec :
  - En-tête institution
  - Résumé exécutif
  - Détail par artefact (nature, origine, pièce, maintenance)
  - Plan d'action priorisé
  - Images intégrées (overlay + heatmap)
"""

import os
import datetime
from typing import Dict, List, Optional

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, PageBreak, Image as RLImage, KeepTogether
)
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT, TA_JUSTIFY
from reportlab.platypus import BaseDocTemplate, Frame, PageTemplate

from knowledge_base import get_artifact_info, URGENCE_ORDRE

# ── Palette ───────────────────────────────────────────────────────────────
C_DARK     = colors.HexColor("#0f172a")
C_NAVY     = colors.HexColor("#1e3a5f")
C_BLUE     = colors.HexColor("#2563eb")
C_BLUE_LT  = colors.HexColor("#dbeafe")
C_BLUE_PLT = colors.HexColor("#eff6ff")
C_RED      = colors.HexColor("#dc2626")
C_RED_LT   = colors.HexColor("#fee2e2")
C_ORANGE   = colors.HexColor("#ea580c")
C_ORANGE_LT= colors.HexColor("#ffedd5")
C_YELLOW   = colors.HexColor("#ca8a04")
C_YELLOW_LT= colors.HexColor("#fef9c3")
C_GREEN    = colors.HexColor("#16a34a")
C_GREEN_LT = colors.HexColor("#dcfce7")
C_GREY     = colors.HexColor("#64748b")
C_GREY_LT  = colors.HexColor("#f1f5f9")
C_WHITE    = colors.white

URGENCE_COLOR_MAP = {
    "critique": (C_RED,    C_RED_LT),
    "haute":    (C_ORANGE, C_ORANGE_LT),
    "moyenne":  (C_YELLOW, C_YELLOW_LT),
    "faible":   (C_GREEN,  C_GREEN_LT),
    "aucune":   (C_BLUE,   C_BLUE_PLT),
}

URGENCE_BADGE = {
    "critique": "● CRITIQUE",
    "haute":    "● HAUTE",
    "moyenne":  "● MOYENNE",
    "faible":   "● FAIBLE",
    "aucune":   "● AUCUNE",
}


class RxMaintenanceReport:

    PAGE_W, PAGE_H = A4
    MARGIN = 1.8 * cm
    CONTENT_W = PAGE_W - 2 * MARGIN

    def __init__(self, output_path: str):
        self.output_path = output_path
        self.styles = getSampleStyleSheet()
        self._init_styles()
        self.story = []

    # ── Styles ─────────────────────────────────────────────────────────────
    def _init_styles(self):
        def S(name, **kw):
            parent = kw.pop("parent", "Normal")
            return ParagraphStyle(name, parent=self.styles[parent], **kw)

        self.sTitle    = S("sTitle",  parent="Title",
                           fontSize=20, textColor=C_WHITE, alignment=TA_CENTER,
                           fontName="Helvetica-Bold", spaceAfter=3, leading=24)
        self.sSub      = S("sSub",    fontSize=10, textColor=C_BLUE_LT,
                           alignment=TA_CENTER, spaceAfter=2)
        self.sMeta     = S("sMeta",   fontSize=8.5, textColor=C_BLUE_LT,
                           alignment=TA_CENTER)
        self.sSect     = S("sSect",   fontSize=12, textColor=C_WHITE,
                           fontName="Helvetica-Bold", leading=15)
        self.sH2       = S("sH2",     fontSize=10, textColor=C_NAVY,
                           fontName="Helvetica-Bold", spaceBefore=6,
                           spaceAfter=3)
        self.sBody     = S("sBody",   fontSize=8.5,
                           textColor=colors.HexColor("#1e293b"),
                           leading=13, spaceAfter=3)
        self.sBullet   = S("sBullet", fontSize=8.5,
                           textColor=colors.HexColor("#1e293b"),
                           leading=13, leftIndent=14, bulletIndent=4,
                           spaceAfter=2)
        self.sNote     = S("sNote",   fontSize=8,
                           textColor=colors.HexColor("#475569"),
                           leading=12, leftIndent=8, italic=True)
        self.sTH       = S("sTH",     fontSize=8, textColor=C_WHITE,
                           fontName="Helvetica-Bold", alignment=TA_CENTER,
                           leading=11)
        self.sTD       = S("sTD",     fontSize=8,
                           textColor=colors.HexColor("#1e293b"),
                           leading=11)
        self.sTDc      = S("sTDc",    fontSize=8,
                           textColor=colors.HexColor("#1e293b"),
                           leading=11, alignment=TA_CENTER)
        self.sPageNum  = S("sPageNum",fontSize=7.5, textColor=C_GREY,
                           alignment=TA_RIGHT)

    # ── Helpers ────────────────────────────────────────────────────────────
    def _section_header(self, text: str, color=None) -> Table:
        color = color or C_NAVY
        t = Table([[Paragraph(text, self.sSect)]],
                  colWidths=[self.CONTENT_W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), color),
            ("TOPPADDING",    (0,0), (-1,-1), 8),
            ("BOTTOMPADDING", (0,0), (-1,-1), 8),
            ("LEFTPADDING",   (0,0), (-1,-1), 12),
        ]))
        return t

    def _badge(self, urgence: str) -> str:
        c1, _ = URGENCE_COLOR_MAP.get(urgence, (C_GREY, C_GREY_LT))
        label  = URGENCE_BADGE.get(urgence, urgence.upper())
        # ReportLab color tag
        hex_c  = c1.hexval() if hasattr(c1, "hexval") else "#64748b"
        return f'<font color="{hex_c}"><b>{label}</b></font>'

    def _info_box(self, text: str, bg=None, border=None) -> Table:
        bg     = bg     or C_BLUE_PLT
        border = border or C_BLUE
        t = Table([[Paragraph(text, self.sNote)]],
                  colWidths=[self.CONTENT_W])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), bg),
            ("LINEABOVE",     (0,0), (-1,0),  2, border),
            ("TOPPADDING",    (0,0), (-1,-1), 7),
            ("BOTTOMPADDING", (0,0), (-1,-1), 7),
            ("LEFTPADDING",   (0,0), (-1,-1), 10),
        ]))
        return t

    def _make_table(self, headers, rows, col_widths,
                    header_color=None) -> Table:
        header_color = header_color or C_NAVY
        data = [[Paragraph(h, self.sTH) for h in headers]]
        for row in rows:
            data.append([Paragraph(str(c),
                                   self.sTDc if i == 0 else self.sTD)
                         for i, c in enumerate(row)])
        t = Table(data, colWidths=col_widths, repeatRows=1)
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  header_color),
            ("ROWBACKGROUNDS",(0,1), (-1,-1), [C_WHITE, C_GREY_LT]),
            ("GRID",          (0,0), (-1,-1), 0.4,
             colors.HexColor("#cbd5e1")),
            ("TOPPADDING",    (0,0), (-1,-1), 5),
            ("BOTTOMPADDING", (0,0), (-1,-1), 5),
            ("LEFTPADDING",   (0,0), (-1,-1), 6),
            ("RIGHTPADDING",  (0,0), (-1,-1), 6),
            ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ("LINEBELOW",     (0,0), (-1,0),  1.5, C_BLUE),
        ]))
        return t

    # ── CONSTRUCTION DU RAPPORT ────────────────────────────────────────────

    def _page_de_garde(self, meta: Dict):
        s = self.story
        # Bandeau titre
        cover = Table([
            [Paragraph("RAPPORT DE CONTRÔLE QUALITÉ", self.sSub)],
            [Paragraph("Radiographie Thoracique Conventionnelle", self.sTitle)],
            [Paragraph("Détection automatique d'artefacts &amp; plan de maintenance",
                       self.sSub)],
            [Spacer(1, 0.3*cm)],
            [HRFlowable(width="100%", thickness=1,
                        color=colors.HexColor("#93c5fd"))],
            [Spacer(1, 0.2*cm)],
            [Paragraph(
                f"Patient : <b>{meta.get('patient_id','N/A')}</b>  |  "
                f"Appareil : <b>{meta.get('device','N/A')}</b>  |  "
                f"Date : <b>{meta.get('date', datetime.date.today().strftime('%d/%m/%Y'))}</b>",
                self.sMeta)],
            [Paragraph(f"Analysé par : RxArtifactNet v1.0  |  "
                       f"Opérateur : {meta.get('operator','Automatique')}",
                       self.sMeta)],
        ], colWidths=[self.CONTENT_W])
        cover.setStyle(TableStyle([
            ("BACKGROUND",    (0,0), (-1,-1), C_DARK),
            ("TOPPADDING",    (0,0), (-1,-1), 14),
            ("BOTTOMPADDING", (0,0), (-1,-1), 14),
            ("LEFTPADDING",   (0,0), (-1,-1), 18),
            ("RIGHTPADDING",  (0,0), (-1,-1), 18),
        ]))
        s.append(cover)
        s.append(Spacer(1, 0.5*cm))

    def _resume_executif(self, detections: List[Dict], analysis_time: float):
        s = self.story
        s.append(self._section_header("1.  Résumé Exécutif"))
        s.append(Spacer(1, 0.3*cm))

        n = len(detections)
        if n == 0:
            s.append(self._info_box(
                "✓  Aucun artefact détecté. Le cliché est de qualité conforme "
                "aux standards de la radiologie thoracique conventionnelle. "
                "Poursuite du plan de maintenance préventive standard recommandée.",
                bg=C_GREEN_LT, border=C_GREEN
            ))
        else:
            # Urgence maximale
            max_urg = min(
                [get_artifact_info(d["class"])["urgence"] for d in detections],
                key=lambda u: URGENCE_ORDRE.get(u, 99)
            )
            c_fg, c_bg = URGENCE_COLOR_MAP.get(max_urg, (C_GREY, C_GREY_LT))
            s.append(self._info_box(
                f"⚠  <b>{n} artefact(s) identifié(s)</b> sur ce cliché. "
                f"Urgence maximale de maintenance : <b>{max_urg.upper()}</b>. "
                f"Temps d'analyse : {analysis_time:.2f} s. "
                f"Une intervention technique est recommandée avant la prochaine "
                f"vacation d'imagerie.",
                bg=c_bg, border=c_fg
            ))

        s.append(Spacer(1, 0.3*cm))

        # Tableau de synthèse
        if detections:
            rows = []
            for i, det in enumerate(detections, 1):
                info = get_artifact_info(det["class"])
                c_fg, _ = URGENCE_COLOR_MAP.get(info["urgence"],
                                                 (C_GREY, C_GREY_LT))
                rows.append([
                    str(i),
                    info["label"],
                    info["nature"],
                    info["composant"].split("—")[0].strip(),
                    f"{det['confidence']*100:.1f}%",
                    info["urgence"].upper(),
                ])
            t = self._make_table(
                ["#", "Artefact", "Nature", "Composant concerné",
                 "Confiance", "Urgence"],
                rows,
                [0.6*cm, 3.8*cm, 3.2*cm, 4.0*cm, 1.8*cm, 2.0*cm]
            )
            # Coloriser la colonne urgence
            for i, det in enumerate(detections, 1):
                info = get_artifact_info(det["class"])
                c_fg, c_bg = URGENCE_COLOR_MAP.get(info["urgence"],
                                                    (C_GREY, C_GREY_LT))
                t.setStyle(TableStyle([
                    ("BACKGROUND",  (5, i), (5, i), c_bg),
                    ("TEXTCOLOR",   (5, i), (5, i), c_fg),
                    ("FONTNAME",    (5, i), (5, i), "Helvetica-Bold"),
                ]))
            s.append(t)

        s.append(Spacer(1, 0.4*cm))
        s.append(Paragraph(
            f"Temps d'analyse : {analysis_time:.3f} secondes  |  "
            f"Modèle : RxArtifactNet v1.0  |  "
            f"Généré le {datetime.datetime.now().strftime('%d/%m/%Y à %H:%M')}",
            self.sNote
        ))

    def _section_images(self, figure_path: Optional[str],
                        overlay_path: Optional[str]):
        s = self.story
        s.append(Spacer(1, 0.4*cm))
        s.append(self._section_header("2.  Visualisation des Artefacts"))
        s.append(Spacer(1, 0.3*cm))

        images_inserted = False
        if figure_path and os.path.exists(figure_path):
            img = RLImage(figure_path, width=self.CONTENT_W,
                          height=self.CONTENT_W * 0.78)
            s.append(img)
            s.append(Paragraph(
                "Figure 1 — Analyse complète : image originale, segmentation "
                "des artefacts, carte de chaleur et probabilités par classe.",
                self.sNote
            ))
            images_inserted = True

        if not images_inserted:
            s.append(self._info_box(
                "Les images de visualisation n'ont pas pu être générées. "
                "Consulter les fichiers d'overlay exportés séparément.",
                bg=C_YELLOW_LT, border=C_YELLOW
            ))

    def _section_detail_artefacts(self, detections: List[Dict]):
        s = self.story
        s.append(PageBreak())
        s.append(self._section_header("3.  Analyse Détaillée par Artefact"))

        if not detections:
            s.append(Spacer(1, 0.3*cm))
            s.append(self._info_box(
                "Aucun artefact détecté — section non applicable.",
                bg=C_GREEN_LT, border=C_GREEN
            ))
            return

        for idx, det in enumerate(detections, 1):
            info   = get_artifact_info(det["class"])
            urgence = info["urgence"]
            c_fg, c_bg = URGENCE_COLOR_MAP.get(urgence, (C_GREY, C_GREY_LT))

            s.append(Spacer(1, 0.4*cm))

            # ── En-tête artefact ──────────────────────────────────────────
            header_data = [[
                Paragraph(
                    f"Artefact #{idx} — {info['label']}",
                    ParagraphStyle("ah", parent=self.sSect, fontSize=10,
                                   textColor=C_WHITE)
                ),
                Paragraph(
                    f"{urgence.upper()}  ({det['confidence']*100:.1f}%)",
                    ParagraphStyle("au", parent=self.sSect, fontSize=10,
                                   textColor=c_bg, alignment=TA_RIGHT)
                ),
            ]]
            ht = Table(header_data,
                       colWidths=[self.CONTENT_W * 0.65,
                                   self.CONTENT_W * 0.35])
            ht.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (-1,-1), c_fg),
                ("TOPPADDING",    (0,0), (-1,-1), 7),
                ("BOTTOMPADDING", (0,0), (-1,-1), 7),
                ("LEFTPADDING",   (0,0), (0,-1),  10),
                ("RIGHTPADDING",  (-1,0), (-1,-1), 10),
                ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ]))
            s.append(KeepTogether([ht]))

            # ── Tableau des propriétés ────────────────────────────────────
            prop_rows = [
                ["Nature",    info["nature"]],
                ["Origine",   info["origine"]],
                ["Composant", info["composant"]],
                ["Gravité",   info["gravite"].capitalize()],
                ["Surface masque",
                 f"{det['mask_area']*100:.2f}% de l'image"],
            ]
            pt = Table(prop_rows,
                       colWidths=[3.0*cm, self.CONTENT_W - 3.0*cm])
            pt.setStyle(TableStyle([
                ("BACKGROUND",    (0,0), (0,-1), c_bg),
                ("TEXTCOLOR",     (0,0), (0,-1), c_fg),
                ("FONTNAME",      (0,0), (0,-1), "Helvetica-Bold"),
                ("FONTSIZE",      (0,0), (-1,-1), 8.5),
                ("ROWBACKGROUNDS",(1,0), (-1,-1),
                 [C_WHITE, C_GREY_LT]),
                ("GRID",          (0,0), (-1,-1), 0.3,
                 colors.HexColor("#e2e8f0")),
                ("TOPPADDING",    (0,0), (-1,-1), 5),
                ("BOTTOMPADDING", (0,0), (-1,-1), 5),
                ("LEFTPADDING",   (0,0), (-1,-1), 8),
                ("VALIGN",        (0,0), (-1,-1), "TOP"),
            ]))
            s.append(pt)

            # ── Actions de maintenance ─────────────────────────────────────
            s.append(Spacer(1, 0.2*cm))
            s.append(Paragraph(
                "Actions de maintenance recommandées :", self.sH2
            ))
            for action in info["maintenance"]:
                s.append(Paragraph(f"▸  {action}", self.sBullet))

    def _section_plan_action(self, detections: List[Dict]):
        s = self.story
        s.append(PageBreak())
        s.append(self._section_header(
            "4.  Plan d'Action Maintenance — Priorisé",
            color=colors.HexColor("#166534")
        ))
        s.append(Spacer(1, 0.3*cm))

        if not detections:
            s.append(self._info_box(
                "Aucun artefact détecté. Plan de maintenance préventive "
                "standard à poursuivre (cf. section 5).",
                bg=C_GREEN_LT, border=C_GREEN
            ))
        else:
            # Trier par urgence
            sorted_dets = sorted(
                detections,
                key=lambda d: URGENCE_ORDRE.get(
                    get_artifact_info(d["class"])["urgence"], 99
                )
            )
            rows = []
            for i, det in enumerate(sorted_dets, 1):
                info = get_artifact_info(det["class"])
                actions_txt = "\n".join(
                    [f"  {j+1}. {a}"
                     for j, a in enumerate(info["maintenance"])]
                )
                rows.append([
                    str(i),
                    info["urgence"].upper(),
                    info["label"],
                    info["composant"].split("—")[-1].strip()
                    if "—" in info["composant"]
                    else info["composant"],
                    "\n".join(info["maintenance"]),
                ])

            t = self._make_table(
                ["#", "Urgence", "Artefact",
                 "Pièce / Zone", "Actions à réaliser"],
                rows,
                [0.6*cm, 1.8*cm, 3.0*cm, 3.0*cm, 9.0*cm]
            )
            # Coloriser urgence
            for i, det in enumerate(sorted_dets, 1):
                info = get_artifact_info(det["class"])
                c_fg, c_bg = URGENCE_COLOR_MAP.get(info["urgence"],
                                                    (C_GREY, C_GREY_LT))
                t.setStyle(TableStyle([
                    ("BACKGROUND", (1,i), (1,i), c_bg),
                    ("TEXTCOLOR",  (1,i), (1,i), c_fg),
                    ("FONTNAME",   (1,i), (1,i), "Helvetica-Bold"),
                ]))
            s.append(t)

        s.append(Spacer(1, 0.5*cm))

    def _section_maintenance_preventive(self):
        s = self.story
        s.append(self._section_header("5.  Plan de Maintenance Préventive Standard"))
        s.append(Spacer(1, 0.3*cm))

        freq_rows = [
            ["Quotidienne",
             "Dark field DR automatique • Effacement plaques CR • "
             "Contrôle visuel console • Check-list patient"],
            ["Hebdomadaire",
             "Flat-field DR • Nettoyage plaques CR • Test rémanence "
             "• Oscillation Bucky"],
            ["Mensuelle",
             "Test résolution foyer (outil stellaire) • Contrôle AEC "
             "PMMA • Puissance laser CR • Lampe UV effacement"],
            ["Trimestrielle",
             "Coïncidence champ RX/lumineux • Centrage grille • "
             "Chambre ionisation AEC • Optique lecteur CR"],
            ["Semestrielle",
             "kV/mAs/temps (CV < 5%) • DQE/pixels morts DR • "
             "Inspection anode/stator • Test AEC dosimétrique"],
            ["Annuelle",
             "Isolation câbles HT • Audit dosimétrique complet (PMSI) • "
             "MTF capteur • LUTs et protocoles PACS • Lampe collimateur"],
        ]

        freq_colors_bg = {
            "Quotidienne":   C_GREEN_LT,
            "Hebdomadaire":  C_BLUE_PLT,
            "Mensuelle":     colors.HexColor("#fef9c3"),
            "Trimestrielle": colors.HexColor("#fce7f3"),
            "Semestrielle":  colors.HexColor("#ede9fe"),
            "Annuelle":      C_ORANGE_LT,
        }

        header_row = [Paragraph(h, self.sTH)
                      for h in ["Fréquence", "Actions"]]
        data = [header_row]
        for freq, actions in freq_rows:
            data.append([
                Paragraph(freq, ParagraphStyle(
                    "freq", parent=self.sTDc,
                    fontName="Helvetica-Bold", fontSize=8.5
                )),
                Paragraph(actions, self.sTD),
            ])

        t = Table(data, colWidths=[3.2*cm, self.CONTENT_W - 3.2*cm],
                  repeatRows=1)
        ts = TableStyle([
            ("BACKGROUND",    (0,0), (-1,0),  C_NAVY),
            ("GRID",          (0,0), (-1,-1), 0.4,
             colors.HexColor("#cbd5e1")),
            ("TOPPADDING",    (0,0), (-1,-1), 6),
            ("BOTTOMPADDING", (0,0), (-1,-1), 6),
            ("LEFTPADDING",   (0,0), (-1,-1), 8),
            ("VALIGN",        (0,0), (-1,-1), "MIDDLE"),
            ("LINEBELOW",     (0,0), (-1,0),  1.5, C_BLUE),
        ])
        for i, (freq, _) in enumerate(freq_rows, 1):
            bg = freq_colors_bg.get(freq, C_WHITE)
            ts.add("BACKGROUND", (0,i), (-1,i), bg)
        t.setStyle(ts)
        s.append(t)
        s.append(Spacer(1, 0.4*cm))

    def _footer_notes(self):
        s = self.story
        s.append(HRFlowable(width="100%", thickness=0.5,
                             color=colors.HexColor("#94a3b8")))
        s.append(Spacer(1, 0.2*cm))
        s.append(self._info_box(
            "Avertissement  ·  Ce rapport est généré automatiquement par "
            "RxArtifactNet. Les résultats doivent être validés par un "
            "radiophysicien ou technicien de maintenance qualifié avant "
            "toute intervention. Les probabilités de détection sont "
            "indicatives et dépendent de la qualité du cliché soumis.",
            bg=C_YELLOW_LT, border=C_YELLOW
        ))
        s.append(Spacer(1, 0.2*cm))
        s.append(self._info_box(
            "Références  ·  IAEA Quality Control in Diagnostic Radiology (2011) "
            "— IPEM Report 91 — Recommandations SFR/SFPM sur le contrôle qualité "
            "en radiologie numérique — IEC 62220-1 (DQE) — EN 61223-3-1.",
            bg=C_BLUE_PLT, border=C_BLUE
        ))

    # ── POINT D'ENTRÉE ──────────────────────────────────────────────────────

    def build(self,
              detections: List[Dict],
              meta: Dict,
              figure_path: Optional[str] = None,
              overlay_path: Optional[str] = None,
              analysis_time: float = 0.0) -> str:
        """
        Construit et sauvegarde le rapport PDF.

        Parameters
        ----------
        detections   : liste des artefacts détectés (sortie du modèle)
        meta         : dict avec patient_id, device, date, operator
        figure_path  : chemin vers la figure matplotlib (optionnel)
        overlay_path : chemin vers l'image overlay cv2 (optionnel)
        analysis_time: durée de l'inférence en secondes
        """
        doc = SimpleDocTemplate(
            self.output_path,
            pagesize=A4,
            leftMargin=self.MARGIN, rightMargin=self.MARGIN,
            topMargin=1.8*cm, bottomMargin=1.8*cm,
            title="Rapport Artefacts Rx Thorax",
            author="RxArtifactNet v1.0"
        )

        self._page_de_garde(meta)
        self._resume_executif(detections, analysis_time)
        self._section_images(figure_path, overlay_path)
        self._section_detail_artefacts(detections)
        self._section_plan_action(detections)
        self._section_maintenance_preventive()
        self._footer_notes()

        doc.build(self.story)
        return self.output_path
