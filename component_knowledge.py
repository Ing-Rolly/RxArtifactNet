"""
component_knowledge.py
======================
Base de connaissances niveau 2 — VERSION CORRIGÉE.
Alignée strictement sur les 12 artefacts machine retenus.

Mapping : artefact machine → composants physiques (probabilité décroissante)
"""

from knowledge_base import CLASS_NAMES

# ══════════════════════════════════════════════════════════════════════════
# 22 COMPOSANTS PHYSIQUES COUVERTS
# ══════════════════════════════════════════════════════════════════════════

COMPONENTS = {

    # ── GÉNÉRATEUR HAUTE TENSION ──────────────────────────────────────────
    "regulateur_ht": {
        "nom":           "Régulateur haute tension",
        "sous_systeme":  "Générateur HT",
        "zone":          "Armoire technique / baie générateur",
        "criticite":     95,
        "duree_vie_ans": 10,
        "cout_remplacement": "Très élevé (15 000–50 000 €)",
        "mtbf_heures":   40000,
    },
    "condensateurs_ht": {
        "nom":           "Condensateurs haute tension",
        "sous_systeme":  "Générateur HT",
        "zone":          "Armoire technique / baie générateur",
        "criticite":     90,
        "duree_vie_ans": 8,
        "cout_remplacement": "Élevé (3 000–8 000 €)",
        "mtbf_heures":   35000,
    },
    "chambre_ionisation": {
        "nom":           "Chambre d'ionisation (CAE/AEC)",
        "sous_systeme":  "Contrôle automatique exposition",
        "zone":          "Sous la table / devant le capteur",
        "criticite":     85,
        "duree_vie_ans": 7,
        "cout_remplacement": "Moyen (1 500–4 000 €)",
        "mtbf_heures":   30000,
    },
    "onduleur": {
        "nom":           "Onduleur / alimentation",
        "sous_systeme":  "Générateur HT",
        "zone":          "Armoire technique",
        "criticite":     80,
        "duree_vie_ans": 6,
        "cout_remplacement": "Moyen (2 000–5 000 €)",
        "mtbf_heures":   25000,
    },

    # ── TUBE RADIOGÈNE ────────────────────────────────────────────────────
    "anode_tournante": {
        "nom":           "Anode tournante",
        "sous_systeme":  "Tube radiogène",
        "zone":          "Statif / bras porte-tube",
        "criticite":     100,
        "duree_vie_ans": 5,
        "cout_remplacement": "Très élevé (8 000–25 000 €)",
        "mtbf_heures":   20000,
    },
    "filament": {
        "nom":           "Filament (cathode)",
        "sous_systeme":  "Tube radiogène",
        "zone":          "Statif / bras porte-tube",
        "criticite":     100,
        "duree_vie_ans": 5,
        "cout_remplacement": "Très élevé (8 000–25 000 €)",
        "mtbf_heures":   20000,
    },
    "stator_roulements": {
        "nom":           "Stator / roulements anodiques",
        "sous_systeme":  "Tube radiogène",
        "zone":          "Statif / bras porte-tube",
        "criticite":     95,
        "duree_vie_ans": 4,
        "cout_remplacement": "Très élevé (8 000–25 000 €)",
        "mtbf_heures":   18000,
    },

    # ── COLLIMATEUR ───────────────────────────────────────────────────────
    "lames_collimateur": {
        "nom":           "Lames Pb du collimateur",
        "sous_systeme":  "Collimateur / diaphragme",
        "zone":          "Tête du tube — collimateur",
        "criticite":     65,
        "duree_vie_ans": 15,
        "cout_remplacement": "Faible (500–1 500 €)",
        "mtbf_heures":   80000,
    },
    "moteur_collimateur": {
        "nom":           "Moteur d'entraînement des lames",
        "sous_systeme":  "Collimateur / diaphragme",
        "zone":          "Tête du tube — collimateur",
        "criticite":     60,
        "duree_vie_ans": 12,
        "cout_remplacement": "Faible (300–800 €)",
        "mtbf_heures":   60000,
    },
    "miroir_centrage": {
        "nom":           "Miroir et lampe de centrage",
        "sous_systeme":  "Collimateur / diaphragme",
        "zone":          "Tête du tube — collimateur",
        "criticite":     40,
        "duree_vie_ans": 5,
        "cout_remplacement": "Très faible (50–200 €)",
        "mtbf_heures":   15000,
    },

    # ── GRILLE BUCKY ──────────────────────────────────────────────────────
    "moteur_bucky": {
        "nom":           "Moteur d'oscillation du Bucky",
        "sous_systeme":  "Grille anti-diffusante",
        "zone":          "Tiroir porte-grille / table",
        "criticite":     70,
        "duree_vie_ans": 8,
        "cout_remplacement": "Moyen (800–2 000 €)",
        "mtbf_heures":   30000,
    },
    "grille_pb": {
        "nom":           "Corps de grille (lamelles Pb)",
        "sous_systeme":  "Grille anti-diffusante",
        "zone":          "Tiroir porte-grille / table",
        "criticite":     60,
        "duree_vie_ans": 10,
        "cout_remplacement": "Moyen (500–1 500 €)",
        "mtbf_heures":   50000,
    },
    "support_grille": {
        "nom":           "Support / fixation de grille",
        "sous_systeme":  "Grille anti-diffusante",
        "zone":          "Tiroir porte-grille / table",
        "criticite":     50,
        "duree_vie_ans": 15,
        "cout_remplacement": "Faible (200–600 €)",
        "mtbf_heures":   70000,
    },

    # ── CAPTEUR DR ───────────────────────────────────────────────────────
    "matrice_tft": {
        "nom":           "Matrice TFT (transistors)",
        "sous_systeme":  "Capteur DR Flat Panel",
        "zone":          "Bucky numérique / statif mural",
        "criticite":     90,
        "duree_vie_ans": 8,
        "cout_remplacement": "Très élevé (20 000–80 000 €)",
        "mtbf_heures":   50000,
    },
    "scintillateur_csi": {
        "nom":           "Scintillateur CsI",
        "sous_systeme":  "Capteur DR Flat Panel",
        "zone":          "Bucky numérique / statif mural",
        "criticite":     85,
        "duree_vie_ans": 10,
        "cout_remplacement": "Très élevé (20 000–80 000 €)",
        "mtbf_heures":   60000,
    },
    "electronique_lecture_dr": {
        "nom":           "Électronique de lecture DR",
        "sous_systeme":  "Capteur DR Flat Panel",
        "zone":          "Bucky numérique / statif mural",
        "criticite":     80,
        "duree_vie_ans": 8,
        "cout_remplacement": "Élevé (5 000–15 000 €)",
        "mtbf_heures":   40000,
    },

    # ── CAPTEUR CR ───────────────────────────────────────────────────────
    "plaque_ip": {
        "nom":           "Plaque phosphore IP",
        "sous_systeme":  "Lecteur CR",
        "zone":          "Cassette / lecteur CR",
        "criticite":     40,
        "duree_vie_ans": 3,
        "cout_remplacement": "Faible (150–400 € / plaque)",
        "mtbf_heures":   None,
    },
    "laser_lecture": {
        "nom":           "Diode laser de lecture",
        "sous_systeme":  "Lecteur CR",
        "zone":          "Lecteur CR (intérieur)",
        "criticite":     75,
        "duree_vie_ans": 5,
        "cout_remplacement": "Moyen (1 000–3 000 €)",
        "mtbf_heures":   20000,
    },
    "lampe_uv": {
        "nom":           "Lampe UV d'effacement",
        "sous_systeme":  "Lecteur CR",
        "zone":          "Lecteur CR (intérieur)",
        "criticite":     60,
        "duree_vie_ans": 3,
        "cout_remplacement": "Faible (100–300 €)",
        "mtbf_heures":   8000,
    },
    "guide_transport": {
        "nom":           "Guide de transport plaque",
        "sous_systeme":  "Lecteur CR",
        "zone":          "Lecteur CR (mécanique)",
        "criticite":     45,
        "duree_vie_ans": 6,
        "cout_remplacement": "Faible (200–500 €)",
        "mtbf_heures":   25000,
    },

    # ── CONSOLE / LOGICIEL ────────────────────────────────────────────────
    "lut_logiciel": {
        "nom":           "LUT / algorithme post-traitement",
        "sous_systeme":  "Console d'acquisition",
        "zone":          "Poste de travail / serveur",
        "criticite":     50,
        "duree_vie_ans": None,
        "cout_remplacement": "Faible (recalibration / mise à jour)",
        "mtbf_heures":   None,
    },
    "serveur_pacs": {
        "nom":           "Serveur PACS / compression",
        "sous_systeme":  "Système d'information radiologique",
        "zone":          "Salle serveurs",
        "criticite":     55,
        "duree_vie_ans": 7,
        "cout_remplacement": "Élevé (5 000–20 000 €)",
        "mtbf_heures":   50000,
    },
}

# ══════════════════════════════════════════════════════════════════════════
# MAPPING ARTEFACT → COMPOSANTS
# Strictement aligné sur les 12 artefacts machine retenus
# ══════════════════════════════════════════════════════════════════════════

ARTIFACT_TO_COMPONENTS = {

    # Générateur HT
    "overexposure": [
        ("chambre_ionisation", 0.45),
        ("regulateur_ht",      0.30),
        ("condensateurs_ht",   0.15),
        ("onduleur",           0.10),
    ],
    "underexposure": [
        ("regulateur_ht",      0.40),
        ("onduleur",           0.30),
        ("chambre_ionisation", 0.20),
        ("condensateurs_ht",   0.10),
    ],

    # Tube radiogène
    "focal_spot_defect": [
        ("anode_tournante",    0.50),
        ("filament",           0.35),
        ("stator_roulements",  0.15),
    ],
    "anode_artifact": [
        ("stator_roulements",  0.55),
        ("anode_tournante",    0.35),
        ("filament",           0.10),
    ],

    # Collimateur
    "collimator_misalignment": [
        ("lames_collimateur",  0.40),
        ("moteur_collimateur", 0.35),
        ("miroir_centrage",    0.25),
    ],

    # Grille Bucky
    "grid_lines": [
        ("moteur_bucky",       0.70),
        ("grille_pb",          0.20),
        ("support_grille",     0.10),
    ],
    "grid_cutoff": [
        ("support_grille",     0.55),
        ("grille_pb",          0.30),
        ("moteur_bucky",       0.15),
    ],

    # Capteur DR
    "dead_pixels": [
        ("matrice_tft",              0.60),
        ("scintillateur_csi",        0.30),
        ("electronique_lecture_dr",  0.10),
    ],
    "ghosting_dr": [
        ("electronique_lecture_dr",  0.65),
        ("matrice_tft",              0.25),
        ("scintillateur_csi",        0.10),
    ],

    # Capteur CR
    "cr_scratch": [
        ("plaque_ip",          0.60),
        ("guide_transport",    0.40),
    ],
    "cr_fog": [
        ("lampe_uv",           0.65),
        ("plaque_ip",          0.25),
        ("laser_lecture",      0.10),
    ],

    # Console / PACS
    "processing_artifact": [
        ("lut_logiciel",       0.60),
        ("serveur_pacs",       0.40),
    ],
}

# ── Vérification cohérence ────────────────────────────────────────────────
_missing = [c for c in CLASS_NAMES if c not in ARTIFACT_TO_COMPONENTS]
assert not _missing, f"Artefacts sans mapping composant : {_missing}"

# ── Matrice de risque ─────────────────────────────────────────────────────
RISK_HORIZONS = [
    (85, 100, "IMMÉDIAT",     "Arrêt de l'appareil recommandé avant le prochain patient",  "rouge"),
    (65,  84, "URGENT",       "Intervention dans les 48–72 heures",                         "orange"),
    (45,  64, "PLANIFIÉ",     "Planifier l'intervention dans les 2–4 semaines",             "jaune"),
    (20,  44, "SURVEILLANCE", "Renforcer la fréquence des contrôles qualité",               "bleu"),
    ( 0,  19, "NORMAL",       "Poursuite du plan de maintenance préventive standard",       "vert"),
]

def get_risk_horizon(score: float) -> dict:
    for lo, hi, label, action, couleur in RISK_HORIZONS:
        if lo <= score <= hi:
            return {"label": label, "action": action,
                    "couleur": couleur, "score": score}
    return {"label": "NORMAL", "action": "Maintenance standard",
            "couleur": "vert", "score": score}
