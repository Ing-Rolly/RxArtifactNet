"""
knowledge_base.py
=================
Base de connaissances RxArtifactNet — VERSION CORRIGÉE.

12 artefacts machine UNIQUEMENT (artefacts patient exclus).
Chaque artefact est lié à : composant → zone physique → urgence → maintenance.
"""

# ══════════════════════════════════════════════════════════════════════════
# 12 ARTEFACTS MACHINE RETENUS
# ══════════════════════════════════════════════════════════════════════════

ARTIFACT_KNOWLEDGE = {

    # ── GÉNÉRATEUR HAUTE TENSION ──────────────────────────────────────────
    "overexposure": {
        "label":      "Surexposition systématique",
        "nature":     "Artefact de dose — signal trop élevé",
        "origine":    "Dérive des paramètres kV/mAs ou défaillance CAE/AEC",
        "composant":  "Générateur HT — régulateur de tension / chambre d'ionisation",
        "zone":       "Armoire technique / baie générateur",
        "gravite":    "critique",
        "urgence":    "haute",
        "couleur_viz": (255, 80, 80),
        "maintenance": [
            "Mesurer la reproductibilité kV/mAs avec un dosimètre (CV < 5 %)",
            "Tester la chambre d'ionisation AEC avec fantôme PMMA",
            "Vérifier les condensateurs HT et l'onduleur",
            "Recalibrer le seuil de déclenchement AEC",
        ],
    },
    "underexposure": {
        "label":      "Sous-exposition systématique",
        "nature":     "Artefact de dose — signal insuffisant",
        "origine":    "Dérive des paramètres kV/mAs — tension de sortie instable",
        "composant":  "Générateur HT — régulateur / AEC",
        "zone":       "Armoire technique / baie générateur",
        "gravite":    "critique",
        "urgence":    "haute",
        "couleur_viz": (100, 100, 255),
        "maintenance": [
            "Contrôle dosimétrique semestriel (kV, mAs, temps de pose)",
            "Tester reproductibilité avec fantôme PMMA et dosimètre",
            "Vérifier circuit de régulation et amplificateur AEC",
        ],
    },

    # ── TUBE RADIOGÈNE ────────────────────────────────────────────────────
    "focal_spot_defect": {
        "label":      "Défaut de foyer (tache focale)",
        "nature":     "Dégradation du tube — foyer usé ou fissuré",
        "origine":    "Anode usée, fissurée ou contaminée (vaporisation tungstène)",
        "composant":  "Tube radiogène — anode tournante / filament",
        "zone":       "Statif / bras porte-tube",
        "gravite":    "critique",
        "urgence":    "critique",
        "couleur_viz": (200, 0, 200),
        "maintenance": [
            "Mesurer la taille du foyer avec l'outil stellaire (semestriel)",
            "Effectuer la courbe MTF / test de résolution spatiale",
            "Inspecter visuellement l'ampoule (noircissement = remplacement immédiat)",
            "Écouter le bruit de rotation de l'anode (roulement usé)",
            "Planifier le remplacement du tube si dégradation confirmée",
        ],
    },
    "anode_artifact": {
        "label":      "Artefact anodique (bandes)",
        "nature":     "Dégradation mécanique du tube",
        "origine":    "Vitesse de rotation anodique irrégulière — anode voilée ou fissurée",
        "composant":  "Tube radiogène — stator / roulements / disque anodique",
        "zone":       "Statif / bras porte-tube",
        "gravite":    "critique",
        "urgence":    "critique",
        "couleur_viz": (180, 0, 180),
        "maintenance": [
            "Mesure du bruit de rotation du stator",
            "Test de montée en vitesse à vide",
            "Inspection thermique du tube après série d'acquisitions",
            "Remplacement du tube si vibrations ou bruit anormal confirmés",
        ],
    },

    # ── COLLIMATEUR / DIAPHRAGME ──────────────────────────────────────────
    "collimator_misalignment": {
        "label":      "Désalignement du collimateur",
        "nature":     "Artefact géométrique — champ RX ≠ champ lumineux",
        "origine":    "Lames Pb mal alignées ou coincées",
        "composant":  "Collimateur / diaphragme — lames Pb / moteur",
        "zone":       "Tête du tube — collimateur",
        "gravite":    "modérée",
        "urgence":    "moyenne",
        "couleur_viz": (0, 200, 200),
        "maintenance": [
            "Test de coïncidence champ lumineux vs champ RX (tolérance ≤ 2 % DFP)",
            "Inspection et lubrification des lames Pb",
            "Remplacement ressorts ou moteur si lame coincée",
            "Vérification optique (miroir, lampe de centrage) annuelle",
        ],
    },

    # ── GRILLE ANTI-DIFFUSANTE (BUCKY) ────────────────────────────────────
    "grid_lines": {
        "label":      "Lignes de grille visibles",
        "nature":     "Artefact de grille — oscillation bloquée",
        "origine":    "Moteur d'oscillation Bucky bloqué ou grille fixe utilisée",
        "composant":  "Grille anti-diffusante — mécanisme Bucky / moteur d'oscillation",
        "zone":       "Tiroir porte-grille / table",
        "gravite":    "modérée",
        "urgence":    "moyenne",
        "couleur_viz": (0, 180, 100),
        "maintenance": [
            "Tester l'oscillation du Bucky (écoute moteur, mesure vitesse)",
            "Vérifier l'alignement et le centrage de la grille",
            "Contrôler le rapport de grille vs kV utilisé",
            "Inspecter les lamelles Pb (choc = remplacement grille)",
        ],
    },
    "grid_cutoff": {
        "label":      "Coupure de grille (cut-off)",
        "nature":     "Artefact de grille — densification latérale",
        "origine":    "Grille décentrée ou inclinée par rapport à l'axe du faisceau",
        "composant":  "Grille anti-diffusante — support / fixation",
        "zone":       "Tiroir porte-grille / table",
        "gravite":    "modérée",
        "urgence":    "moyenne",
        "couleur_viz": (0, 220, 130),
        "maintenance": [
            "Vérifier la verticalité et le centrage de la grille",
            "Contrôler la distance focale grille (DFP nominale vs utilisée)",
            "Resserrer la fixation mécanique du support de grille",
        ],
    },

    # ── CAPTEUR NUMÉRIQUE DR (Flat Panel) ────────────────────────────────
    "dead_pixels": {
        "label":      "Pixels morts / lignes mortes",
        "nature":     "Défaut capteur DR — matrice TFT endommagée",
        "origine":    "Défaillance de la matrice TFT ou de la couche de conversion",
        "composant":  "Détecteur DR — matrice TFT / scintillateur CsI",
        "zone":       "Bucky numérique / statif mural",
        "gravite":    "élevée",
        "urgence":    "haute",
        "couleur_viz": (255, 50, 150),
        "maintenance": [
            "Cartographier les pixels défectueux (outil constructeur)",
            "Comparer au seuil acceptable (DQE) — remplacer si dépassé",
            "Effectuer une calibration flat-field hebdomadaire",
            "Programmer le remplacement du capteur si dégradation progressive",
        ],
    },
    "ghosting_dr": {
        "label":      "Rémanence / image fantôme DR",
        "nature":     "Défaut capteur DR — recharge incomplète",
        "origine":    "Recharge incomplète du capteur entre expositions (couche a-Se / a-Si)",
        "composant":  "Détecteur DR — électronique de lecture",
        "zone":       "Bucky numérique / statif mural",
        "gravite":    "modérée",
        "urgence":    "moyenne",
        "couleur_viz": (200, 100, 200),
        "maintenance": [
            "Appliquer une exposition d'effacement (scrubbing) entre patients",
            "Vérifier le paramétrage du temps de recharge du capteur",
            "Contrôler l'acquisition dark field automatique quotidienne",
        ],
    },

    # ── CAPTEUR NUMÉRIQUE CR (Plaque photostimulable) ─────────────────────
    "cr_scratch": {
        "label":      "Rayures linéaires CR",
        "nature":     "Défaut plaque photostimulable ou lecteur",
        "origine":    "Poussières ou rayures sur la plaque IP ou dans le lecteur CR",
        "composant":  "Plaque IP — guide de transport du lecteur CR",
        "zone":       "Cassette / lecteur CR",
        "gravite":    "modérée",
        "urgence":    "faible",
        "couleur_viz": (150, 220, 50),
        "maintenance": [
            "Nettoyage hebdomadaire des plaques IP avec chiffon non pelucheux",
            "Inspection mensuelle du guide de transport du lecteur",
            "Réformer les plaques présentant des rayures profondes",
            "Rotation systématique des plaques (durée de vie ~3000 cycles)",
        ],
    },
    "cr_fog": {
        "label":      "Voile uniforme CR",
        "nature":     "Défaut d'effacement CR — lampe UV défaillante",
        "origine":    "Effacement UV incomplet ou exposition parasite de la plaque",
        "composant":  "Unité d'effacement UV du lecteur CR",
        "zone":       "Lecteur CR (intérieur)",
        "gravite":    "modérée",
        "urgence":    "moyenne",
        "couleur_viz": (200, 200, 50),
        "maintenance": [
            "Vérifier la puissance de la lampe UV d'effacement",
            "Contrôler la durée du cycle d'effacement",
            "Stocker les plaques à l'abri de la lumière ambiante",
            "Remplacer la lampe UV si vieillissement confirmé",
        ],
    },

    # ── TRAITEMENT NUMÉRIQUE (Console / PACS) ─────────────────────────────
    "processing_artifact": {
        "label":      "Artefact de traitement numérique",
        "nature":     "Erreur logicielle — LUT ou algorithme mal calibré",
        "origine":    "Algorithme post-traitement mal calibré, fenêtrage inadapté, LUT erronée",
        "composant":  "Console d'acquisition — logiciel / LUT / PACS",
        "zone":       "Poste de travail / serveur",
        "gravite":    "modérée",
        "urgence":    "moyenne",
        "couleur_viz": (100, 150, 255),
        "maintenance": [
            "Recalibrer les présets de traitement (fenêtrage, LUT)",
            "Vérifier et verrouiller les tables de correspondance (LUT)",
            "Contrôler le taux de compression DICOM (préférer lossless)",
            "Mettre à jour le logiciel d'acquisition",
        ],
    },
}

# ── Constantes globales ────────────────────────────────────────────────────
CLASS_NAMES  = list(ARTIFACT_KNOWLEDGE.keys())
NUM_CLASSES  = len(CLASS_NAMES)          # 12

URGENCE_ORDRE = {
    "critique": 0,
    "haute":    1,
    "moyenne":  2,
    "faible":   3,
    "aucune":   4,
}

def get_artifact_info(class_name: str) -> dict:
    """Retourne les infos d'un artefact, ou un artefact vide si inconnu."""
    return ARTIFACT_KNOWLEDGE.get(class_name, {
        "label":      "Inconnu",
        "nature":     "—",
        "origine":    "—",
        "composant":  "—",
        "zone":       "—",
        "gravite":    "inconnue",
        "urgence":    "aucune",
        "couleur_viz": (100, 100, 100),
        "maintenance": ["Contacter le service technique"],
    })

def get_machine_artifacts() -> list:
    """Retourne la liste des 12 noms de classes machine."""
    return CLASS_NAMES

# ── Vérification au chargement ────────────────────────────────────────────
assert NUM_CLASSES == 12, f"Attendu 12 artefacts machine, trouvé {NUM_CLASSES}"
