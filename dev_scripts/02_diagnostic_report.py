"""
02_diagnostic_report.py
========================
Development/validation script for RxArtifactNet's diagnostic engine (level 2).
Simulates 5 successive analyses on the same device to show the rising
risk score and trend calculation.

Not part of the final application (see ../app.py for the deployed GUI).
"""

import os, sys, time, datetime
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

from diagnostic_engine import DiagnosticEngine
from report_niveau2    import DiagnosticReport
from component_knowledge import COMPONENTS

OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs", "rxnet_v2")
HISTORY_DIR = os.path.join(PROJECT_ROOT, "outputs", "rxnet_v2", "historique")

def make_detection(artifact_class, confidence=0.82, mask_area=0.08):
    return {
        "class":      artifact_class,
        "confidence": confidence,
        "mask_area":  mask_area,
    }

def run_simulation():
    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    device_id = "GE_Discovery_XR656_Salle_A"
    engine    = DiagnosticEngine(device_id=device_id,
                                  history_dir=HISTORY_DIR)

    # ── Âges des composants de l'appareil ────────────────────────────────
    component_ages = {
        "anode_tournante":    3.5,
        "stator_roulements":  3.5,
        "filament":           3.5,
        "chambre_ionisation": 2.0,
        "regulateur_ht":      4.0,
        "matrice_tft":        1.5,
        "moteur_bucky":       5.0,
        "grille_pb":          5.0,
    }

    # ── Simulation de 5 analyses successives (aggravation progressive) ──
    scenarios = [
        {
            "patient_id": "PAT_101",
            "artefacts":  [make_detection("grid_lines", 0.65, 0.04)],
            "label":      "Analyse 1 — Lignes de grille légères",
        },
        {
            "patient_id": "PAT_102",
            "artefacts":  [make_detection("grid_lines", 0.78, 0.06),
                           make_detection("anode_artifact", 0.55, 0.03)],
            "label":      "Analyse 2 — Grille + début artefact anodique",
        },
        {
            "patient_id": "PAT_103",
            "artefacts":  [make_detection("grid_lines",    0.88, 0.09),
                           make_detection("anode_artifact", 0.74, 0.07),
                           make_detection("overexposure",   0.60, 0.30)],
            "label":      "Analyse 3 — Aggravation multiple",
        },
        {
            "patient_id": "PAT_104",
            "artefacts":  [make_detection("anode_artifact",   0.91, 0.12),
                           make_detection("focal_spot_defect", 0.82, 0.09),
                           make_detection("overexposure",      0.75, 0.35)],
            "label":      "Analyse 4 — Tube en dégradation critique",
        },
        {
            "patient_id": "PAT_105",
            "artefacts":  [make_detection("anode_artifact",   0.95, 0.18),
                           make_detection("focal_spot_defect", 0.93, 0.14),
                           make_detection("overexposure",      0.88, 0.45),
                           make_detection("grid_lines",        0.90, 0.11)],
            "label":      "Analyse 5 — Situation critique — intervention requise",
        },
    ]

    print("\n" + "═"*65)
    print("  RxArtifactNet v2.0 — Simulation maintenance prédictive")
    print("  Appareil :", device_id)
    print("═"*65)

    last_diag = None
    for i, sc in enumerate(scenarios, 1):
        print(f"\n[{i}/5] {sc['label']}")

        diag = engine.diagnose(
            detections=sc["artefacts"],
            component_ages=component_ages,
            save_history=True,
        )
        last_diag = diag
        sc["diag"] = diag

        # Affichage terminal
        score = diag["score_global"]
        hz    = diag["horizon_global"]["label"]
        print(f"       Score global   : {score:.1f}/100  →  {hz}")
        if diag["composant_prioritaire"]:
            cp = diag["composant_prioritaire"]
            print(f"       Composant n°1  : {cp['composant_nom']}")
            print(f"       Panne 30j       : {cp['prob_panne_30j']:.0f}%  |  "
                  f"Tendance : {cp['tendance']}")

    # ── Rapport PDF du dernier état (le plus critique) ───────────────────
    print("\n[Génération du rapport PDF — Analyse 5]")

    history_summary = engine.get_device_history_summary()

    meta = {
        "patient_id": "PAT_105",
        "device":     device_id,
        "date":       datetime.date.today().strftime("%d/%m/%Y"),
        "operator":   "Système automatique",
    }

    report_path = os.path.join(OUTPUT_DIR, "rapport_diagnostic_niveau2.pdf")
    DiagnosticReport(report_path).build(
        diagnosis=last_diag,
        meta=meta,
        history_summary=history_summary,
    )
    print(f"  Rapport → {report_path}")

    # ── Résumé terminal final ────────────────────────────────────────────
    print("\n" + "═"*65)
    print("  ÉVOLUTION DU SCORE DE RISQUE SUR 5 ANALYSES")
    print("═"*65)
    labels = ["NORMAL","SURVEILLANCE","PLANIFIÉ","URGENT","IMMÉDIAT"]
    for i, sc in enumerate(scenarios, 1):
        score = sc["diag"]["score_global"]
        hz    = sc["diag"]["horizon_global"]["label"]
        bar   = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  Analyse {i}  [{bar}]  {score:5.1f}/100  {hz}")

    print("\n  Historique appareil :")
    hs = history_summary
    print(f"    Analyses enregistrées : {hs['n_analyses']}")
    print(f"    Score actuel          : {hs['score_actuel']:.1f}/100")
    print(f"    Score moyen           : {hs['score_moyen']:.1f}/100")
    print(f"    Tendance globale      : {hs['tendance_globale']}")
    print(f"\n  Rapport PDF : {report_path}")
    print("═"*65 + "\n")

if __name__ == "__main__":
    run_simulation()
