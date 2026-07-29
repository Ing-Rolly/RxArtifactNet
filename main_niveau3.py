"""
main_niveau3.py
===============
Démonstration complète du système niveau 3.
Simule 8 analyses sur 3 mois pour montrer la montée progressive du risque,
la prédiction temporelle et le déclenchement des alertes.
"""

import os, sys, json, time, datetime
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from diagnostic_engine  import DiagnosticEngine
from predictive_engine  import PredictiveEngine
from report_niveau2     import DiagnosticReport
from report_niveau3     import (PredictiveReport, plot_degradation_curves,
                                 plot_timeline, plot_score_history)
from component_knowledge import COMPONENTS

OUTPUT_DIR  = "/mnt/user-data/outputs/rxnet_v3"
HISTORY_DIR = "/mnt/user-data/outputs/rxnet_v3/historique"
ALERT_LOG   = "/mnt/user-data/outputs/rxnet_v3/alertes.json"

def make_det(cls, conf, area=0.07):
    return {"class": cls, "confidence": conf,
            "mask_area": area, "mask_bin": None}

def main():
    os.makedirs(OUTPUT_DIR,  exist_ok=True)
    os.makedirs(HISTORY_DIR, exist_ok=True)

    device_id = "Siemens_Ysio_Max_Salle_B"

    diag_engine = DiagnosticEngine(
        device_id=device_id, history_dir=HISTORY_DIR
    )
    pred_engine = PredictiveEngine(
        device_id=device_id,
        history_dir=HISTORY_DIR,
        alert_log=ALERT_LOG,
    )

    # ── Âges composants ───────────────────────────────────────────────────
    component_ages = {
        "anode_tournante":    4.2,
        "stator_roulements":  4.2,
        "filament":           4.2,
        "chambre_ionisation": 3.0,
        "regulateur_ht":      5.5,
        "matrice_tft":        2.8,
        "moteur_bucky":       6.1,
        "grille_pb":          6.1,
        "lames_collimateur":  8.0,
    }

    # ── 8 analyses simulées sur 3 mois (dégradation progressive) ─────────
    # On injecte des dates rétrospectives dans l'historique
    base_date  = datetime.datetime.now() - datetime.timedelta(days=90)

    scenarios = [
        # (jours depuis base, artefacts)
        ( 0,  [make_det("grid_lines",    0.52, 0.03)]),
        (12,  [make_det("grid_lines",    0.63, 0.05),
               make_det("anode_artifact",0.44, 0.02)]),
        (25,  [make_det("grid_lines",    0.74, 0.07),
               make_det("anode_artifact",0.61, 0.05)]),
        (38,  [make_det("anode_artifact",0.77, 0.09),
               make_det("focal_spot_defect",0.58, 0.04),
               make_det("overexposure",  0.50, 0.20)]),
        (52,  [make_det("anode_artifact",0.84, 0.12),
               make_det("focal_spot_defect",0.72, 0.08),
               make_det("overexposure",  0.65, 0.30)]),
        (65,  [make_det("anode_artifact",0.89, 0.15),
               make_det("focal_spot_defect",0.83, 0.11),
               make_det("overexposure",  0.77, 0.38),
               make_det("grid_lines",    0.80, 0.10)]),
        (78,  [make_det("anode_artifact",0.92, 0.18),
               make_det("focal_spot_defect",0.90, 0.14),
               make_det("overexposure",  0.85, 0.45),
               make_det("collimator_misalignment",0.61, 0.06)]),
        (90,  [make_det("anode_artifact",0.96, 0.22),
               make_det("focal_spot_defect",0.94, 0.17),
               make_det("overexposure",  0.91, 0.50),
               make_det("grid_lines",    0.88, 0.13),
               make_det("collimator_misalignment",0.74, 0.08)]),
    ]

    print("\n" + "═"*65)
    print("  RxArtifactNet v3.0 — Prédiction temporelle de panne")
    print(f"  Appareil : {device_id}")
    print("═"*65)

    for i, (delta_j, artefacts) in enumerate(scenarios, 1):
        date_ana = (base_date + datetime.timedelta(days=delta_j)).isoformat()

        diag = diag_engine.diagnose(
            detections=artefacts,
            component_ages=component_ages,
            save_history=False,   # On gère manuellement les dates
        )

        # Injecter l'entrée avec la bonne date dans l'historique
        entry = {
            "date":             date_ana,
            "device_id":        device_id,
            "n_artefacts":      len(artefacts),
            "score_global":     diag["score_global"],
            "component_scores": {
                cd["composant_id"]: cd["score_risque"]
                for cd in diag["component_diagnostics"]
            },
            "artefacts": [d["class"] for d in artefacts],
        }
        diag_engine.history.save(device_id, entry)

        score = diag["score_global"]
        hz    = diag["horizon_global"]["label"]
        date_str = (base_date + datetime.timedelta(days=delta_j)).strftime("%d/%m/%Y")
        bar = "█" * int(score / 5) + "░" * (20 - int(score / 5))
        print(f"  J+{delta_j:3d} ({date_str})  [{bar}]  {score:5.1f}  {hz}")

    # ── Dernière analyse = état actuel ────────────────────────────────────
    print("\n[Calcul des prédictions niveau 3...]")
    last_artefacts = scenarios[-1][1]
    last_diag = diag_engine.diagnose(
        detections=last_artefacts,
        component_ages=component_ages,
        save_history=False,
    )

    prediction = pred_engine.predict(
        current_diagnosis=last_diag,
        component_ages=component_ages,
        component_knowledge=COMPONENTS,
    )

    # ── Affichage terminal des prédictions ────────────────────────────────
    print("\n" + "─"*65)
    print("  PRÉDICTIONS TEMPORELLES")
    print("─"*65)
    for pred in prediction["predictions"]:
        a   = pred["alerte"]
        fw  = pred["fenetre_intervention"]
        jp  = fw.get("jours_restants", "?")
        print(f"\n  {a['emoji']} [{a['niveau']:8s}] {pred['composant_nom']}")
        print(f"     Score actuel  : {pred['score_actuel']:.1f}/100")
        print(f"     P(panne/30j)  : {pred['weibull'].get('prob_panne_30j',0):.0f}%  "
              f"(Weibull)")
        print(f"     Panne estimée : J+{jp}")
        print(f"     Intervenir    : {fw.get('debut','?')} → {fw.get('fin','?')}")
        print(f"     Confiance     : {pred['regression'].get('confiance_pct',0):.0f}%")

    # ── Génération des graphiques ─────────────────────────────────────────
    print("\n[Génération des graphiques...]")
    history_raw = diag_engine.history.load(device_id)

    fig_deg  = plot_degradation_curves(
        prediction["predictions"],
        os.path.join(OUTPUT_DIR, "degradation_curves.png")
    )
    fig_tl   = plot_timeline(
        prediction["predictions"],
        device_id,
        os.path.join(OUTPUT_DIR, "timeline_interventions.png")
    )
    fig_hist = plot_score_history(
        history_raw,
        os.path.join(OUTPUT_DIR, "score_history.png")
    )
    print(f"  Courbes Weibull → {fig_deg}")
    print(f"  Timeline        → {fig_tl}")
    print(f"  Historique      → {fig_hist}")

    # ── Rapport PDF niveau 3 ─────────────────────────────────────────────
    print("\n[Génération du rapport PDF niveau 3...]")
    report_path = os.path.join(OUTPUT_DIR, "rapport_predictif_niveau3.pdf")
    meta = {
        "device":  device_id,
        "date":    datetime.date.today().strftime("%d/%m/%Y"),
        "patient": "—",
    }
    PredictiveReport(report_path).build(
        prediction=prediction,
        meta=meta,
        fig_degradation=fig_deg,
        fig_timeline=fig_tl,
        fig_history=fig_hist,
    )
    print(f"  Rapport PDF → {report_path}")

    # ── Récapitulatif final ───────────────────────────────────────────────
    ag = prediction["alerte_globale"]
    print("\n" + "═"*65)
    print("  RÉCAPITULATIF FINAL")
    print("═"*65)
    print(f"  Appareil         : {device_id}")
    print(f"  Alerte globale   : {ag['emoji']} {ag['niveau']}")
    print(f"  Action requise   : {ag['delai_intervention']}")
    top = prediction["predictions"][0] if prediction["predictions"] else None
    if top:
        fw = top["fenetre_intervention"]
        print(f"  Composant n°1   : {top['composant_nom']}")
        print(f"  Fenêtre optimale: {fw.get('debut','?')} → {fw.get('fin','?')}")
        print(f"  Date panne est. : J+{fw.get('jours_restants','?')}")
    print(f"\n  Alertes actives  : {prediction['n_alertes_actives']}")
    print(f"  Rapport PDF      : {report_path}")
    print("═"*65 + "\n")

if __name__ == "__main__":
    main()
