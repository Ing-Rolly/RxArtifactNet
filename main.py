"""
main.py
=======
Démonstration complète de RxArtifactNet.
Lance une analyse sur plusieurs artefacts simulés et génère les rapports.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline import RxAnalysisPipeline

OUTPUT_DIR = "/mnt/user-data/outputs/rxnet"

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    pipeline = RxAnalysisPipeline(
        output_dir=OUTPUT_DIR,
        seg_threshold=0.30,
        cls_threshold=0.12,
    )

    pipeline.model.summary()

    # ── Cas 1 : Corps étranger (bijou / électrode ECG) ─────────────────
    print("\n>>> CAS 1 : Corps étranger")
    r1 = pipeline.analyze(
        artifact_simulation="foreign_object",
        patient_id="PAT_001",
        device_name="Salle_RX_A — GE Discovery XR656",
        operator="M. Dupont",
    )
    pipeline.print_summary(r1)

    # ── Cas 2 : Lignes de grille (Bucky bloqué) ────────────────────────
    print("\n>>> CAS 2 : Lignes de grille anti-diffusante")
    r2 = pipeline.analyze(
        artifact_simulation="grid_lines",
        patient_id="PAT_002",
        device_name="Salle_RX_B — Siemens Ysio Max",
        operator="Mme Martin",
    )
    pipeline.print_summary(r2)

    # ── Cas 3 : Flou cinétique (mouvement) ────────────────────────────
    print("\n>>> CAS 3 : Flou cinétique")
    r3 = pipeline.analyze(
        artifact_simulation="motion_blur",
        patient_id="PAT_003",
        device_name="Salle_RX_A — GE Discovery XR656",
        operator="M. Dupont",
    )
    pipeline.print_summary(r3)

    # ── Cas 4 : Pixels morts capteur DR ───────────────────────────────
    print("\n>>> CAS 4 : Pixels morts (capteur DR)")
    r4 = pipeline.analyze(
        artifact_simulation="dead_pixels",
        patient_id="PAT_004",
        device_name="Salle_RX_C — Canon CXDI-810C",
        operator="Automatique",
    )
    pipeline.print_summary(r4)

    # ── Cas 5 : Surexposition ─────────────────────────────────────────
    print("\n>>> CAS 5 : Surexposition (CAE défaillant)")
    r5 = pipeline.analyze(
        artifact_simulation="overexposure",
        patient_id="PAT_005",
        device_name="Salle_RX_B — Siemens Ysio Max",
        operator="Mme Martin",
    )
    pipeline.print_summary(r5)

    print("\n" + "="*60)
    print("  FICHIERS GÉNÉRÉS")
    print("="*60)
    for f in sorted(os.listdir(OUTPUT_DIR)):
        fpath = os.path.join(OUTPUT_DIR, f)
        size  = os.path.getsize(fpath) / 1024
        print(f"  {f:<55} {size:>7.1f} KB")
    print("="*60)

if __name__ == "__main__":
    main()
