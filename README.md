# RxArtifactNet

Automated diagnosis of technical artifacts on chest/bone X-ray images, based on metrological analysis of image artefacts. Developed as part of a Master's thesis in Metrology and Biomedical Engineering (University of Dschang, 2024–2026).

## What it does

Radiography equipment (X-ray generators, detectors, grids, CR/DR sensors) can produce **technical artifacts** on images — visible defects caused by a hardware or calibration issue rather than by the patient's anatomy. RxArtifactNet analyzes an X-ray image, detects and classifies these artifacts, and links each one to:

- the likely faulty **component** (generator, collimator, Bucky grid, DR/CR sensor, PACS/console...)
- the physical **zone** to inspect
- an **urgency level** for maintenance
- **suggested maintenance actions**

It is aimed at radiology technicians and biomedical maintenance staff, to speed up first-level troubleshooting before calling a specialized technician.

## Detected artifact classes (12)

`overexposure`, `underexposure`, `focal_spot_defect`, `anode_artifact`, `collimator_misalignment`, `grid_lines`, `grid_cutoff`, `dead_pixels`, `ghosting_dr`, `cr_scratch`, `cr_fog`, `processing_artifact`

## Project structure

Main deliverable: a desktop GUI application (`app.py`) intended for hospital/biomedical maintenance staff, with a future deployment target on Raspberry Pi (embedded, low-cost diagnostic station).

| File | Role |
|---|---|
| `app.py` | **Main application** — desktop GUI (Tkinter) so non-technical hospital staff can run a diagnosis without writing code. Deployment target: Raspberry Pi |
| `preprocessing.py` | Image loading (supports DICOM via `pydicom`, PNG/JPEG via OpenCV/Pillow), normalization, resizing, CLAHE contrast enhancement, and synthetic artifact simulation for training data |
| `knowledge_base.py` | Domain knowledge base: artifact → component → zone → urgency → maintenance actions |
| `component_knowledge.py` | Additional structured knowledge about hardware components |
| `model.py` / `model_torch.py` | Neural network architecture (PyTorch) for artifact classification |
| `train_simple.py` | Model training script |
| `pipeline.py` | End-to-end analysis pipeline: preprocessing → model inference → diagnostic reasoning |
| `diagnostic_engine.py` | Core diagnostic logic combining model predictions with the knowledge base |
| `predictive_engine.py` | Predictive/trend logic (e.g. recurring faults, maintenance forecasting) |
| `report_generator.py`, `report_niveau2.py`, `report_niveau3.py` | Generate diagnostic reports (PDF via ReportLab) at increasing levels of detail |
| `visualizer.py` | Visualization utilities (matplotlib) for inspecting predictions |
| `dev_scripts/` | Development/validation scripts used while building each module (basic pipeline, diagnostic report, predictive report) — not part of the final application |

## Roadmap

- [ ] Package and optimize the pipeline for deployment on Raspberry Pi (embedded, low-cost diagnostic station for radiology departments)
- [ ] Replace synthetic artifact simulation with a real annotated dataset
- [ ] Add automated tests

## Tech stack

Python · NumPy · OpenCV · Pillow · pydicom · PyTorch / torchvision · SciPy · Matplotlib · ReportLab (PDF reports) · Tkinter (desktop UI)

## Installation

```bash
git clone https://github.com/Ing-Rolly/RxArtifactNet.git
cd RxArtifactNet
pip install -r requirements.txt
```

## Usage

Launch the desktop application (main deliverable):

```bash
python app.py
```

Development/validation scripts (used to test each module independently while building the project) are available in `dev_scripts/`:

```bash
python dev_scripts/01_basic_pipeline.py
python dev_scripts/02_diagnostic_report.py
python dev_scripts/03_predictive_report.py
```

> Output directories (`outputs/`, `historique/`) are created automatically at runtime, relative to the project folder — no absolute paths required.

## Status & limitations

This is a research/thesis project, not a certified medical device. Artifact simulation is currently based on synthetic degradation of clean images rather than a large real-world annotated dataset. The authentication feature in the desktop app uses a local demo account and is not intended for production use.

## Author

TAMEZA Rolly Midèle — Biomedical Engineer, University of Dschang
[GitHub](https://github.com/ing-Rolly) · [LinkedIn](https://linkedin.com/in/rolly-midele-tameza)
