"""
pipeline.py
===========
Pipeline d'inférence — VERSION CORRIGÉE.
"""

import os, sys, time, datetime
import numpy as np
from typing import Dict, Optional, List

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base      import CLASS_NAMES, NUM_CLASSES, get_artifact_info, URGENCE_ORDRE
from model               import RxArtifactNet
from preprocessing       import RxPreprocessor
from visualizer          import create_analysis_figure, export_overlay_image
from report_generator    import RxMaintenanceReport

assert NUM_CLASSES == 12, f"Attendu 12 classes machine, trouvé {NUM_CLASSES}"


class RxAnalysisPipeline:
    def __init__(self,
                 weights_path:  Optional[str] = None,
                 output_dir:    str  = "./outputs",
                 seg_threshold: float = 0.30,
                 cls_threshold: float = 0.12):

        self.output_dir    = output_dir
        self.seg_threshold = seg_threshold
        self.cls_threshold = cls_threshold
        os.makedirs(output_dir, exist_ok=True)

        print("[Pipeline] Initialisation RxArtifactNet (12 artefacts machine)...")
        self.model        = RxArtifactNet(seed=42)
        self.preprocessor = RxPreprocessor(
            target_size=(512, 512),
            clahe_clip=2.0,
            augment=False
        )

        if weights_path and os.path.exists(weights_path):
            self.model.load_weights(weights_path)
        else:
            print("[Pipeline] Mode démo — poids aléatoires + simulation intégrée")

        print(f"[Pipeline] Prêt — {NUM_CLASSES} classes machine actives")

    def analyze(self,
                image_path:          Optional[str]   = None,
                image_array:         Optional[np.ndarray] = None,
                patient_id:          str  = "PATIENT_001",
                device_name:         str  = "Appareil_RX_01",
                operator:            str  = "Automatique",
                artifact_simulation: Optional[str]   = None,
                generate_report:     bool = True) -> Dict:
                
        if artifact_simulation and artifact_simulation not in CLASS_NAMES:
            raise ValueError(f"Artefact inconnu : '{artifact_simulation}'")

        ts    = time.time()
        date  = datetime.date.today().strftime("%Y%m%d_%H%M%S")
        prefix = f"{patient_id}_{date}"

        # 1. Prétraitement
        if image_path is not None:
            tensor, img_display = self.preprocessor.process_file(image_path, artifact_simulation)
        elif image_array is not None:
            tensor, img_display = self.preprocessor.process_array(image_array, artifact_simulation)
        else:
            if artifact_simulation is None:
                rng = np.random.default_rng(int(time.time()) % 1000)
                artifact_simulation = rng.choice(CLASS_NAMES)
            tensor, img_display = self.preprocessor.create_synthetic_rx(artifact=artifact_simulation, seed=42)

        # 2. Inférence (Le bug n'aura plus lieu ici grâce à la correction de model.py)
        result = self.model.predict(
            tensor,
            seg_threshold=self.seg_threshold,
            cls_threshold=self.cls_threshold,
        )

        # 3. Post-traitement
        detections = self._post_process(result, artifact_simulation)

        # 4. Visualisation
        fig_path     = os.path.join(self.output_dir, f"{prefix}_analyse.png")
        overlay_path = os.path.join(self.output_dir, f"{prefix}_overlay.png")

        fig_paths = create_analysis_figure(
            image_display=img_display,
            detections=detections,
            full_seg_map=result["full_seg_map"],
            all_probs=result["all_probs"],
            class_names=CLASS_NAMES,
            output_path=fig_path,
            patient_id=patient_id,
        )
        fig_path = fig_paths[0] if fig_paths else fig_path
        export_overlay_image(img_display, detections, overlay_path)

        # 5. Rapport PDF
        report_path = None
        if generate_report:
            report_path = os.path.join(self.output_dir, f"{prefix}_rapport.pdf")
            RxMaintenanceReport(report_path).build(
                detections=detections,
                meta={
                    "patient_id": patient_id,
                    "device":     device_name,
                    "date":       datetime.date.today().strftime("%d/%m/%Y"),
                    "operator":   operator,
                },
                figure_path=fig_path,
                overlay_path=overlay_path,
                analysis_time=time.time() - ts,
            )

        return {
            "detections":          detections,
            "all_probs":           result["all_probs"],
            "full_seg_map":        result["full_seg_map"],
            "figure_path":         fig_path,
            "figure_paths":        fig_paths,
            "overlay_path":        overlay_path,
            "report_path":         report_path,
            "analysis_time":       time.time() - ts,
            "artifact_simulated":  artifact_simulation,
        }

    def _post_process(self, result: Dict, artifact_simulation: Optional[str]) -> List[Dict]:
        detections = result["detections"]
        if artifact_simulation and len(detections) < 2:
            sim_det = self._make_demo_detection(artifact_simulation, result["full_seg_map"].shape)
            if sim_det:
                existing = [d["class"] for d in detections]
                if artifact_simulation not in existing:
                    detections.insert(0, sim_det)
        return [d for d in detections if d["class"] in CLASS_NAMES][:5]

    def _make_demo_detection(self, artifact_class: str, seg_shape: tuple) -> Optional[Dict]:
        if artifact_class not in CLASS_NAMES: return None
        _, H, W = seg_shape
        rng  = np.random.default_rng(42)
        mask = np.zeros((H, W), dtype=np.float32)
        import cv2

        if artifact_class == "overexposure":
            mask = np.ones((H, W), dtype=np.float32) * 0.75
            mask = cv2.GaussianBlur(mask, (31,31), 10)
        elif artifact_class == "underexposure":
            mask = np.ones((H, W), dtype=np.float32) * 0.70
            mask = cv2.GaussianBlur(mask, (31,31), 10)
        elif artifact_class == "focal_spot_defect":
            mask = rng.uniform(0.3, 0.7, (H, W)).astype(np.float32)
            mask = cv2.GaussianBlur(mask, (61,61), 25)
        elif artifact_class == "anode_artifact":
            freq = rng.integers(10, 20)
            for y in range(0, H, freq): mask[y:y+3, :] = rng.uniform(0.6, 0.9)
        elif artifact_class == "collimator_misalignment":
            width = rng.integers(40, 100)
            mask[:, :width] = 0.85
        elif artifact_class == "grid_lines":
            freq = rng.integers(16, 24)
            for x in range(0, W, freq): mask[:, x:x+3] = rng.uniform(0.6, 0.9)
        elif artifact_class == "grid_cutoff":
            for x in range(W):
                dist = min(x, W-1-x)
                mask[:, x] = max(0, 0.9 - dist / (W * 0.3))
        elif artifact_class == "dead_pixels":
            ys = rng.integers(0, H, 120)
            xs = rng.integers(0, W, 120)
            mask[ys, xs] = 1.0
            mask = cv2.dilate(mask, np.ones((3,3)), iterations=2)
        elif artifact_class == "ghosting_dr":
            ghost = rng.uniform(0.2, 0.6, (H, W)).astype(np.float32)
            mask  = cv2.GaussianBlur(ghost, (41,41), 15)
        elif artifact_class == "cr_scratch":
            n = rng.integers(3, 8)
            for _ in range(n):
                x = rng.integers(20, W-20)
                mask[:, x:x+2] = rng.uniform(0.7, 0.95)
        elif artifact_class == "cr_fog":
            mask = np.ones((H, W), dtype=np.float32) * rng.uniform(0.55, 0.75)
            mask = cv2.GaussianBlur(mask, (21,21), 8)
        elif artifact_class == "processing_artifact":
            bs = 32
            for i in range(0, H, bs):
                for j in range(0, W, bs): mask[i:i+bs, j:j+bs] = rng.uniform(0.3, 0.8)

        mask     = np.clip(mask, 0, 1)
        mask_bin = (mask > 0.4).astype(np.uint8)
        return {
            "class":      artifact_class,
            "confidence": float(rng.uniform(0.72, 0.95)),
            "mask":       mask,
            "mask_bin":   mask_bin,
            "mask_area":  float(mask_bin.mean()),
        }