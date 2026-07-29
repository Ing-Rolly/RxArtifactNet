"""
diagnostic_engine.py
====================
Moteur de diagnostic niveau 2 — VERSION CORRIGÉE.
Filtre strict : seuls les 12 artefacts machine sont traités.
Aucune référence aux artefacts patient.
"""

import json, os, datetime
import numpy as np
from typing import Dict, List, Optional

from knowledge_base import CLASS_NAMES, NUM_CLASSES, get_artifact_info, URGENCE_ORDRE
from component_knowledge import (
    COMPONENTS, ARTIFACT_TO_COMPONENTS,
    RISK_HORIZONS, get_risk_horizon
)

assert NUM_CLASSES == 12


# ══════════════════════════════════════════════════════════════════════════
# GESTIONNAIRE D'HISTORIQUE
# ══════════════════════════════════════════════════════════════════════════

class HistoryManager:
    """Persiste l'historique des analyses par appareil (JSON)."""

    def __init__(self, history_dir="./historique"):
        self.history_dir = history_dir
        os.makedirs(history_dir, exist_ok=True)

    def _path(self, device_id):
        safe = device_id.replace(" ","_").replace("/","-")
        return os.path.join(self.history_dir, f"{safe}.json")

    def load(self, device_id) -> List[dict]:
        path = self._path(device_id)
        if not os.path.exists(path):
            return []
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)

    def save(self, device_id, entry):
        history = self.load(device_id)
        history.append(entry)
        history = history[-90:]   # 90 analyses max
        with open(self._path(device_id), "w", encoding="utf-8") as f:
            json.dump(history, f, ensure_ascii=False, indent=2)

    def get_component_trend(self, device_id, component_id,
                             n_last=10) -> dict:
        """Tendance d'un composant sur les N dernières analyses."""
        history = self.load(device_id)
        if len(history) < 2:
            return {"pente":0.0, "tendance":"insuffisant",
                    "n_apparitions":0, "score_moyen":0.0}

        recent = history[-n_last:]
        scores = []
        n_app  = 0
        for e in recent:
            cs = e.get("component_scores", {})
            if component_id in cs:
                scores.append(cs[component_id]); n_app += 1
            else:
                scores.append(0.0)

        if len(scores) < 2:
            return {"pente":0.0, "tendance":"stable",
                    "n_apparitions":n_app, "score_moyen":0.0}

        x     = np.arange(len(scores), dtype=float)
        pente = float(np.polyfit(x, scores, 1)[0])

        if pente > 2.0:   tendance = "aggravation_rapide"
        elif pente > 0.5: tendance = "aggravation"
        elif pente < -2.0:tendance = "amelioration_rapide"
        elif pente < -0.5:tendance = "amelioration"
        else:             tendance = "stable"

        moy = float(np.mean([s for s in scores if s > 0])) if n_app > 0 else 0.0
        return {"pente":pente, "tendance":tendance,
                "n_apparitions":n_app, "score_moyen":moy, "scores":scores}


# ══════════════════════════════════════════════════════════════════════════
# CALCUL DU SCORE DE RISQUE
# ══════════════════════════════════════════════════════════════════════════

class RiskScorer:

    def compute_base_score(self, artifact_class, confidence,
                            mask_area) -> Dict[str, float]:
        """Score de base par composant pour un artefact machine donné."""
        # Seuls les artefacts machine sont acceptés
        if artifact_class not in CLASS_NAMES:
            return {}
        if artifact_class not in ARTIFACT_TO_COMPONENTS:
            return {}

        signal = confidence * (1 + min(mask_area * 10, 1.0)) / 2
        scores = {}
        for comp_id, prob in ARTIFACT_TO_COMPONENTS[artifact_class]:
            crit = COMPONENTS[comp_id]["criticite"] / 100.0
            scores[comp_id] = round(min(signal * prob * crit * 100, 100), 2)
        return scores

    def apply_trend_boost(self, scores, trend_data) -> Dict[str, float]:
        boosted = {}
        for comp_id, score in scores.items():
            pente = trend_data.get(comp_id, {}).get("pente", 0.0)
            n_app = trend_data.get(comp_id, {}).get("n_apparitions", 0)
            if pente > 2.0:   boost = 1.35
            elif pente > 0.5: boost = 1.20
            elif pente > 0.0: boost = 1.08
            else:             boost = 1.0
            freq_boost = 1.0 + min(n_app / 20.0, 0.25)
            boosted[comp_id] = round(min(score * boost * freq_boost, 100), 2)
        return boosted

    def apply_age_factor(self, scores,
                          component_ages) -> Dict[str, float]:
        aged = {}
        for comp_id, score in scores.items():
            dvie = COMPONENTS[comp_id].get("duree_vie_ans")
            age  = component_ages.get(comp_id, 0)
            if dvie and dvie > 0 and age > 0:
                ratio     = age / dvie
                age_boost = 1.0 + max(0, ratio - 0.6) * 1.5
            else:
                age_boost = 1.0
            aged[comp_id] = round(min(score * age_boost, 100), 2)
        return aged

    def aggregate_multi_artifact(self,
                                  all_scores: List[Dict]) -> Dict[str, float]:
        agg = {}
        for scores in all_scores:
            for comp_id, score in scores.items():
                if comp_id in agg:
                    agg[comp_id] = round(
                        min(agg[comp_id] + score*(1 - agg[comp_id]/100), 100), 2
                    )
                else:
                    agg[comp_id] = score
        return agg


# ══════════════════════════════════════════════════════════════════════════
# MOTEUR DE DIAGNOSTIC PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

class DiagnosticEngine:
    """
    Orchestre le diagnostic niveau 2 pour les 12 artefacts machine.

    Usage :
        engine = DiagnosticEngine(device_id="GE_XR656_Salle_A")
        diag   = engine.diagnose(detections, component_ages)
    """

    def __init__(self, device_id="APPAREIL_01", history_dir="./historique"):
        self.device_id = device_id
        self.history   = HistoryManager(history_dir)
        self.scorer    = RiskScorer()

    def diagnose(self, detections: List[dict],
                 component_ages: Dict[str, float] = None,
                 save_history: bool = True) -> dict:
        """
        Diagnostic complet niveau 2.

        Parameters
        ----------
        detections      : sortie du pipeline niveau 1
                          (uniquement artefacts machine attendus)
        component_ages  : âge de chaque composant en années
        save_history    : enregistrer dans l'historique JSON
        """
        component_ages = component_ages or {}

        # ── Filtrer strictement les artefacts machine ─────────────────────
        machine_dets = [
            d for d in detections
            if d["class"] in CLASS_NAMES          # 12 classes machine
            and d["class"] in ARTIFACT_TO_COMPONENTS  # mapping composant existe
        ]

        if not machine_dets:
            return self._empty_diagnosis()

        # ── Scores de base ────────────────────────────────────────────────
        all_base = []
        for det in machine_dets:
            base = self.scorer.compute_base_score(
                det["class"], det["confidence"],
                det.get("mask_area", 0.05)
            )
            all_base.append(base)

        # ── Agrégation ────────────────────────────────────────────────────
        aggregated = self.scorer.aggregate_multi_artifact(all_base)

        # ── Tendances historiques ─────────────────────────────────────────
        trend_data = {
            comp_id: self.history.get_component_trend(self.device_id, comp_id)
            for comp_id in aggregated
        }

        # ── Boost tendance + âge ──────────────────────────────────────────
        with_trend   = self.scorer.apply_trend_boost(aggregated, trend_data)
        final_scores = self.scorer.apply_age_factor(with_trend, component_ages)

        # ── Classement par score décroissant ──────────────────────────────
        ranked = sorted(final_scores.items(), key=lambda x: -x[1])

        # ── Construire les diagnostics composant ──────────────────────────
        comp_diags = []
        for comp_id, score in ranked:
            comp    = COMPONENTS[comp_id]
            horizon = get_risk_horizon(score)
            trend   = trend_data.get(comp_id, {})
            age     = component_ages.get(comp_id, 0)
            dvie    = comp.get("duree_vie_ans")

            prob_panne = self._estimate_failure_prob(score, trend, age, dvie)

            comp_diags.append({
                "composant_id":      comp_id,
                "composant_nom":     comp["nom"],
                "sous_systeme":      comp["sous_systeme"],
                "zone":              comp["zone"],
                "score_risque":      score,
                "horizon":           horizon,
                "tendance":          trend.get("tendance", "insuffisant"),
                "pente":             trend.get("pente", 0.0),
                "n_apparitions":     trend.get("n_apparitions", 0),
                "score_moyen_hist":  trend.get("score_moyen", 0.0),
                "age_ans":           age,
                "duree_vie_ans":     dvie,
                "prob_panne_30j":    prob_panne["j30"],
                "prob_panne_90j":    prob_panne["j90"],
                "cout_remplacement": comp["cout_remplacement"],
                "actions":           self._get_actions(comp_id, score, trend),
            })

        score_global   = self._global_score(comp_diags)
        horizon_global = get_risk_horizon(score_global)

        # ── Sauvegarder dans l'historique ─────────────────────────────────
        if save_history:
            self.history.save(self.device_id, {
                "date":             datetime.datetime.now().isoformat(),
                "device_id":        self.device_id,
                "n_artefacts":      len(machine_dets),
                "score_global":     score_global,
                "component_scores": {c: s for c, s in final_scores.items()},
                "artefacts":        [d["class"] for d in machine_dets],
            })

        return {
            "device_id":             self.device_id,
            "date_analyse":          datetime.datetime.now().isoformat(),
            "n_artefacts_machine":   len(machine_dets),
            "artefacts_detectes":    [d["class"] for d in machine_dets],
            "component_diagnostics": comp_diags,
            "score_global":          score_global,
            "horizon_global":        horizon_global,
            "composant_prioritaire": comp_diags[0] if comp_diags else None,
            "resume":                self._resume(comp_diags, horizon_global),
        }

    # ── HELPERS INTERNES ───────────────────────────────────────────────────

    def _estimate_failure_prob(self, score, trend, age, dvie):
        base_j30 = score / 100 * 0.6
        base_j90 = score / 100 * 0.85
        pente    = trend.get("pente", 0.0)
        tf = 1.4 if pente > 2.0 else (1.2 if pente > 0.5 else 1.0)
        if dvie and dvie > 0 and age > 0:
            af = 1.0 + max(0, age/dvie - 0.7) * 2.0
        else:
            af = 1.0
        return {
            "j30": round(min(base_j30 * tf * af, 0.99) * 100, 1),
            "j90": round(min(base_j90 * tf * af, 0.99) * 100, 1),
        }

    def _get_actions(self, comp_id, score, trend):
        """Actions correctives adaptées au score et à la tendance."""
        comp     = COMPONENTS[comp_id]
        sous_sys = comp["sous_systeme"]
        tendance = trend.get("tendance", "stable")

        actions_map = {
            "Générateur HT": [
                "Mesurer kV/mAs avec dosimètre (reproductibilité < 5 %)",
                "Tester la chambre d'ionisation avec fantôme PMMA",
                "Vérifier la régulation de tension en charge",
                "Inspecter les condensateurs HT",
            ],
            "Contrôle automatique exposition": [
                "Tester l'AEC avec fantôme PMMA et dosimètre",
                "Nettoyer et repositionner la chambre d'ionisation",
                "Vérifier le câblage de la chambre",
                "Recalibrer le seuil de déclenchement AEC",
            ],
            "Tube radiogène": [
                "Mesurer le foyer avec l'outil stellaire",
                "Écouter le bruit de rotation de l'anode",
                "Test de montée en vitesse à vide",
                "Inspecter l'ampoule (noircissement = remplacement immédiat)",
                "Vérifier la courbe MTF / résolution spatiale",
            ],
            "Collimateur / diaphragme": [
                "Tester la coïncidence champ lumineux / champ RX",
                "Inspecter les lames Pb (coincement, usure)",
                "Vérifier le moteur d'entraînement des lames",
                "Contrôler la lampe et le miroir de centrage",
            ],
            "Grille anti-diffusante": [
                "Tester l'oscillation du Bucky (écoute moteur)",
                "Vérifier le centrage et la verticalité de la grille",
                "Inspecter les lamelles Pb (impact, déformation)",
                "Contrôler la distance focale grille / source",
            ],
            "Capteur DR Flat Panel": [
                "Cartographier les pixels défectueux (outil constructeur)",
                "Effectuer une calibration flat-field",
                "Mesurer la DQE (Detective Quantum Efficiency)",
                "Vérifier la connexion électronique de lecture",
            ],
            "Lecteur CR": [
                "Nettoyer la plaque IP avec chiffon non pelucheux",
                "Mesurer la puissance du laser de lecture",
                "Vérifier la lampe UV d'effacement",
                "Inspecter le guide de transport",
            ],
            "Console d'acquisition": [
                "Recalibrer les LUT et présets de traitement",
                "Vérifier le taux de compression DICOM (lossless)",
                "Mettre à jour le logiciel d'acquisition",
                "Contrôler les paramètres de fenêtrage automatique",
            ],
            "Système d'information radiologique": [
                "Vérifier les paramètres de compression PACS",
                "Contrôler la bande passante réseau",
                "Mettre à jour le serveur PACS",
            ],
        }

        actions = actions_map.get(sous_sys, ["Contacter le service technique"])

        if score >= 85:
            actions = ["🔴 ARRÊT RECOMMANDÉ — Vérifier immédiatement"] + actions[:3]
        elif score >= 65:
            actions = ["🟠 Planifier intervention dans 48–72 h"] + actions[:3]
        elif tendance in ("aggravation_rapide", "aggravation"):
            actions = ["🟡 Tendance aggravante — renforcer la surveillance"] + actions[:2]

        return actions

    def _global_score(self, diagnostics):
        if not diagnostics:
            return 0.0
        scores = sorted([d["score_risque"] for d in diagnostics], reverse=True)
        if len(scores) == 1:
            return round(scores[0], 1)
        elif len(scores) == 2:
            return round(scores[0]*0.7 + scores[1]*0.3, 1)
        rest = np.mean(scores[2:])
        return round(scores[0]*0.60 + scores[1]*0.30 + rest*0.10, 1)

    def _resume(self, diagnostics, horizon_global):
        if not diagnostics:
            return "Aucun composant à risque identifié."
        top = diagnostics[0]
        return (
            f"Composant prioritaire : {top['composant_nom']} "
            f"({top['sous_systeme']}) — Score {top['score_risque']:.0f}/100. "
            f"Horizon : {horizon_global['label']}. "
            f"P(panne/30j) : {top['prob_panne_30j']:.0f}%."
        )

    def _empty_diagnosis(self):
        return {
            "device_id":             self.device_id,
            "date_analyse":          datetime.datetime.now().isoformat(),
            "n_artefacts_machine":   0,
            "artefacts_detectes":    [],
            "component_diagnostics": [],
            "score_global":          0.0,
            "horizon_global":        get_risk_horizon(0),
            "composant_prioritaire": None,
            "resume":                "Aucun artefact machine détecté — appareil conforme.",
        }

    def get_device_history_summary(self):
        """Résumé de l'historique de l'appareil."""
        history = self.history.load(self.device_id)
        if not history:
            return {"n_analyses": 0,
                    "message": "Aucun historique disponible"}

        scores = [e["score_global"] for e in history]
        dates  = [e["date"][:10]    for e in history]

        art_counts = {}
        for e in history:
            for a in e.get("artefacts", []):
                if a in CLASS_NAMES:   # uniquement artefacts machine
                    art_counts[a] = art_counts.get(a, 0) + 1

        top_arts = sorted(art_counts.items(), key=lambda x: -x[1])[:5]

        pente_glob = float(np.polyfit(range(len(scores)), scores,
                                       min(1, len(scores)-1))[0]) \
                     if len(scores) >= 2 else 0.0

        if pente_glob > 3:    tg = "aggravation_rapide"
        elif pente_glob > 0.5:tg = "aggravation"
        elif pente_glob < -3: tg = "amelioration_rapide"
        elif pente_glob < -0.5:tg= "amelioration"
        else:                 tg = "stable"

        return {
            "device_id":       self.device_id,
            "n_analyses":      len(history),
            "premiere_date":   dates[0]  if dates  else "—",
            "derniere_date":   dates[-1] if dates  else "—",
            "score_moyen":     round(float(np.mean(scores)), 1),
            "score_max":       round(float(np.max(scores)),  1),
            "score_actuel":    round(scores[-1], 1) if scores else 0.0,
            "top_artefacts":   top_arts,
            "tendance_globale":tg,
        }
