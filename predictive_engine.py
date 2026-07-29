"""
predictive_engine.py
====================
Niveau 3 — Prédiction temporelle de panne.

Modèles utilisés :
  1. Régression polynomiale sur l'historique des scores
     → Prédit la date probable de franchissement du seuil critique
  2. Modèle de Weibull (fiabilité industrielle)
     → Probabilité de survie du composant dans le temps
  3. Score de confiance de la prédiction
     → Basé sur la quantité et la régularité des données historiques

Sorties :
  - Date estimée de panne (avec intervalle de confiance)
  - Courbe de dégradation projetée
  - Fenêtre d'intervention optimale (avant la panne)
  - Niveau d'alerte : VERT / JAUNE / ORANGE / ROUGE / CRITIQUE
"""

import numpy as np
import json
import os
import datetime
from typing import Dict, List, Optional, Tuple
from scipy import stats
from scipy.special import gamma as gamma_func


# ══════════════════════════════════════════════════════════════════════════
# MODÈLE DE WEIBULL (fiabilité industrielle)
# ══════════════════════════════════════════════════════════════════════════

class WeibullModel:
    """
    Modèle de Weibull pour l'estimation de la durée de vie résiduelle.

    La distribution de Weibull est le standard industriel pour modéliser
    la défaillance des équipements médicaux.

    f(t) = (β/η) × (t/η)^(β-1) × exp(-(t/η)^β)

    β (shape)  : < 1 → défauts précoces
                   1 → défaillances aléatoires
                 > 1 → usure progressive (cas des composants Rx)
    η (scale)  : durée de vie caractéristique (63.2% des unités tombées)
    """

    def __init__(self, beta: float = 2.5, eta_hours: float = 30000):
        self.beta = beta        # Paramètre de forme (usure progressive)
        self.eta  = eta_hours   # Paramètre d'échelle (heures)

    @classmethod
    def from_component(cls, component_id: str,
                       component_knowledge: dict) -> "WeibullModel":
        """Instancie un modèle Weibull adapté au composant."""
        comp = component_knowledge.get(component_id, {})
        mtbf = comp.get("mtbf_heures", 30000) or 30000
        crit = comp.get("criticite", 70) / 100.0

        # β plus élevé pour composants mécaniques (usure rapide)
        sous_sys = comp.get("sous_systeme", "")
        if "Tube" in sous_sys or "radiogène" in sous_sys:
            beta = 3.2   # Usure progressive forte
        elif "Bucky" in sous_sys or "Grille" in sous_sys:
            beta = 2.8
        elif "Capteur" in sous_sys or "DR" in sous_sys:
            beta = 2.2
        elif "CR" in sous_sys or "Lecteur" in sous_sys:
            beta = 2.0
        else:
            beta = 2.5

        # η tel que MTBF = η × Γ(1 + 1/β)
        eta = mtbf / gamma_func(1 + 1 / beta)

        return cls(beta=beta, eta_hours=eta)

    def survival(self, t_hours: float) -> float:
        """P(survie jusqu'à t) = exp(-(t/η)^β)"""
        return float(np.exp(-((t_hours / self.eta) ** self.beta)))

    def failure_prob(self, t_hours: float) -> float:
        """P(panne avant t) = 1 - survie(t)"""
        return 1.0 - self.survival(t_hours)

    def hazard_rate(self, t_hours: float) -> float:
        """Taux de défaillance instantané h(t) = (β/η)(t/η)^(β-1)"""
        return float((self.beta / self.eta) *
                     ((t_hours / self.eta) ** (self.beta - 1)))

    def percentile_life(self, p: float) -> float:
        """Durée à laquelle p% des composants ont défailli (en heures)."""
        return float(self.eta * (-np.log(1 - p)) ** (1 / self.beta))

    def remaining_life(self, age_hours: float,
                       confidence: float = 0.10) -> dict:
        """
        Durée de vie résiduelle à partir de l'âge actuel.
        confidence = risque acceptable (0.10 = 10% de risque de panne).
        """
        t_total = self.percentile_life(1 - confidence)
        remaining = max(0.0, t_total - age_hours)
        survival_now = self.survival(age_hours)
        return {
            "remaining_hours":    remaining,
            "remaining_days":     remaining / 24,
            "survival_now":       survival_now * 100,
            "t_total_hours":      t_total,
            "confidence_level":   confidence,
        }

    def project_curve(self, age_hours: float,
                      horizon_days: int = 365) -> List[dict]:
        """Projette la courbe de dégradation sur horizon_days jours."""
        points = []
        for d in range(0, horizon_days + 1, 7):   # par semaine
            t = age_hours + d * 24
            points.append({
                "jour":          d,
                "date":          (datetime.date.today() +
                                  datetime.timedelta(days=d)).isoformat(),
                "prob_panne":    round(self.failure_prob(t) * 100, 2),
                "survie":        round(self.survival(t) * 100, 2),
                "taux_defail":   round(self.hazard_rate(t) * 1000, 4),
            })
        return points


# ══════════════════════════════════════════════════════════════════════════
# RÉGRESSION SUR L'HISTORIQUE DES SCORES
# ══════════════════════════════════════════════════════════════════════════

class ScoreRegressor:
    """
    Régression polynomiale sur la série temporelle des scores de risque.
    Prédit la date de franchissement du seuil critique (score = 85).
    """

    SEUIL_CRITIQUE  = 85.0
    SEUIL_URGENT    = 65.0
    SEUIL_PLANIFIE  = 45.0

    def fit(self, dates: List[str], scores: List[float]) -> dict:
        if len(scores) < 2:
            return self._insufficient()

        # Convertir les dates en jours relatifs
        t0 = self._parse_date(dates[0])
        x_days = np.array([
            (self._parse_date(d) - t0).days for d in dates
        ], dtype=float)
        y = np.array(scores, dtype=float)

        # Vérifier qu'il y a au moins deux dates distinctes
        if len(np.unique(x_days)) < 2:
            return {
                "status": "insufficient_data",
                "n_points": len(scores),
                "message": "Pas assez de variation dans les dates pour la régression",
                "score_actuel": float(y[-1]),
                "confiance_pct": 0,
                "r2": 0,
                "pente_actuelle": 0,
                "score_j30": None,
                "score_j60": None,
                "score_j90": None,
                "dates_seuil": {},
                "coeffs": [],
                "degre": 0,
            }

        # Vérifier que la plage temporelle est suffisante
        if x_days.max() - x_days.min() < 2.0:
            return {
                "status": "insufficient_data",
                "n_points": len(scores),
                "message": "Historique trop court (moins de 2 jours) pour une prédiction fiable",
                "score_actuel": float(y[-1]),
                "confiance_pct": 0,
                "r2": 0,
                "pente_actuelle": 0,
                "score_j30": None,
                "score_j60": None,
                "score_j90": None,
                "dates_seuil": {},
                "coeffs": [],
                "degre": 0,
            }

        # Choisir le degré selon le nombre de points
        deg = min(2, len(scores) - 1)

        # Régression polynomiale avec sécurité
        try:
            coeffs = np.polyfit(x_days, y, deg)
        except np.linalg.LinAlgError:
            # Fallback : régression linéaire si l'ajustement polynomial échoue
            if len(scores) >= 2:
                coeffs = np.polyfit(x_days, y, 1)
            else:
                return self._insufficient()

        poly = np.poly1d(coeffs)

        # Qualité d'ajustement (R²)
        y_pred = poly(x_days)
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - np.mean(y)) ** 2)
        r2 = float(1 - ss_res / (ss_tot + 1e-9))

        # Pente actuelle (dérivée au dernier point)
        dpoly = poly.deriv()
        pente = float(dpoly(x_days[-1]))

        # Projeter jusqu'au seuil critique
        today_x = x_days[-1]
        dates_seuil = {}
        for seuil_label, seuil_val in [
            ("critique",  self.SEUIL_CRITIQUE),
            ("urgent",    self.SEUIL_URGENT),
            ("planifie",  self.SEUIL_PLANIFIE),
        ]:
            d = self._find_crossing(poly, today_x, seuil_val)
            if d is not None:
                date_abs = t0 + datetime.timedelta(days=int(d))
                days_left = int(d - today_x)
                dates_seuil[seuil_label] = {
                    "date":      date_abs.isoformat(),
                    "jours":     max(days_left, 0),
                    "depasse":   days_left < 0,
                }
            else:
                dates_seuil[seuil_label] = None

        # Score prédit dans 30 / 60 / 90 jours (borné)
        score_j30 = min(100, max(0, round(float(poly(today_x + 30)), 1)))
        score_j60 = min(100, max(0, round(float(poly(today_x + 60)), 1)))
        score_j90 = min(100, max(0, round(float(poly(today_x + 90)), 1)))

        # Confiance de la prédiction
        confiance = self._confidence(r2, len(scores), pente)

        return {
            "status":       "ok",
            "n_points":     len(scores),
            "r2":           round(r2, 3),
            "pente_actuelle": round(pente, 3),
            "score_actuel": round(float(y[-1]), 1),
            "score_j30":    score_j30,
            "score_j60":    score_j60,
            "score_j90":    score_j90,
            "dates_seuil":  dates_seuil,
            "confiance_pct": confiance,
            "coeffs":       coeffs.tolist(),
            "degre":        deg,
        }

    def _find_crossing(self, poly: np.poly1d,
                       x_start: float,
                       threshold: float,
                       horizon: int = 730) -> Optional[float]:
        """Trouve le premier x où poly(x) >= threshold."""
        if float(poly(x_start)) >= threshold:
            return x_start   # Déjà franchi
        xs = np.linspace(x_start, x_start + horizon, 1000)
        ys = poly(xs)
        crossing = np.where(ys >= threshold)[0]
        if len(crossing) == 0:
            return None
        return float(xs[crossing[0]])

    def _confidence(self, r2: float,
                    n: int, pente: float) -> float:
        """Score de confiance de la prédiction (0–100)."""
        r2_score  = max(r2, 0) * 40          # max 40 pts pour R²
        n_score   = min(n / 20, 1.0) * 35    # max 35 pts pour nb points
        pente_pen = max(0, 1 - abs(pente) / 20) * 25  # pente stable = fiable
        return round(r2_score + n_score + pente_pen, 1)

    def _parse_date(self, d: str) -> datetime.date:
        try:
            return datetime.datetime.fromisoformat(d).date()
        except Exception:
            return datetime.date.today()

    def _insufficient(self) -> dict:
        return {
            "status":        "insufficient_data",
            "n_points":      0,
            "message":       "Minimum 2 analyses requises pour la prédiction",
            "score_j30":     None, "score_j60": None, "score_j90": None,
            "dates_seuil":   {}, "confiance_pct": 0,
            "score_actuel":  0,
            "r2": 0,
            "pente_actuelle": 0,
            "coeffs": [],
            "degre": 0,
        }


# ══════════════════════════════════════════════════════════════════════════
# SYSTÈME D'ALERTES
# ══════════════════════════════════════════════════════════════════════════

ALERT_LEVELS = {
    "CRITIQUE": {
        "couleur": "#DC2626", "emoji": "🔴",
        "description": "Panne imminente ou déjà en cours",
        "delai_intervention": "Immédiat — avant le prochain patient",
    },
    "ROUGE": {
        "couleur": "#EA580C", "emoji": "🟠",
        "description": "Panne prévue dans moins de 7 jours",
        "delai_intervention": "Dans les 24–48 heures",
    },
    "ORANGE": {
        "couleur": "#CA8A04", "emoji": "🟡",
        "description": "Panne prévue dans moins de 30 jours",
        "delai_intervention": "Dans la semaine",
    },
    "JAUNE": {
        "couleur": "#2563EB", "emoji": "🔵",
        "description": "Panne prévue dans 30–90 jours",
        "delai_intervention": "Planifier dans le mois",
    },
    "VERT": {
        "couleur": "#16A34A", "emoji": "🟢",
        "description": "Composant stable — aucune panne prévue à court terme",
        "delai_intervention": "Maintenance préventive standard",
    },
}


class AlertSystem:
    """
    Génère les alertes à partir des prédictions.
    Peut enregistrer les alertes dans un fichier journal.
    """

    def __init__(self, alert_log_path: str = "./alertes.json"):
        self.log_path = alert_log_path

    def compute_alert(self,
                      component_id:   str,
                      component_nom:  str,
                      regression:     dict,
                      weibull:        dict,
                      score_actuel:   float) -> dict:
        """Calcule le niveau d'alerte pour un composant."""

        level = "VERT"
        jours_avant_panne = None

        # Critère 1 : score actuel déjà critique
        if score_actuel >= 85:
            level = "CRITIQUE"
            jours_avant_panne = 0

        # Critère 2 : régression — jours avant seuil critique
        elif regression.get("status") == "ok":
            ds = regression.get("dates_seuil", {})
            crit = ds.get("critique")
            urg  = ds.get("urgent")

            if crit and not crit.get("depasse", False):
                j = crit["jours"]
                jours_avant_panne = j
                if j <= 7:    level = "ROUGE"
                elif j <= 30: level = "ORANGE"
                elif j <= 90: level = "JAUNE"
            elif urg and not urg.get("depasse", False):
                j = urg["jours"]
                jours_avant_panne = j
                if j <= 14:   level = "ORANGE"
                elif j <= 60: level = "JAUNE"

        # Critère 3 : probabilité Weibull 30j > 60%
        prob_w30 = weibull.get("prob_panne_30j", 0)
        if prob_w30 >= 80 and level not in ("CRITIQUE", "ROUGE"):
            level = "ROUGE"
        elif prob_w30 >= 50 and level == "VERT":
            level = "ORANGE"

        alert_info = ALERT_LEVELS[level]

        return {
            "composant_id":       component_id,
            "composant_nom":      component_nom,
            "niveau":             level,
            "couleur":            alert_info["couleur"],
            "emoji":              alert_info["emoji"],
            "description":        alert_info["description"],
            "delai_intervention": alert_info["delai_intervention"],
            "jours_avant_panne":  jours_avant_panne,
            "score_actuel":       score_actuel,
            "prob_panne_30j":     prob_w30,
            "date_alerte":        datetime.datetime.now().isoformat(),
        }

    def log_alert(self, device_id: str, alert: dict):
        """Enregistre l'alerte dans le journal."""
        try:
            if os.path.exists(self.log_path):
                with open(self.log_path, "r") as f:
                    log = json.load(f)
            else:
                log = []
            log.append({"device_id": device_id, **alert})
            log = log[-500:]   # 500 alertes max
            with open(self.log_path, "w") as f:
                json.dump(log, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def get_active_alerts(self, device_id: str) -> List[dict]:
        """Retourne les alertes actives (ORANGE, ROUGE, CRITIQUE)."""
        try:
            if not os.path.exists(self.log_path):
                return []
            with open(self.log_path, "r") as f:
                log = json.load(f)
            actives = [
                e for e in log
                if e.get("device_id") == device_id
                and e.get("niveau") in ("CRITIQUE", "ROUGE", "ORANGE")
            ]
            return actives[-20:]
        except Exception:
            return []


# ══════════════════════════════════════════════════════════════════════════
# MOTEUR PRÉDICTIF PRINCIPAL
# ══════════════════════════════════════════════════════════════════════════

class PredictiveEngine:
    """
    Orchestre la prédiction temporelle niveau 3.

    Pour chaque composant :
      1. Régression sur l'historique des scores → date de panne estimée
      2. Modèle de Weibull → probabilité de survie
      3. Combinaison → alerte graduée
      4. Fenêtre d'intervention optimale
    """

    def __init__(self,
                 device_id:    str,
                 history_dir:  str = "./historique",
                 alert_log:    str = "./alertes.json"):
        self.device_id  = device_id
        self.history_dir = history_dir
        self.alert_sys  = AlertSystem(alert_log)
        self.regressor  = ScoreRegressor()

    def _load_history(self) -> List[dict]:
        path = os.path.join(
            self.history_dir,
            self.device_id.replace(" ", "_").replace("/", "-") + ".json"
        )
        if not os.path.exists(path):
            return []
        with open(path, "r") as f:
            return json.load(f)

    def predict(self,
                current_diagnosis: dict,
                component_ages:    Dict[str, float] = None,
                component_knowledge: dict = None) -> dict:
        """
        Effectue la prédiction temporelle complète.

        Parameters
        ----------
        current_diagnosis    : sortie de DiagnosticEngine.diagnose()
        component_ages       : âge en années de chaque composant
        component_knowledge  : base de connaissances des composants

        Returns
        -------
        dict complet avec prédictions, alertes, fenêtres d'intervention
        """
        from component_knowledge import COMPONENTS
        comp_kb = component_knowledge or COMPONENTS

        component_ages = component_ages or {}
        history = self._load_history()

        predictions = []
        all_alerts  = []

        comp_diags = current_diagnosis.get("component_diagnostics", [])
        if not comp_diags:
            return self._empty_prediction()

        for cd in comp_diags:
            comp_id  = cd["composant_id"]
            comp_nom = cd["composant_nom"]
            score    = cd["score_risque"]

            # ── Régression temporelle ─────────────────────────────────
            dates_h, scores_h = [], []
            for entry in history:
                cs = entry.get("component_scores", {})
                if comp_id in cs:
                    dates_h.append(entry["date"])
                    scores_h.append(cs[comp_id])

            regression = self.regressor.fit(dates_h, scores_h)

            # ── Modèle de Weibull ─────────────────────────────────────
            weibull_model = WeibullModel.from_component(comp_id, comp_kb)
            age_h = component_ages.get(comp_id, 0) * 8760  # ans → heures
            weibull_data  = weibull_model.remaining_life(age_h)
            weibull_curve = weibull_model.project_curve(age_h, 180)

            # Proba panne 30/90j via Weibull
            p30 = weibull_model.failure_prob(age_h + 30 * 24) * 100
            p90 = weibull_model.failure_prob(age_h + 90 * 24) * 100

            weibull_preds = {
                **weibull_data,
                "prob_panne_30j": round(p30, 1),
                "prob_panne_90j": round(p90, 1),
                "curve":          weibull_curve[:13],  # 3 mois
            }

            # ── Fenêtre d'intervention optimale ──────────────────────
            window = self._optimal_window(regression, weibull_preds, score)

            # ── Alerte ────────────────────────────────────────────────
            alert = self.alert_sys.compute_alert(
                comp_id, comp_nom, regression, weibull_preds, score
            )
            self.alert_sys.log_alert(self.device_id, alert)
            all_alerts.append(alert)

            predictions.append({
                "composant_id":    comp_id,
                "composant_nom":   comp_nom,
                "sous_systeme":    cd["sous_systeme"],
                "score_actuel":    score,
                "regression":      regression,
                "weibull":         weibull_preds,
                "alerte":          alert,
                "fenetre_intervention": window,
                "actions_prioritaires": cd.get("actions", [])[:3],
            })

        # Alerte globale = pire alerte composant
        priority = {"CRITIQUE":0,"ROUGE":1,"ORANGE":2,"JAUNE":3,"VERT":4}
        all_alerts_sorted = sorted(
            all_alerts, key=lambda a: priority.get(a["niveau"], 9)
        )
        alerte_globale = all_alerts_sorted[0] if all_alerts_sorted else None

        return {
            "device_id":       self.device_id,
            "date_prediction": datetime.datetime.now().isoformat(),
            "predictions":     predictions,
            "alerte_globale":  alerte_globale,
            "n_alertes_actives": sum(
                1 for a in all_alerts
                if a["niveau"] in ("CRITIQUE","ROUGE","ORANGE")
            ),
            "resume_prediction": self._resume(predictions, alerte_globale),
        }

    def _optimal_window(self,
                        regression: dict,
                        weibull:    dict,
                        score:      float) -> dict:
        """
        Détermine la fenêtre d'intervention optimale :
        - Pas trop tôt (coût inutile)
        - Pas trop tard (risque de panne en production)
        """
        date_panne = None
        jours_panne = None

        ds = regression.get("dates_seuil", {})
        if ds.get("critique") and not ds["critique"].get("depasse"):
            jours_panne = ds["critique"]["jours"]
            date_panne  = ds["critique"]["date"]
        elif weibull.get("remaining_days", 0) > 0:
            jours_panne = int(weibull["remaining_days"])
            date_panne  = (datetime.date.today() +
                           datetime.timedelta(days=jours_panne)).isoformat()

        if jours_panne is None:
            # Déjà critique
            return {
                "debut": datetime.date.today().isoformat(),
                "fin":   datetime.date.today().isoformat(),
                "duree_jours": 0,
                "message": "Intervention immédiate requise",
                "optimal": True,
            }

        # Fenêtre = [70% de la durée restante, 85% de la durée restante]
        debut_j = max(0, int(jours_panne * 0.65))
        fin_j   = max(0, int(jours_panne * 0.85))

        debut_date = (datetime.date.today() +
                      datetime.timedelta(days=debut_j)).isoformat()
        fin_date   = (datetime.date.today() +
                      datetime.timedelta(days=fin_j)).isoformat()

        return {
            "debut":         debut_date,
            "fin":           fin_date,
            "duree_jours":   fin_j - debut_j,
            "date_panne_est":date_panne,
            "jours_restants":jours_panne,
            "message":       (
                f"Intervenir entre J+{debut_j} et J+{fin_j} "
                f"(panne estimée à J+{jours_panne})"
            ),
            "optimal": True,
        }

    def _resume(self, predictions: List[dict],
                alerte_globale: Optional[dict]) -> str:
        if not predictions or not alerte_globale:
            return "Aucune prédiction disponible."
        top = predictions[0]
        w   = top["fenetre_intervention"]
        return (
            f"Composant prioritaire : {top['composant_nom']}. "
            f"Alerte : {alerte_globale['niveau']}. "
            f"Fenêtre d'intervention optimale : {w.get('debut','?')} "
            f"→ {w.get('fin','?')}. "
            f"Date de panne estimée : {w.get('date_panne_est','inconnue')}."
        )

    def _empty_prediction(self) -> dict:
        return {
            "device_id":       self.device_id,
            "date_prediction": datetime.datetime.now().isoformat(),
            "predictions":     [],
            "alerte_globale":  None,
            "n_alertes_actives": 0,
            "resume_prediction": "Aucun composant à risque — appareil conforme.",
        }