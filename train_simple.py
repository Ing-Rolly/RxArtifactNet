"""
train_simple.py
===============
Entraînement par dossiers — VERSION CORRIGÉE.
12 classes machine uniquement. Structure attendue :

dataset_simple/
├── train/
│   ├── overexposure/          ← images avec surexposition machine
│   ├── underexposure/
│   ├── focal_spot_defect/
│   ├── anode_artifact/
│   ├── collimator_misalignment/
│   ├── grid_lines/
│   ├── grid_cutoff/
│   ├── dead_pixels/
│   ├── ghosting_dr/
│   ├── cr_scratch/
│   ├── cr_fog/
│   └── processing_artifact/
└── val/
    └── (mêmes dossiers)

LANCEMENT :
  python train_simple.py --demo
  python train_simple.py --data_dir ./dataset_simple --epochs 60
"""

import os, sys, time, argparse, pickle
import numpy as np
import cv2
from pathlib import Path
from typing import List, Tuple, Dict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_base    import CLASS_NAMES, NUM_CLASSES
from model             import RxArtifactNet
from preprocessing     import RxPreprocessor

# ── Vérification : 12 classes exactement ─────────────────────────────────
assert NUM_CLASSES == 12, f"Erreur : {NUM_CLASSES} classes trouvées, 12 attendues"

DEFAULT_CONFIG = {
    "data_dir":       "./dataset_simple",
    "output_dir":     "./trained_model_simple",
    "epochs":         10,
    "batch_size":     8,
    "learning_rate":  1e-3,
    "lr_decay":       0.95,
    "augment":        True,
    "save_every":     10,
    "early_stopping": 15,
    "input_size":     (512, 512),
}

VALID_EXTS = {".png", ".jpg", ".jpeg", ".dcm"}


# ══════════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════════

class MachineFaultDataset:
    """
    Dataset organisé par dossiers — 12 classes machine uniquement.
    Un dossier = une classe d'artefact machine.
    """

    def __init__(self, split_dir: str, input_size=(512, 512),
                 augment=False):
        self.split_dir    = split_dir
        self.preprocessor = RxPreprocessor(target_size=input_size,
                                            augment=augment)
        self.samples: List[Tuple[str, int]] = []
        self.class_counts: Dict[str, int]   = {}
        self._scan()

    def _scan(self):
        if not os.path.exists(self.split_dir):
            raise FileNotFoundError(
                f"Dossier introuvable : {self.split_dir}\n\n"
                f"Créez la structure suivante :\n"
                + "\n".join(f"  {self.split_dir}/{c}/" for c in CLASS_NAMES)
            )

        unknown = []
        for entry in sorted(os.scandir(self.split_dir), key=lambda e: e.name):
            if not entry.is_dir():
                continue
            cls = entry.name
            if cls not in CLASS_NAMES:
                unknown.append(cls)
                continue
            idx   = CLASS_NAMES.index(cls)
            count = 0
            for f in os.scandir(entry.path):
                if Path(f.name).suffix.lower() in VALID_EXTS:
                    self.samples.append((f.path, idx))
                    count += 1
            if count > 0:
                self.class_counts[cls] = count

        split = os.path.basename(self.split_dir)
        print(f"\n  [{split}] {len(self.samples)} images — "
              f"{len(self.class_counts)}/{NUM_CLASSES} classes présentes")

        for cls in CLASS_NAMES:
            n   = self.class_counts.get(cls, 0)
            bar = "█" * min(n, 30) if n > 0 else "░ (absent)"
            print(f"    {cls:<30} {bar} ({n})")

        if unknown:
            print(f"\n  ATTENTION — dossiers ignorés (nom inconnu) : {unknown}")
            print(f"  Noms valides : {CLASS_NAMES}")

        missing = [c for c in CLASS_NAMES if c not in self.class_counts]
        if missing:
            print(f"\n  Classes manquantes : {missing}")
            print("  Le modèle ne pourra pas apprendre ces classes.")

        self._check_balance()

    def _check_balance(self):
        if len(self.class_counts) < 2:
            return
        counts = list(self.class_counts.values())
        ratio  = max(counts) / (min(counts) + 1e-9)
        if ratio > 5:
            maxi = max(self.class_counts, key=self.class_counts.get)
            mini = min(self.class_counts, key=self.class_counts.get)
            print(f"\n  DÉSÉQUILIBRE (ratio {ratio:.1f}x) :")
            print(f"    Plus représentée : {maxi} ({self.class_counts[maxi]})")
            print(f"    Moins représentée: {mini} ({self.class_counts[mini]})")
            print("    Conseil : visez un ratio < 3x")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, cls_idx = self.samples[idx]
        tensor, _ = self.preprocessor.process_file(path)
        gt = np.zeros(NUM_CLASSES, dtype=np.float32)
        gt[cls_idx] = 1.0
        return tensor, gt

    def get_all(self) -> List[Tuple]:
        print(f"  Chargement de {len(self)} images...")
        data = []
        for i in range(len(self)):
            data.append(self[i])
            if (i+1) % 20 == 0 or (i+1) == len(self):
                pct = (i+1) / len(self) * 100
                bar = "█" * int(pct/5) + "░" * (20 - int(pct/5))
                print(f"\r    [{bar}] {i+1}/{len(self)}", end="", flush=True)
        print()
        return data


# ══════════════════════════════════════════════════════════════════════════
# MÉTRIQUES
# ══════════════════════════════════════════════════════════════════════════

def bce_loss(pred, target):
    pred = np.clip(pred, 1e-7, 1-1e-7)
    return float(-np.mean(target*np.log(pred) + (1-target)*np.log(1-pred)))

def cce_loss(probs, target, eps=1e-9):
    """Cross-entropy catégorielle — adaptée à une sortie softmax + cible one-hot."""
    probs = np.clip(probs, eps, 1.0)
    return float(-np.sum(target * np.log(probs)))

def top1_acc(pred, target):
    return float(np.argmax(pred) == np.argmax(target))

def per_class_report(preds, targets):
    """
    Calcule les métriques retenues pour l'évaluation (cf. Chapitre 3) :
    Précision, Rappel, F1-score par classe, F1-macro, F1-pondéré,
    Accuracy globale, et matrice de confusion — en NumPy pur.

    NB : le Dice/IoU (segmentation) n'est pas calculé ici car la tête de
    segmentation (seg_head) n'est pas entraînée par ce script (cf. limites,
    Chapitre 3, section 3.y.3).
    """
    n = len(CLASS_NAMES)
    y_true = [int(np.argmax(t)) for t in targets]
    y_pred = [int(np.argmax(p)) for p in preds]

    # ── Matrice de confusion (lignes = réel, colonnes = prédit) ──────────
    cm = np.zeros((n, n), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[yt, yp] += 1

    tp = np.diag(cm).astype(float)
    fp = cm.sum(axis=0) - tp   # faux positifs = prédit classe c, mais pas réellement c
    fn = cm.sum(axis=1) - tp   # faux négatifs = réellement classe c, mais pas prédit c
    support = cm.sum(axis=1)   # nb d'images réelles par classe

    with np.errstate(divide="ignore", invalid="ignore"):
        precision = np.where((tp+fp) > 0, tp/(tp+fp), 0.0)
        recall    = np.where((tp+fn) > 0, tp/(tp+fn), 0.0)
        f1        = np.where((precision+recall) > 0,
                              2*precision*recall/(precision+recall), 0.0)

    accuracy    = float(tp.sum() / max(1, cm.sum()))
    f1_macro    = float(np.mean(f1[support > 0])) if (support > 0).any() else 0.0
    total_sup   = support[support > 0].sum()
    f1_weighted = float(np.sum(f1[support > 0] * support[support > 0]) / total_sup) \
                  if total_sup > 0 else 0.0

    print("\n  Précision / Rappel / F1-score par classe :")
    print(f"    {'Classe':<28}{'Précision':>10}{'Rappel':>10}{'F1':>10}{'Support':>10}")
    for i, cls in enumerate(CLASS_NAMES):
        if support[i] == 0:
            continue
        flag = "  ⚠" if f1[i] < 0.7 else ""
        print(f"    {cls:<28}{precision[i]*100:>9.1f}%{recall[i]*100:>9.1f}%"
              f"{f1[i]*100:>9.1f}%{int(support[i]):>10}{flag}")

    print(f"\n  Accuracy globale : {accuracy*100:.1f}%")
    print(f"  F1-macro         : {f1_macro*100:.1f}%")
    print(f"  F1-pondéré       : {f1_weighted*100:.1f}%")

    print("\n  Matrice de confusion (lignes=réel, colonnes=prédit) :")
    present = [i for i in range(n) if support[i] > 0]
    header = "".join(f"{CLASS_NAMES[j][:6]:>8}" for j in present)
    print(f"    {'':<10}{header}")
    for i in present:
        row = "".join(f"{cm[i,j]:>8}" for j in present)
        print(f"    {CLASS_NAMES[i][:10]:<10}{row}")

    return {
        "confusion_matrix": cm.tolist(),
        "precision":  {CLASS_NAMES[i]: float(precision[i]) for i in present},
        "recall":     {CLASS_NAMES[i]: float(recall[i])    for i in present},
        "f1":         {CLASS_NAMES[i]: float(f1[i])        for i in present},
        "support":    {CLASS_NAMES[i]: int(support[i])     for i in present},
        "accuracy":   accuracy,
        "f1_macro":   f1_macro,
        "f1_weighted": f1_weighted,
    }


# ══════════════════════════════════════════════════════════════════════════
# TRAINER
# ══════════════════════════════════════════════════════════════════════════

class SimpleTrainer:
    def __init__(self, model, config):
        self.model  = model
        self.config = config
        self.lr     = config["learning_rate"]
        self.rng    = np.random.default_rng(42)
        self.best   = float("inf")
        self.no_imp = 0
        self.history = {"train_loss":[], "val_loss":[],
                        "train_acc":[],  "val_acc":[]}

    def _update(self, features, probs, target):
        """
        Descente de gradient analytique (softmax + cross-entropie catégorielle)
        appliquée UNIQUEMENT à la tête de classification (cls_fc_W, cls_fc_b).
        L'encodeur/décodeur restent figés (approche type Extreme Learning Machine :
        extracteur de caractéristiques aléatoire fixe + classifieur entraîné [Huang et al., 2006]).

        Gradient : dL/dlogits = probs - target (propriété du couple softmax + CE)
                   dL/dW      = outer(dL/dlogits, features)
                   dL/db      = dL/dlogits
        """
        grad_logits = probs - target                    # (12,)
        grad_W      = np.outer(grad_logits, features)    # (12, feat_dim)
        grad_b      = grad_logits                        # (12,)

        self.model.cls_fc_W -= self.lr * grad_W
        self.model.cls_fc_b -= self.lr * grad_b

    def train_epoch(self, data, epoch):
        idx = self.rng.permutation(len(data))
        losses, accs = [], []
        bs = self.config["batch_size"]
        nb = max(1, len(data) // bs)
        for b in range(nb):
            batch = idx[b*bs:(b+1)*bs]
            bl, ba = [], []
            for i in batch:
                tensor, gt = data[int(i)]
                out      = self.model.forward(tensor)
                probs    = out["cls_probs"][0]
                features = out["features"][0]
                bl.append(cce_loss(probs, gt))
                ba.append(top1_acc(probs, gt))
                self._update(features, probs, gt)
            losses.append(np.mean(bl)); accs.append(np.mean(ba))
            pct = (b+1)/nb
            bar = "█"*int(pct*25) + "░"*(25-int(pct*25))
            print(f"\r  Epoch {epoch:3d} [{bar}] loss={np.mean(losses):.4f} acc={np.mean(accs)*100:.1f}%",
                  end="", flush=True)
        print()
        return {"loss": float(np.mean(losses)), "acc": float(np.mean(accs))}

    def evaluate(self, data, split="val"):
        losses, accs, preds, targets = [], [], [], []
        for tensor, gt in data:
            out   = self.model.forward(tensor)
            probs = out["cls_probs"][0]
            losses.append(cce_loss(probs, gt))
            accs.append(top1_acc(probs, gt))
            preds.append(probs); targets.append(gt)
        loss = float(np.mean(losses)); acc = float(np.mean(accs))
        print(f"  [{split:5s}]  loss={loss:.4f}  acc={acc*100:.1f}%")
        return {"loss": loss, "acc": acc, "preds": preds, "targets": targets}

    def lr_step(self):
        self.lr *= self.config["lr_decay"]

    def should_stop(self):
        return self.no_imp >= self.config["early_stopping"]


# ══════════════════════════════════════════════════════════════════════════
# DONNÉES SYNTHÉTIQUES (mode démo)
# ══════════════════════════════════════════════════════════════════════════

def make_demo_data(n_train=36, n_val=12):
    """Génère des données synthétiques pour les 12 classes machine."""
    preprocessor = RxPreprocessor(augment=True)
    # 3 images par classe pour le train, 1 pour le val
    n_per_class_train = max(1, n_train // NUM_CLASSES)
    n_per_class_val   = max(1, n_val   // NUM_CLASSES)

    def make_split(n_per_cls, seed_off=0):
        data = []
        for ci, cls in enumerate(CLASS_NAMES):
            for j in range(n_per_cls):
                seed   = ci * 100 + j + seed_off
                tensor, _ = preprocessor.create_synthetic_rx(
                    artifact=cls, seed=seed
                )
                gt = np.zeros(NUM_CLASSES, dtype=np.float32)
                gt[ci] = 1.0
                data.append((tensor, gt))
        return data

    print(f"  Génération démo : {n_per_class_train} images × {NUM_CLASSES} classes (train)")
    print(f"                    {n_per_class_val}   images × {NUM_CLASSES} classes (val)")
    return make_split(n_per_class_train, 0), make_split(n_per_class_val, 9999)


# ══════════════════════════════════════════════════════════════════════════
# POINT D'ENTRÉE
# ══════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",   default=DEFAULT_CONFIG["data_dir"])
    p.add_argument("--output_dir", default=DEFAULT_CONFIG["output_dir"])
    p.add_argument("--epochs",     type=int,   default=DEFAULT_CONFIG["epochs"])
    p.add_argument("--batch_size", type=int,   default=DEFAULT_CONFIG["batch_size"])
    p.add_argument("--lr",         type=float, default=DEFAULT_CONFIG["learning_rate"])
    p.add_argument("--no_augment", action="store_true")
    p.add_argument("--resume",     default=None)
    p.add_argument("--demo",       action="store_true")
    return p.parse_args()


def main():
    args   = parse_args()
    config = DEFAULT_CONFIG.copy()
    config.update({
        "data_dir":      args.data_dir,
        "output_dir":    args.output_dir,
        "epochs":        args.epochs,
        "batch_size":    args.batch_size,
        "learning_rate": args.lr,
        "augment":       not args.no_augment,
    })
    os.makedirs(config["output_dir"], exist_ok=True)

    print("\n" + "═"*60)
    print("  RxArtifactNet — Entraînement (12 artefacts machine)")
    print("═"*60)
    print(f"  Classes : {NUM_CLASSES}")
    print(f"  Dataset : {config['data_dir']}")
    print(f"  Sorties : {config['output_dir']}")
    print(f"  Époques : {config['epochs']}  |  Batch : {config['batch_size']}")

    for i, cls in enumerate(CLASS_NAMES, 1):
        print(f"    {i:2d}. {cls}")

    model = RxArtifactNet(seed=42)
    model.summary()

    if args.resume and os.path.exists(args.resume):
        model.load_weights(args.resume)

    trainer = SimpleTrainer(model, config)

    # ── Données ────────────────────────────────────────────────────────────
    if args.demo:
        print("\n[Données] Mode démo — données synthétiques")
        train_data, val_data = make_demo_data()
    else:
        train_dir = os.path.join(config["data_dir"], "train")
        val_dir   = os.path.join(config["data_dir"], "val")
        if not os.path.exists(train_dir):
            print(f"\n  ERREUR : {train_dir} introuvable.")
            print("  Lancez avec --demo pour tester sans données réelles.")
            print(f"\n  Structure attendue :")
            for cls in CLASS_NAMES:
                print(f"    {config['data_dir']}/train/{cls}/rx_001.png")
            sys.exit(1)
        train_data = MachineFaultDataset(train_dir,
                                         config["input_size"],
                                         config["augment"]).get_all()
        val_data   = MachineFaultDataset(val_dir,
                                         config["input_size"],
                                         False).get_all()

    print(f"\n  Train : {len(train_data)} images")
    print(f"  Val   : {len(val_data)} images")

    # ── Entraînement ────────────────────────────────────────────────────────
    print("\n" + "═"*60)
    t0 = time.time()

    for epoch in range(1, config["epochs"] + 1):
        t_ep = time.time()
        tm   = trainer.train_epoch(train_data, epoch)
        vm   = trainer.evaluate(val_data)

        trainer.lr_step()
        trainer.history["train_loss"].append(tm["loss"])
        trainer.history["val_loss"].append(vm["loss"])
        trainer.history["train_acc"].append(tm["acc"])
        trainer.history["val_acc"].append(vm["acc"])

        is_best = vm["loss"] < trainer.best
        if is_best:
            trainer.best   = vm["loss"]
            trainer.no_imp = 0
            best_path = os.path.join(config["output_dir"], "best_model.pkl")
            model.save_weights(best_path)
            print(f"  ★ Meilleur modèle → {best_path}")
        else:
            trainer.no_imp += 1

        if epoch % config["save_every"] == 0:
            model.save_weights(os.path.join(
                config["output_dir"], f"checkpoint_{epoch:03d}.pkl"
            ))

        ep_t = time.time() - t_ep
        eta  = (config["epochs"] - epoch) * ep_t
        print(f"  Époque {epoch:3d}/{config['epochs']}  "
              f"LR={trainer.lr:.2e}  {ep_t:.1f}s  ETA={eta/60:.1f}min")

        if trainer.should_stop():
            print(f"\n  Early stopping à l'époque {epoch}")
            break

    # ── Résultats ────────────────────────────────────────────────────────────
    total = time.time() - t0
    print(f"\n  Terminé en {total/60:.1f} min")

    final_val = trainer.evaluate(val_data, "final")
    metrics = per_class_report(final_val["preds"], final_val["targets"])

    model.save_weights(os.path.join(config["output_dir"], "final_model.pkl"))
    with open(os.path.join(config["output_dir"], "history.pkl"), "wb") as f:
        pickle.dump(trainer.history, f)

    # Export des métriques pour le Chapitre 3 (Tableau A, matrice de confusion)
    import json
    metrics_path = os.path.join(config["output_dir"], "metriques_evaluation.json")
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    print(f"\n  Métriques exportées → {metrics_path}")
    print("  (à copier directement dans le Tableau A du Chapitre 3)")

    print("\n" + "═"*60)
    print("  Pour utiliser le modèle :")
    print(f"    from pipeline import RxAnalysisPipeline")
    print(f"    pipeline = RxAnalysisPipeline(")
    print(f"        weights_path='{config['output_dir']}/best_model.pkl'")
    print(f"    )")
    print("═"*60)


if __name__ == "__main__":
    main()
