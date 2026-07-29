"""
model.py – ResNet-18 NumPy avec ajustement automatique des dimensions
"""

import numpy as np
import pickle
from knowledge_base import CLASS_NAMES, NUM_CLASSES

assert NUM_CLASSES == 12

def relu(x):        return np.maximum(0, x)
def sigmoid(x):     return 1 / (1 + np.exp(-np.clip(x, -50, 50)))
def softmax(x, axis=-1):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / (e.sum(axis=axis, keepdims=True) + 1e-9)

def batch_norm(x, eps=1e-5):
    mean = x.mean(axis=(0,2,3), keepdims=True)
    std  = x.std(axis=(0,2,3), keepdims=True) + eps
    return (x - mean) / std

def fast_conv2d(x, W, b, stride=1, padding=1):
    N, C, H, Ww = x.shape
    F, C_, kH, kW = W.shape
    xp = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)))
    Ho = (H + 2*padding - kH) // stride + 1
    Wo = (Ww + 2*padding - kW) // stride + 1
    cols = np.lib.stride_tricks.sliding_window_view(xp, (kH, kW), axis=(2,3))[:, :, ::stride, ::stride, :, :]
    cols = cols.reshape(N, C*kH*kW, Ho*Wo)
    W_flat = W.reshape(F, -1)
    out = (W_flat @ cols).reshape(N, F, Ho, Wo) + b.reshape(1,F,1,1)
    return out

def maxpool2d(x, size=2, stride=2, padding=0):
    if padding > 0:
        x = np.pad(x, ((0,0),(0,0),(padding,padding),(padding,padding)),
                   mode="constant", constant_values=-np.inf)
    N, C, H, W = x.shape
    if size == stride and H % stride == 0 and W % stride == 0:
        Ho, Wo = H // stride, W // stride
        x_blocks = x.reshape(N, C, Ho, stride, Wo, stride)
        return x_blocks.max(axis=(3, 5))
    Ho = (H - size) // stride + 1
    Wo = (W - size) // stride + 1
    out = np.zeros((N, C, Ho, Wo), dtype=x.dtype)
    for i in range(Ho):
        for j in range(Wo):
            out[:,:,i,j] = x[:,:,i*stride:i*stride+size,
                               j*stride:j*stride+size].max(axis=(2,3))
    return out

def upsample2d(x, scale=2):
    return x.repeat(scale, axis=2).repeat(scale, axis=3)

def global_avg_pool(x): return x.mean(axis=(2,3))
def linear(x, W, b):   return x @ W.T + b

def raw_intensity_stats(x):
    mean = x.mean(axis=(1,2,3))
    std  = x.std(axis=(1,2,3))
    return np.stack([mean, std], axis=1).astype(np.float32)

class ResNetBasicBlock:
    def __init__(self, in_ch, out_ch, stride=1, seed=42):
        rng = np.random.default_rng(seed)
        scale1 = np.sqrt(2.0 / (in_ch * 9))
        scale2 = np.sqrt(2.0 / (out_ch * 9))
        self.W1 = rng.normal(0, scale1, (out_ch, in_ch, 3, 3)).astype(np.float32)
        self.b1 = np.zeros(out_ch, dtype=np.float32)
        self.W2 = rng.normal(0, scale2, (out_ch, out_ch, 3, 3)).astype(np.float32)
        self.b2 = np.zeros(out_ch, dtype=np.float32)
        self.stride = stride
        if stride != 1 or in_ch != out_ch:
            scale_s = np.sqrt(2.0 / (in_ch * 1))
            self.W_s = rng.normal(0, scale_s, (out_ch, in_ch, 1, 1)).astype(np.float32)
            self.b_s = np.zeros(out_ch, dtype=np.float32)
            self.has_shortcut = True
        else:
            self.has_shortcut = False

    def forward(self, x):
        out = fast_conv2d(x, self.W1, self.b1, stride=self.stride, padding=1)
        out = relu(out)
        out = fast_conv2d(out, self.W2, self.b2, stride=1, padding=1)
        if self.has_shortcut:
            shortcut = fast_conv2d(x, self.W_s, self.b_s, stride=self.stride, padding=0)
        else:
            shortcut = x
        return relu(out + shortcut)

class ResNetEncoder:
    def __init__(self, seed=42):
        rng = np.random.default_rng(seed)
        self.stem_W = rng.normal(0, np.sqrt(2.0/(3*49)), (64, 3, 7, 7)).astype(np.float32)
        self.stem_b = np.zeros(64, dtype=np.float32)
        self.layer1 = [ResNetBasicBlock(64,64,1,seed+10), ResNetBasicBlock(64,64,1,seed+11)]
        self.layer2 = [ResNetBasicBlock(64,128,2,seed+20), ResNetBasicBlock(128,128,1,seed+21)]
        self.layer3 = [ResNetBasicBlock(128,256,2,seed+30), ResNetBasicBlock(256,256,1,seed+31)]
        self.layer4 = [ResNetBasicBlock(256,512,2,seed+40), ResNetBasicBlock(512,512,1,seed+41)]

    def forward(self, x):
        x = fast_conv2d(x, self.stem_W, self.stem_b, stride=2, padding=3)
        x = relu(x)
        x = maxpool2d(x, size=3, stride=2, padding=1)
        x0 = x   # AVANT layer1 — nécessaire au bon branchement de dec1
        for b in self.layer1: x = b.forward(x)
        s1 = x
        for b in self.layer2: x = b.forward(x)
        s2 = x
        for b in self.layer3: x = b.forward(x)
        s3 = x
        for b in self.layer4: x = b.forward(x)
        s4 = x
        return x0, s1, s2, s3, s4

class DecoderBlock:
    def __init__(self, in_ch, out_ch, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (in_ch * 9))
        self.W = rng.normal(0, scale, (out_ch, in_ch, 3, 3)).astype(np.float32)
        self.b = np.zeros(out_ch, dtype=np.float32)

    def forward(self, x, skip):
        x = upsample2d(x)
        if x.shape[2] != skip.shape[2] or x.shape[3] != skip.shape[3]:
            x = x[:, :, :skip.shape[2], :skip.shape[3]]
        cat = np.concatenate([x, skip], axis=1)
        out = fast_conv2d(cat, self.W, self.b, stride=1, padding=1)
        return relu(out)

class RxArtifactNet:
    def __init__(self, seed=42):
        self.encoder = ResNetEncoder(seed=seed)
        self.dec4 = DecoderBlock(512+256,256,seed+500)
        self.dec3 = DecoderBlock(256+128,128,seed+600)
        self.dec2 = DecoderBlock(128+64,64,seed+700)
        self.dec1 = DecoderBlock(64+64,64,seed+800)
        rng = np.random.default_rng(seed+900)
        self.seg_head_W = rng.normal(0, 0.01, (NUM_CLASSES,64,1,1)).astype(np.float32)
        self.seg_head_b = np.zeros(NUM_CLASSES, dtype=np.float32)
        self.cls_fc_W = rng.normal(0, 0.01, (NUM_CLASSES,512+2)).astype(np.float32)
        self.cls_fc_b = np.zeros(NUM_CLASSES, dtype=np.float32)

    def forward_cls_only(self, x):
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        _, _, _, _, b = self.encoder.forward(x)
        gap = global_avg_pool(b)
        stats = raw_intensity_stats(x)
        feats = np.concatenate([gap, stats], axis=1)
        logits = linear(feats, self.cls_fc_W, self.cls_fc_b)
        probs = softmax(logits, axis=-1)
        return {"cls_probs": probs, "cls_logits": logits, "features": feats}

    def forward(self, x):
        if x.shape[1] == 1:
            x = np.repeat(x, 3, axis=1)
        x0, s1, s2, s3, s4 = self.encoder.forward(x)
        b = s4
        gap = global_avg_pool(b)
        stats = raw_intensity_stats(x)
        feats = np.concatenate([gap, stats], axis=1)
        
        # --- SÉCURITÉ CONVERGENCE MATRICIELLE ---
        # Si la couche finale attend 514 dims mais qu'on en a extrait 512 (ou vice-versa)
        if feats.shape[1] == 512 and self.cls_fc_W.shape[1] == 514:
            zeros = np.zeros((feats.shape[0], 2), dtype=np.float32)
            feats = np.concatenate([feats, zeros], axis=1)
        elif feats.shape[1] == 514 and self.cls_fc_W.shape[1] == 512:
            feats = feats[:, :512]
            
        logits = linear(feats, self.cls_fc_W, self.cls_fc_b)
        probs = softmax(logits, axis=-1)
        # Branchement correct : dec4->s3, dec3->s2, dec2->s1, dec1->x0
        d4 = self.dec4.forward(b,  s3)
        d3 = self.dec3.forward(d4, s2)
        d2 = self.dec2.forward(d3, s1)
        d1 = self.dec1.forward(d2, x0)
        seg = fast_conv2d(d1, self.seg_head_W, self.seg_head_b, stride=1, padding=0)
        while seg.shape[2] < x.shape[2]:
            seg = upsample2d(seg)
        seg = sigmoid(seg)
        return {"seg_map": seg, "cls_probs": probs, "cls_logits": logits, "features": feats}

    def predict(self, x, seg_threshold=0.35, cls_threshold=0.15):
        out = self.forward(x)
        seg = out["seg_map"][0]
        probs = out["cls_probs"][0]
        detections = []
        for i, cls_name in enumerate(CLASS_NAMES):
            p = float(probs[i])
            mask = seg[i]
            mask_area = float((mask > seg_threshold).mean())
            if p > cls_threshold or mask_area > 0.01:
                detections.append({
                    "class": cls_name,
                    "confidence": p,
                    "mask": mask,
                    "mask_bin": (mask > seg_threshold).astype(np.uint8),
                    "mask_area": mask_area,
                })
        detections.sort(key=lambda d: d["confidence"], reverse=True)
        return {"detections": detections, "all_probs": probs, "full_seg_map": seg}

    def save_weights(self, path):
        params = {"stem_W":self.encoder.stem_W, "stem_b":self.encoder.stem_b,
                  "layer1":[(b.W1,b.b1,b.W2,b.b2,b.W_s if b.has_shortcut else None,
                             b.b_s if b.has_shortcut else None) for b in self.encoder.layer1],
                  "layer2":[(b.W1,b.b1,b.W2,b.b2,b.W_s if b.has_shortcut else None,
                             b.b_s if b.has_shortcut else None) for b in self.encoder.layer2],
                  "layer3":[(b.W1,b.b1,b.W2,b.b2,b.W_s if b.has_shortcut else None,
                             b.b_s if b.has_shortcut else None) for b in self.encoder.layer3],
                  "layer4":[(b.W1,b.b1,b.W2,b.b2,b.W_s if b.has_shortcut else None,
                             b.b_s if b.has_shortcut else None) for b in self.encoder.layer4],
                  "dec4_W":self.dec4.W,"dec4_b":self.dec4.b,
                  "dec3_W":self.dec3.W,"dec3_b":self.dec3.b,
                  "dec2_W":self.dec2.W,"dec2_b":self.dec2.b,
                  "dec1_W":self.dec1.W,"dec1_b":self.dec1.b,
                  "seg_W":self.seg_head_W,"seg_b":self.seg_head_b,
                  "cls_W":self.cls_fc_W,"cls_b":self.cls_fc_b,
                  "num_classes":NUM_CLASSES}
        with open(path,"wb") as f: pickle.dump(params,f)
        print(f"[RxArtifactNet] Poids sauvegardés → {path}")

    def load_weights(self, path):
        with open(path,"rb") as f:
            params = pickle.load(f)
        if params.get("num_classes",0) != NUM_CLASSES:
            raise ValueError("Incompatibilité du nombre de classes.")
        self.encoder.stem_W = params["stem_W"]; self.encoder.stem_b = params["stem_b"]
        for layer, data in zip([self.encoder.layer1,self.encoder.layer2,
                                self.encoder.layer3,self.encoder.layer4],
                               [params["layer1"],params["layer2"],
                                params["layer3"],params["layer4"]]):
            for b, (W1,b1,W2,b2,Ws,bs) in zip(layer, data):
                b.W1,b.b1,b.W2,b.b2 = W1,b1,W2,b2
                if b.has_shortcut:
                    b.W_s,b.b_s = Ws,bs
        self.dec4.W,self.dec4.b = params["dec4_W"],params["dec4_b"]
        self.dec3.W,self.dec3.b = params["dec3_W"],params["dec3_b"]
        self.dec2.W,self.dec2.b = params["dec2_W"],params["dec2_b"]
        self.dec1.W,self.dec1.b = params["dec1_W"],params["dec1_b"]
        self.seg_head_W = params["seg_W"]; self.seg_head_b = params["seg_b"]

        cls_W = params["cls_W"]
        cls_b = params["cls_b"]
        if cls_W.shape[1] == 512:
            print("[RxArtifactNet] Ajustement : ajout de 2 colonnes de zéros à cls_W (stats globales)")
            zeros = np.zeros((cls_W.shape[0], 2), dtype=np.float32)
            cls_W = np.concatenate([cls_W, zeros], axis=1)
        self.cls_fc_W = cls_W
        self.cls_fc_b = cls_b
        print(f"[RxArtifactNet] Poids chargés ← {path}")

    def summary(self):
        print("="*62)
        print(f"{'RxArtifactNet — ResNet-18 NumPy — 12 artefacts':^62}")
        print("="*62)
        print(f"  Classes machine : {NUM_CLASSES}")
        print(f"  Classes : {CLASS_NAMES}")
        print("="*62)