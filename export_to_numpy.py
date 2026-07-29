"""
export_to_numpy.py
==================
Convertit les poids d'un ResUNetSegCls entraîné (PyTorch)
vers le format pickle de RxArtifactNet (NumPy).
"""

import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from model_torch import ResUNetSegCls
from knowledge_base import CLASS_NAMES, NUM_CLASSES

def fuse_conv_bn(conv, bn):
    if conv.bias is not None:
        conv_bias = conv.bias
    else:
        conv_bias = torch.zeros_like(bn.running_mean)
    gamma = bn.weight
    beta = bn.bias
    mean = bn.running_mean
    var = bn.running_var
    eps = bn.eps
    fused_W = conv.weight * (gamma / torch.sqrt(var + eps)).reshape(-1,1,1,1)
    fused_b = (conv_bias - mean) * (gamma / torch.sqrt(var + eps)) + beta
    return fused_W.detach().cpu().numpy().astype(np.float32), fused_b.detach().cpu().numpy().astype(np.float32)

def extract_res_layer(layer):
    blobs = []
    for block in layer:
        W1,b1 = fuse_conv_bn(block.conv1, block.bn1)
        W2,b2 = fuse_conv_bn(block.conv2, block.bn2)
        if block.downsample is not None:
            W_s,b_s = fuse_conv_bn(block.downsample[0], block.downsample[1])
        else:
            W_s,b_s = None, None
        blobs.append((W1,b1,W2,b2,W_s,b_s))
    return blobs

def extract_dec(dec_block):
    """
    Extrait la Conv2d et la BatchNorm2d d'un bloc de décodage (nn.Sequential).
    Recherche par type pour être robuste à l'ordre des couches.
    """
    conv = None
    bn = None
    for module in dec_block:
        if isinstance(module, nn.Conv2d):
            conv = module
        elif isinstance(module, nn.BatchNorm2d):
            bn = module
    if conv is None or bn is None:
        raise ValueError(
            "Le bloc de décodage ne contient pas à la fois Conv2d et BatchNorm2d.\n"
            f"Contenu : {dec_block}"
        )
    return fuse_conv_bn(conv, bn)

if __name__ == "__main__":
    pth_path = "./trained_model_simple/best_model.pth"
    pkl_path = "./trained_model_simple/model_weights.pkl"

    if not os.path.exists(pth_path):
        print(f"❌ Fichier introuvable : {pth_path}")
        print("   Lancez d'abord train.py")
        exit(1)

    checkpoint = torch.load(pth_path, map_location="cpu")
    ckpt_classes = checkpoint.get("class_names")
    if ckpt_classes != CLASS_NAMES:
        raise ValueError(
            "Incohérence d'ordre des classes entre le checkpoint et knowledge_base.py !\n"
            f"Checkpoint : {ckpt_classes}\n"
            f"Attendu   : {CLASS_NAMES}"
        )

    # Instancier le modèle avec le nombre de classes de knowledge_base (12)
    model = ResUNetSegCls(num_classes=NUM_CLASSES, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    params = {}

    # Stem
    conv1 = model.enc0[0]; bn1 = model.enc0[1]
    params["stem_W"], params["stem_b"] = fuse_conv_bn(conv1, bn1)

    # ResNet layers
    params["layer1"] = extract_res_layer(model.enc1)
    params["layer2"] = extract_res_layer(model.enc2)
    params["layer3"] = extract_res_layer(model.enc3)
    params["layer4"] = extract_res_layer(model.enc4)

    # Décodeur (utilisation de la nouvelle fonction robuste)
    params["dec4_W"], params["dec4_b"] = extract_dec(model.dec4)
    params["dec3_W"], params["dec3_b"] = extract_dec(model.dec3)
    params["dec2_W"], params["dec2_b"] = extract_dec(model.dec2)
    params["dec1_W"], params["dec1_b"] = extract_dec(model.dec1)

    # Têtes
    params["seg_W"] = model.seg_head.weight.detach().cpu().numpy().astype(np.float32)
    params["seg_b"] = model.seg_head.bias.detach().cpu().numpy().astype(np.float32)
    params["cls_W"] = model.cls_fc.weight.detach().cpu().numpy().astype(np.float32)
    params["cls_b"] = model.cls_fc.bias.detach().cpu().numpy().astype(np.float32)
    params["num_classes"] = NUM_CLASSES

    with open(pkl_path, "wb") as f:
        pickle.dump(params, f)

    print(f"✅ Export terminé : {pkl_path}")
    print(f"   Taille : {round(len(pickle.dumps(params))/1024/1024, 1)} Mo")
    print(f"   Classes (ordre global) : {CLASS_NAMES}")