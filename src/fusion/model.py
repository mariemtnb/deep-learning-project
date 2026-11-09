"""Prototype du module de fusion retenu (Partie B) : FiLM clinique + porte + tête multi-label.

Idée : la radiographie porte l'essentiel de l'information, les 4 variables cliniques
(âge, sexe, vue AP/PA, numéro de suivi) servent surtout à « régler » la lecture de
l'image. Exemple concret : sur un cliché AP le cœur paraît plus gros, donc la même
silhouette cardiaque ne signifie pas la même chose qu'en PA. FiLM permet exactement ça :
le vecteur clinique produit, pour chaque canal visuel, un gain γ et un biais β.

    F  = E_img(x)                     carte de caractéristiques  (B, C, H, W)
    t  = E_tab(c)  ou  t_absent        vecteur clinique           (B, d_t)
    γ, β = W_γ t + b_γ ,  W_β t + b_β                              (B, C)
    F' = (1 + γ) ⊙ F + β               modulation canal par canal
    v  = GAP(F')                       (B, C)
    g  = σ(W_g [v ; t])                porte apprise (B, d) : part de la clinique
    z  = P_v v + g ⊙ P_t t             représentation fusionnée   (B, d)
    ŷ  = W_h z                         logits des 14 pathologies

Une tête auxiliaire image seule (ŷ_img = W_a GAP(F)) sert de perte d'appoint pour que
le réseau n'apprenne pas à prédire depuis la vue AP seule (raccourci connu sur NIH).
"""
from __future__ import annotations

import torch
from torch import nn

N_TAB = 4  # âge normalisé, sexe (M=1), vue (AP=1), log(1 + numéro de suivi)


def encode_tabular(age, is_male, is_ap, followup) -> torch.Tensor:
    """Même encodage que src/analysis/modalities.py, pour rester cohérent."""
    age = (torch.as_tensor(age, dtype=torch.float32) - 47.0) / 17.0
    return torch.stack([
        age,
        torch.as_tensor(is_male, dtype=torch.float32),
        torch.as_tensor(is_ap, dtype=torch.float32),
        torch.log1p(torch.as_tensor(followup, dtype=torch.float32)),
    ], dim=-1)


class ImageEncoder(nn.Module):
    """DenseNet-121 (ImageNet) sans sa tête : sortie (B, 1024, 7, 7) pour une entrée 224."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        from torchvision.models import DenseNet121_Weights, densenet121
        weights = DenseNet121_Weights.IMAGENET1K_V1 if pretrained else None
        net = densenet121(weights=weights)
        self.features = net.features
        self.out_channels = net.classifier.in_features  # 1024

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.features(x))


class TabularEncoder(nn.Module):
    def __init__(self, n_in: int = N_TAB, d: int = 64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(n_in, d), nn.ReLU(), nn.Linear(d, d), nn.ReLU())

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        return self.net(c)


class FiLMGatedFusion(nn.Module):
    def __init__(self, c_img: int, d_tab: int, d: int):
        super().__init__()
        self.film = nn.Linear(d_tab, 2 * c_img)
        # Initialisation à zéro : au départ γ = β = 0, donc F' = F. Le modèle démarre
        # comme un modèle image seule et n'apprend la modulation que si elle aide.
        nn.init.zeros_(self.film.weight)
        nn.init.zeros_(self.film.bias)
        self.proj_v = nn.Linear(c_img, d)
        self.proj_t = nn.Linear(d_tab, d)
        self.gate = nn.Linear(c_img + d_tab, d)

    def forward(self, fmap: torch.Tensor, t: torch.Tensor):
        gamma, beta = self.film(t).chunk(2, dim=-1)
        fmap = (1 + gamma[..., None, None]) * fmap + beta[..., None, None]
        v = fmap.mean(dim=(2, 3))
        g = torch.sigmoid(self.gate(torch.cat([v, t], dim=-1)))
        z = self.proj_v(v) + g * self.proj_t(t)
        return z, g


class ChestXrayFusionNet(nn.Module):
    def __init__(self, n_classes: int = 14, d_tab: int = 64, d: int = 256,
                 modality_dropout: float = 0.2, pretrained: bool = True,
                 image_encoder: nn.Module | None = None):
        super().__init__()
        self.img_enc = image_encoder or ImageEncoder(pretrained)
        c = self.img_enc.out_channels
        self.tab_enc = TabularEncoder(N_TAB, d_tab)
        self.absent = nn.Parameter(torch.zeros(d_tab))  # jeton « clinique absente » appris
        self.fusion = FiLMGatedFusion(c, d_tab, d)
        self.head = nn.Sequential(nn.Dropout(0.2), nn.Linear(d, n_classes))
        self.aux_head = nn.Linear(c, n_classes)
        self.p_drop = modality_dropout

    def forward(self, x: torch.Tensor, c: torch.Tensor, tab_mask: torch.Tensor | None = None):
        """x : (B,3,H,W) ; c : (B,4) ; tab_mask : (B,) 1 = clinique présente."""
        b = x.shape[0]
        if tab_mask is None:
            tab_mask = torch.ones(b, device=x.device)
        if self.training and self.p_drop > 0:
            # Modality dropout : on cache la clinique pour une partie du batch.
            keep = (torch.rand(b, device=x.device) >= self.p_drop).float()
            tab_mask = tab_mask * keep
        fmap = self.img_enc(x)
        t = self.tab_enc(c)
        m = tab_mask[:, None]
        t = m * t + (1 - m) * self.absent.expand(b, -1)
        z, gate = self.fusion(fmap, t)
        return {"logits": self.head(z), "aux_logits": self.aux_head(fmap.mean(dim=(2, 3))),
                "gate": gate}


def fusion_loss(out: dict, y: torch.Tensor, pos_weight: torch.Tensor | None = None,
                aux_weight: float = 0.3) -> torch.Tensor:
    """L = BCE pondérée(ŷ, y) + λ · BCE pondérée(ŷ_img, y)."""
    bce = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    return bce(out["logits"], y) + aux_weight * bce(out["aux_logits"], y)
