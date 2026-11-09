"""Passe avant du module de fusion sur un vrai batch issu du pipeline (bonus Partie B)."""
from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")

from torch.utils.data import DataLoader  # noqa: E402

from src.fusion.dataset import ChestXrayDataset  # noqa: E402
from src.fusion.model import ChestXrayFusionNet, FiLMGatedFusion, fusion_loss  # noqa: E402


@pytest.fixture(scope="module")
def batch(pipeline_done):
    import pandas as pd
    cfg = pipeline_done
    table = pd.read_parquet(cfg.processed("dataset.parquet"))
    ds = ChestXrayDataset(table, cfg.path("nih_images"), cfg["preprocess"]["pathologies"],
                          size=cfg["fusion"]["image_size"])
    return next(iter(DataLoader(ds, batch_size=4, shuffle=False)))


def test_batch_shapes(batch):
    assert batch["image"].shape == (4, 3, 224, 224)
    assert batch["tabular"].shape == (4, 4)
    assert batch["y"].shape == (4, 14)


def test_forward_backward_on_real_batch(batch):
    torch.manual_seed(0)
    model = ChestXrayFusionNet(pretrained=False)
    out = model(batch["image"], batch["tabular"], batch["tab_mask"])
    assert out["logits"].shape == (4, 14)
    assert out["aux_logits"].shape == (4, 14)
    assert out["gate"].shape == (4, 256)
    loss = fusion_loss(out, batch["y"], pos_weight=torch.full((14,), 5.0))
    loss.backward()
    assert torch.isfinite(loss)
    assert model.fusion.film.weight.grad is not None


def test_missing_clinical_data_uses_learned_token(batch):
    model = ChestXrayFusionNet(pretrained=False).eval()
    with torch.no_grad():
        a = model(batch["image"], batch["tabular"], torch.zeros(4))["logits"]
        b = model(batch["image"], batch["tabular"] * 0 + 99, torch.zeros(4))["logits"]
    assert torch.allclose(a, b)  # les valeurs cliniques sont bien ignorées si masquées


def test_film_starts_as_identity():
    fusion = FiLMGatedFusion(c_img=8, d_tab=4, d=6)
    fmap, t = torch.randn(2, 8, 3, 3), torch.randn(2, 4)
    gamma, beta = fusion.film(t).chunk(2, dim=-1)
    assert torch.count_nonzero(gamma) == 0 and torch.count_nonzero(beta) == 0
    z, g = fusion(fmap, t)
    assert z.shape == (2, 6) and ((g > 0) & (g < 1)).all()
