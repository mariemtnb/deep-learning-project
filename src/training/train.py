"""Emplacement réservé à l'entraînement (Tâche 4).

Ce qui est déjà décidé et sera branché ici :
- données : processed/dataset.parquet + ChestXrayDataset (src/fusion/dataset.py) ;
- modèle  : ChestXrayFusionNet (src/fusion/model.py) et les baselines du rapport ;
- suivi   : MLflow (section `tracking` de configs/params.yaml), un run par configuration,
            avec les paramètres, le hash DVC des données et l'AUC par pathologie ;
- étape DVC `train` (commentée dans dvc.yaml) qui dépendra de `split` et `validate`.
"""
from __future__ import annotations

from src.common.config import Config


def start_tracking(cfg: Config):
    import mlflow
    t = cfg["tracking"]
    mlflow.set_tracking_uri(t["uri"])
    mlflow.set_experiment(t["experiment"])
    return mlflow


def run(cfg: Config) -> None:  # pragma: no cover
    raise NotImplementedError("Entraînement prévu en Tâche 4.")
