# Diagnostic assisté sur radiographie thoracique : pipeline DataOps et fusion multimodale

Module Deep Learning, projet en binôme, Tâche 3. Entrée multimodale : une radiographie
thoracique (NIH ChestX-ray14) et les données cliniques associées (âge, sexe, vue AP/PA,
numéro de suivi). Sortie visée : 14 pathologies, multi-label.

Ce dépôt contient :

- un pipeline de données automatisé (`dvc repro`), configuré par `configs/params.yaml`,
  avec des contrôles qualité qui l'arrêtent en cas de problème ;
- des tests (unitaires, données, prototype de fusion) et une CI GitHub Actions qui
  exécute tout sur un échantillon ;
- le rapport de la tâche (`report/rapport_tache3.pdf`) avec l'étude des méthodes de fusion
  et l'architecture retenue, dont un prototype PyTorch est dans `src/fusion/`.

## Installation

Python 3.11 à 3.13.

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows ; sous Linux/macOS : source .venv/bin/activate
pip install -r requirements.txt
# optionnel, pour le prototype de fusion :
pip install -r requirements-dl.txt --extra-index-url https://download.pytorch.org/whl/cpu
```

## Essayer tout de suite (sans télécharger les 45 Go)

Un échantillon de 40 patients (112 radiographies réduites à 128×128) est versionné dans
`data/sample/`. C'est lui qu'utilise la CI.

```bash
python -m src.cli all --config configs/sample.yaml   # pipeline complet, ~3 s
pytest                                               # 17 tests
python scripts/demo_quality_failure.py --fault leak  # échec contrôlé d'un test qualité
```

Résultats dans `build/sample/` (rapport qualité HTML, statistiques, logs JSON).

## Récupérer les données

Trois niveaux, selon ce dont vous avez besoin.

**1. Données préparées (DVC).** Les métadonnées brutes et les tables produites sont
versionnées avec DVC. Le stockage distant `shared` est un dossier partagé (dossier Google
Drive synchronisé par Drive pour ordinateur). Une fois l'accès au dossier obtenu :

```bash
dvc remote modify --local shared url "G:/Mon Drive/dvc-deep-learning"   # votre chemin
dvc pull
```

`--local` écrit dans `.dvc/config.local`, qui n'est pas versionné : chacun garde son chemin.

**2. Images (Kaggle).** Les images ne sont pas dans DVC (45 Go, immuables, déjà
hébergées). Téléchargez depuis Kaggle :

- NIH : <https://www.kaggle.com/datasets/nih-chest-xrays/data>
- Pneumonia : <https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia>

Décompressez dans `project deep learning datasets/` à la racine du dépôt (ou changez
`paths.nih_images` et `paths.pneumonia_images` dans `configs/params.yaml`) :

```
project deep learning datasets/
├── NIH Chest X-rays/images_001 ... images_012/images/*.png
└── Chest X-Ray Images (Pneumonia)/chest_xray/chest_xray/{train,val,test}/{NORMAL,PNEUMONIA}
```

**3. Métadonnées seules.** Si vous n'avez pas accès au stockage DVC, copiez
`Data_Entry_2017.csv`, `BBox_List_2017.csv`, `test_list.txt` et `train_val_list.txt`
du téléchargement NIH dans `data/raw/nih/`.

## Exécuter le pipeline

```bash
dvc repro          # reconstruit ce qui a changé, dans l'ordre des dépendances
dvc dag            # graphe des étapes
dvc metrics show   # analyse des modalités et dérive
```

Sans DVC, la même chose étape par étape :

```bash
python -m src.cli all                   # tout
python -m src.cli validate              # une seule étape
```

> Sous Windows avec Python 3.13, si `dvc` plante sans message (code 139), utilisez
> `python scripts/dvc_safe.py repro` : c'est un contournement d'un plantage de `logging`
> propre à certaines installations de Python, pas un problème du projet.

| Étape | Rôle | Sortie principale |
|---|---|---|
| `ingest` | lecture des CSV NIH, renommage des colonnes | `data/interim/metadata_raw.parquet` |
| `index_images` | inventaire des 112 120 images, contrôle taille/mode sur 300 | `data/interim/image_index.parquet` |
| `validate_raw` | schémas pandera, manquants, doublons, âges, couverture image ↔ CSV | `reports/validation_raw.json` |
| `preprocess` | réparation des âges, labels multi-hot, variables dérivées | `data/interim/metadata_clean.parquet` |
| `align` | appariement image ↔ clinique, masque de modalités | `data/interim/aligned.parquet` |
| `split` | train/val/test par patient | `data/processed/dataset.parquet` |
| `validate` | fuite de patients, cohérence des labels et des modalités | `reports/validation_processed.json` |
| `analyze` | apport de la clinique seule, prévalence par vue | `reports/modality_analysis.json` |
| `report` | statistiques et rapport qualité HTML | `outputs/summary.json`, `reports/quality_report.html` |
| `examples`, `dashboard` | tableau de bord du projet | `outputs/dashboard.html` |
| `drift` | dérive par rapport à la version de référence (PSI) | `reports/drift.json` |

Chaque étape écrit aussi `reports/stages/<etape>.json` (volumes, durée, empreintes MD5
des entrées et sorties, commit) et ajoute ses événements à `logs/pipeline.jsonl`.

## Arborescence

```
configs/            params.yaml (complet), sample.yaml (CI)
data/raw/nih.dvc    métadonnées NIH suivies par DVC
data/sample/        échantillon versionné dans Git
src/
  ingestion/        une source par module (métadonnées, images)
  validation/       schémas pandera et contrôles qualité
  preprocessing/    nettoyage, variables
  alignment/        appariement image ↔ clinique
  splitting/        découpage par patient
  analysis/         mesures pour le choix de fusion
  reporting/        statistiques, rapport qualité, tableau de bord
  monitoring/       dérive entre versions
  fusion/           prototype PyTorch du modèle de fusion
  training/         réservé à la tâche 4 (MLflow)
tests/              tests unitaires, de données et du prototype
docs/DATA_CARD.md   fiche de données
report/             rapport PDF et sa source
app/                interface web (Flask + React)
dvc.yaml, dvc.lock  définition et état du pipeline
```

## Interface web

```bash
python app/run.py      # http://localhost:8000
```

Elle lit `data/processed/dataset.parquet` (après `dvc pull` ou `dvc repro`) et
`outputs/dashboard.html`. Pour modifier le front : `npm install` puis `npm run build`
dans `app/web`.

## Docker

```bash
docker build -t chest-xray-pipeline .
docker run --rm chest-xray-pipeline     # pipeline sur l'échantillon
```

## Démonstration (5 à 10 min)

1. `dvc repro` : tout est à jour, rien n'est relancé. Modifier `split.val_fraction`
   dans `configs/params.yaml`, relancer : seules `split` et les étapes en aval tournent.
2. `python scripts/demo_quality_failure.py --fault leak` : un patient se retrouve dans
   deux ensembles, l'étape `validate` échoue et le pipeline s'arrête (code 1). Variantes :
   `--fault missing-image`, `--fault bad-values`.
3. Ouvrir `reports/quality_report.html`, puis présenter l'architecture (rapport, section 5)
   et `pytest tests/test_fusion.py -v` (passe avant sur un vrai batch).
