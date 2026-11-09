# Fiche de données (data card)

Projet : diagnostic assisté sur radiographie thoracique, entrée multimodale image + données cliniques.
Version des données décrite : `data/raw/nih.dvc` et `dvc.lock` du tag `task3`.

## 1. Sources

| Source | Contenu | Accès / licence | Rôle |
|---|---|---|---|
| NIH ChestX-ray14 (Wang et al., 2017), miroir Kaggle `nih-chest-xrays/data` | 112 120 radiographies PNG 1024×1024, 30 805 patients, 14 pathologies extraites des comptes rendus, âge, sexe, vue (PA/AP), numéro de suivi, 984 boîtes englobantes sur 880 images | Public, dé-identifié par le NIH Clinical Center. Licence CC0 sur Kaggle ; le NIH demande de citer l'article et de mentionner le NIH Clinical Center. | Jeu principal (image + tabulaire) |
| Chest X-Ray Images (Pneumonia), Kermany et al. (2018), Kaggle `paultimothymooney/chest-xray-pneumonia` | 5 856 radiographies pédiatriques JPEG, 2 classes (NORMAL / PNEUMONIA) | Public, CC BY 4.0 | Prototype binaire, contrôle externe. Pas de données cliniques : non utilisé pour la fusion. |

## 2. Volumes (exécution complète du pipeline)

| | Images | Patients | Images normales |
|---|---|---|---|
| train | 77 724 | 25 208 | 58,6 % |
| val | 8 800 | 2 800 | 56,6 % |
| test (officiel NIH) | 25 596 | 2 797 | 38,5 % |

- 3,64 radiographies par patient en moyenne, médiane 1, maximum 184.
- Images sur disque : 112 120 / 112 120 retrouvées, 0 orpheline. 45,06 Go.
- Jeu pédiatrique : train 1 341 normales / 3 875 pneumonies, val 8 / 8, test 234 / 390.

## 3. Variables (table `data/processed/dataset.parquet`)

| Colonne | Type | Description |
|---|---|---|
| `image` | str | Nom du fichier, clé de jointure avec l'image |
| `patient_id` | int | Identifiant patient (clé de découpage) |
| `age_clean` | int | Âge après réparation des valeurs impossibles |
| `age_invalid` | bool | Âge d'origine hors [1, 100] |
| `gender`, `view` | cat | M/F ; PA/AP |
| `followup` | int | Rang de la radiographie dans l'historique du patient |
| `width`, `height` | int | Dimensions d'origine avant réduction à 1024×1024 |
| `finding_labels` | str | Labels bruts séparés par `|` |
| `lbl_*` (14) | 0/1 | Encodage multi-label |
| `is_normal`, `n_findings`, `is_multilabel` | int | Dérivées des labels |
| `split` | cat | train / val / test, attribué par patient |
| `image_folder`, `has_image`, `has_tabular`, `has_bbox` | | Alignement et masque de modalités |

## 4. Prétraitements appliqués

1. En-tête CSV réparé (colonnes coupées par une virgule, colonne vide finale), renommage par position.
2. 16 âges impossibles (> 100 ans, ex. 148, 411) remplacés par la médiane des âges valides (49 ans) ; l'indicateur `age_invalid` est conservé.
3. Sexe et vue normalisés (majuscules, espaces).
4. Labels éclatés en 14 colonnes binaires.
5. Appariement image ↔ ligne clinique par nom de fichier ; masque `has_image` / `has_tabular`.
6. Test officiel conservé ; validation = 10 % des patients du train_val, tirés avec la graine 42.

Les images ne sont pas modifiées par le pipeline de cette tâche. Le redimensionnement 224×224 et la normalisation ImageNet sont faits à la volée par `src/fusion/dataset.py`.

## 5. Biais et limites connus

- **Labels bruités.** Les 14 labels sont extraits par NLP des comptes rendus ; les auteurs estiment une précision supérieure à 90 %, et des relectures indépendantes (Oakden-Rayner, 2020) trouvent des erreurs bien plus fréquentes sur certaines classes. Un plafond de performance est donc attendu.
- **Test plus « malade » que train.** Le test officiel contient des patients plus suivis (20 radios de suivi en moyenne contre 5 en train), plus de clichés AP (56,6 % contre 34,8 %) et plus de trois fois plus de pneumothorax (10,4 % contre 3,0 %). Le contrôle d'écart de prévalence val/test le signale (7,4 points sur Pneumothorax). Les scores de validation seront optimistes par rapport au test.
- **Raccourcis possibles.** La vue AP (patient alité) et le numéro de suivi sont corrélés à la gravité : Edema est 11 fois plus fréquent en AP qu'en PA. Un modèle peut apprendre la vue au lieu de la pathologie.
- **Déséquilibre.** Hernia : 0,2 % des images (227) ; Infiltration : 17,7 %.
- **Population.** Un seul hôpital (NIH Clinical Center, États-Unis), 56 % d'hommes, très peu d'enfants. Le jeu pédiatrique ne compense pas : autre hôpital, autre format, autre tâche.
- **2 images RGBA** sur 300 contrôlées : à convertir en niveaux de gris (fait dans le Dataset).

## 6. Usage prévu et usages à éviter

Prévu : enseignement, comparaison de stratégies de fusion. À éviter : toute décision clinique, ou l'affirmation qu'un score sur ce jeu se transpose à un autre hôpital sans validation externe.
