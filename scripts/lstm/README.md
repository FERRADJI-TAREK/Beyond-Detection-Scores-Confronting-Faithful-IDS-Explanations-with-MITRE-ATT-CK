# Pipeline LSTM + xNIDS (CasinoLimit et Zeek)

Ce dossier ajoute un **troisième modèle** (`LSTM`) et une **troisième
méthode d'explicabilité** (`xNIDS`, approximation locale par régression
Sparse Group Lasso) aux deux déjà couverts par `scripts/casinolimit/`
et `scripts/0*_*.py` (`RealMLP`/`CatBoost` + `SHAP`/`TreeSHAP`) — les
trois modèles et les trois méthodes XAI figurant dans la présentation
de l'article ("Model: RealMLP, CatBoost, LSTM" / "XAI: SHAP, TreeSHAP,
xNIDS") sont ainsi tous couverts par du code exécutable, pour les deux
datasets (UWF-Zeek et CasinoLimit).

## Contenu

| Fichier                              | Rôle                                                                 |
|---------------------------------------|------------------------------------------------------------------------|
| `lstm_common.py`                      | Architecture `LSTMBinary`, utilitaires d'entraînement/évaluation, `XNIDSExplainer` (auto-contenu) — partagés par les deux datasets |
| `casinolimit_data.py`                 | Features/séquences CasinoLimit (fenêtre de 10 flux, padding à droite + longueur réelle) |
| `zeek_data.py`                        | Features/séquences Zeek (encodage one-hot dynamique, split par IP + technique) |
| `train_lstm_casinolimit.py`           | Entraînement LSTM CasinoLimit → `models/casinolimit/lstm_timeseries.pt` |
| `train_lstm_zeek.py`                  | Entraînement LSTM Zeek → `models/lstm_binary_zeek.pt`                 |
| `run_lstm_pipeline_casinolimit.py`    | xNIDS → Agent 1 → RAG dense MITRE → Agent 2, sur CasinoLimit          |
| `run_lstm_pipeline_zeek.py`           | xNIDS → Agent 1 → RAG dense MITRE → Agent 2, sur Zeek                  |

## Usage

Les modèles entraînés sont **inclus** dans cette archive
(`models/casinolimit/lstm_timeseries.pt`, `models/lstm_binary_zeek.pt`)
— l'entraînement n'est donc nécessaire que pour reproduire ou améliorer
ces poids (voir "Résultats" ci-dessous).

```bash
# 1. (Optionnel) Ré-entraînement
python scripts/lstm/train_lstm_casinolimit.py
python scripts/lstm/train_lstm_zeek.py

# 2. Pipeline xNIDS + Agents (nécessite un LLM Ollama accessible, voir
#    README.md racine, section Installation)
python scripts/lstm/run_lstm_pipeline_casinolimit.py --n-attacks 5
python scripts/lstm/run_lstm_pipeline_zeek.py --n-per-technique 5
```

