# Trained Models

This folder contains the already-trained **RealMLP** models (`pytabkit`
framework), saved in `joblib` format. Unlike the raw data, they **are
versioned in git** (weights < 1 MB each), since they are needed to replay
SHAP and the multi-agent pipeline without having to retrain.

| File                                         | Description                                                            |
|----------------------------------------------|------------------------------------------------------------------------|
| `realmlp_zeek_binaire.joblib`                | First version of the binary classifier (Benign / Attack), 53 features. |
| `realmlp_zeek_binaire_behavioral.joblib`     | **Reference model** used by the SHAP + Agents pipeline (59 "Behavioral" features, see `EXPECTED_FEATURES`). Trained with exactly the 59 features from Tables I+II of the paper (48 Baseline + 11 Behavioral). |
| `realmlp_zeek_binaire_baseline48.joblib`     | "Baseline" variant (48 features = the 59 `EXPECTED_FEATURES` minus the 11 behavioral features `*_last_10`/sliding window from Table II). Directly loadable via `--model` (see below); use `--fake-shap-variant baseline` if combined with `--fake-shap-constant`. |
| `catboost_zeek_baseline.joblib`              | **CatBoost** Zeek model, 48 "Baseline" features. Loaded by `scripts/06_run_catboost_shap_llm_pipeline.py` (native TreeSHAP + Agent 1 + Dense RAG + Agent 2). |
| `catboost_zeek_behavioral.joblib`            | **CatBoost** Zeek "Behavioral" variant (59 features, same `EXPECTED_FEATURES` as the reference RealMLP). Trained for the same reason as `realmlp_zeek_binaire_behavioral.joblib` above. Directly loadable via `--model` (no code change — `feature_names_` is read from the model). |

**Verification of the two retrained Behavioral models**: learned columns
read directly from each model (`feature_names_` for CatBoost,
`x_converter_.fitted_columns` for RealMLP) — 59/59, identical to
`EXPECTED_FEATURES`, .

Load a variant with the existing RealMLP pipeline, or the CatBoost
Behavioral variant with the CatBoost pipeline (no code change required
in either case — the feature columns are read from the model itself):

```bash
python scripts/05_run_shap_llm_pipeline.py --model models/realmlp_zeek_binaire_baseline48.joblib
python scripts/06_run_catboost_shap_llm_pipeline.py --model models/catboost_zeek_behavioral.joblib
```





## LSTM Model (Zeek)

`lstm_binary_zeek.pt` — bidirectional LSTM model trained on the 48
Baseline features declared in Table I of the paper (see
`scripts/lstm/zeek_data.py` and `scripts/lstm/README.md`), **included**
in this archive. Loaded by `scripts/lstm/run_lstm_pipeline_zeek.py`
(xNIDS + Agent 1 + Dense RAG + Agent 2).