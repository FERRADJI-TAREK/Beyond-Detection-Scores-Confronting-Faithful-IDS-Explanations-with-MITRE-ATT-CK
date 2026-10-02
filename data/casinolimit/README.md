
# CasinoLimit Dataset

`final_dataset_14techniques_benign_behavioral.csv` — second
ATT&CK-labeled dataset used in the paper (14 techniques + benign
traffic, ~426k flows), alongside the UWF-Zeek dataset
(`data/processed/`).

## Schema

Main columns: `machine_name`, `proctitles`, `event_uid`, one-hot
encoding of the protocol (`proto_TCP`, `proto_UDP`, `proto_ICMP`,
`proto_GRE`), `label_technique`, `label_name`, `label_type`, plus the
same causal behavioral feature families as on Zeek (sliding window of
the last 10 events: frequency, diversity, volume, recency).

**History**: an earlier version of the models provided here (RealMLP
Baseline with 3 features and no protocol at all, CatBoost Behavioral
with 22 features including 5 undeclared "internal network" columns)
did not exactly match Tables I and II of the paper. The four
RealMLP/CatBoost models were retrained with **exactly** the declared
features:
- **Baseline (Table I, 7 features)**: `bytes`, `packets`, `duration`,
  `proto_TCP`, `proto_UDP`, `proto_ICMP`, `proto_GRE`.
- **Behavioral (Table II, +10 features, 17 total)**: Baseline +
  `src_event_count_last_10`, `src_unique_dst_last_10`,
  `src_packets_sum_last_10`, `src_bytes_sum_last_10`,
  `src_duration_sum_last_10`, `dst_unique_src_last_10`, `new_destination`,
  `repeated_connection`, `time_since_previous_src_event`,
  `time_since_previous_dst_event`. (`src_failed_conn_last_10`, the 11th
  behavioral feature of Table II, exists only for Zeek — absent from
  CasinoLimit, consistent with the "–" in Table II.)

The dataset also contains 5 "internal network" columns
(`dst_event_count_last_10`, `is_source_internal`, `is_dest_internal`,
`src_dest_internal_count_last_10`, `dst_src_internal_count_last_10`)
which are not declared in any table of the paper — they are
deliberately excluded from training so that the models exactly match
the paper's features (see `models/casinolimit/README.md` for the
measured performance).

The `scripts/casinolimit/run_*_pipeline*.py` scripts read, as a safety
measure, the columns actually expected directly from each loaded model
(rather than using a hardcoded list), in case a different model is
substituted.

## Corresponding code

The SHAP + Agent 1 + Dense RAG + Agent 2 pipeline, for this dataset, is
provided in `scripts/casinolimit/`:
- `run_realmlp_pipeline.py` — with a RealMLP model (real SHAP)
- `run_catboost_pipeline.py` — with a CatBoost model (real SHAP,
  native TreeSHAP, `tree_path_dependent`)

Both scripts separately run a Baseline configuration (7 features,
Table I) and a Behavioral configuration (17 features, Table II), each
on its own disjoint sample, with correlation-aware SHAP, Agent 1
(behavioral summary in English), Dense RAG (Top-25 MITRE candidates)
and Agent 2 (Top-5 selection).

### Adversarial validation (constant SHAP table)

- `run_realmlp_pipeline_adversarial.py`
- `run_catboost_pipeline_adversarial.py`

These two scripts replace the real SHAP computation with a **constant
SHAP table** (`FAKE_SHAP_BASELINE` / `FAKE_SHAP_BEHAVIORAL`, simulating
exfiltration/C2), injected identically for every event regardless of
its actual content. Unlike the Zeek pipeline's adversarial validation
(`--permute-shap`, which permutes the ranking/scores/signs of a real
SHAP), here the explanation is entirely fabricated and fixed: the test
measures whether the LLM agents (Agent 1 then Agent 2) blindly follow
this false explanation rather than the actual content of the observed
event — a more aggressive robustness test.

### LSTM + xNIDS

- `scripts/lstm/train_lstm_casinolimit.py` — LSTM training (sequences
  of 10 flows, explicit padding masking).
- `scripts/lstm/run_lstm_pipeline_casinolimit.py` — xNIDS (Sparse Group
  Lasso, self-contained) + Agent 1 + Dense RAG + Agent 2.

See `scripts/lstm/README.md` for details (including the architectural
choice that made it possible to obtain a complete pipeline from the
provided material).

## What is still missing from this archive

The **trained models** (`realmlp_binary_baseline.joblib`,
`realmlp_binary_behavioral.joblib`, `catboost_binary_baseline.joblib`,
`catboost_binary_behavioral.joblib`) are **all included** in
`models/casinolimit/`, already trained and ready to load (see
`models/casinolimit/README.md` for the measured performance and
hyperparameter details). The script that produced them is not
included.
