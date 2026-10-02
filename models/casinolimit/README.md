# Modèles CasinoLimit

## Inclus

- `lstm_timeseries.pt` et `lstm_timeseries_scaler.joblib` — modèle LSTM +
  xNIDS (7 features Baseline, voir `scripts/lstm/README.md`). Chargé par
  `scripts/lstm/run_lstm_pipeline_casinolimit.py`.
- `realmlp_binary_baseline.joblib` et `catboost_binary_baseline.joblib`
  (7 features, Table I de l'article : `bytes`, `packets`, `duration`,
  `proto_TCP`, `proto_UDP`, `proto_ICMP`, `proto_GRE`).
- `realmlp_binary_behavioral.joblib` et `catboost_binary_behavioral.joblib`
  (17 features, Table II : Baseline + `src_event_count_last_10`,
  `src_unique_dst_last_10`, `src_packets_sum_last_10`,
  `src_bytes_sum_last_10`, `src_duration_sum_last_10`,
  `dst_unique_src_last_10`, `new_destination`, `repeated_connection`,
  `time_since_previous_src_event`, `time_since_previous_dst_event`).

Les quatre modèles RealMLP/CatBoost ont été entraînés avec les
hyperparamètres déclarés Table III de l'article :
- **RealMLP** : `RealMLP_TD_Classifier` sans surcharge d'hyperparamètre
  (configuration "TD" tuned-default, comme déclaré dans l'article).
- **CatBoost** : `iterations=500`, `learning_rate=0.05`, `depth=8`,
  `loss_function="Logloss"`.
- **Split** : groupé par équipe (`file_name`), 80/20, seed fixe (42) —
  identique au protocole déclaré Section IV-D de l'article.


