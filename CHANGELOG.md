# Changes

## 0.1.0

- RETAIN (Choi et al., NeurIPS 2016) for MEDS behind the MEDS algorithm contract, `supervised` profile:
  `preprocess_data`, `train_supervised`, `predict_supervised`.
- Converted from the earlier `meds-model-retain` layout to `MEDS_algorithm_template` @ `731e759` with
  `meds-algorithm==0.4.0a5`. The model, its defaults and its scores are unchanged.
- Validation: model fidelity against a NumPy transcription of the official equations, data-path and
  temporal-cutoff tests, a signal test with a shuffled-label control, the shared conformance suite, and the
  temporal-leakage checks of `meds-algorithm` main (future-rewrite rerun and null world).
