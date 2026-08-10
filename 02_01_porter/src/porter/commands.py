"""Which of the five commands this model supports — the ``probe`` DAG.

The ``COMMANDS`` registry maps each supported
:class:`~meds_model_base.commands.base.CommandName` to the class implementing it. ``meds-model`` consults
it: an unsupported command exits with a message naming what *is* supported.

This is the one piece of the DAG that has to be Python rather than configuration — the dispatcher needs it
before Hydra composes anything. Edit it freely: swap in your own subclass, add a command, or drop one.

**One command is subclassed here, and the reason is a correctness bug, not a preference.** See
:class:`OrderStableInferCommand`.

Keep module-level imports light: the dispatcher imports this file to answer ``meds-model commands`` and
``--help``, and a tier-1 smoke test depends on that staying cheap. Everything heavy (torch, lightning,
polars) is imported inside the method that needs it.
"""

from meds_model_base.commands import (
    CommandName,
    DefaultInferCommand,
    DefaultPreprocessDataCommand,
    DefaultPretrainCommand,
    ProbePredictCommand,
    ProbeTrainCommand,
)

#: Probability above which `predicted_boolean_value` is emitted as a hard label. ``null`` — the default
#: — emits the column as all-null instead, which is what PORTER wants; see
#: :class:`ScorablePredictCommand`. Overridable as ``predict.decision_threshold=0.5``.
DEFAULT_DECISION_THRESHOLD = None


def ordered_split_loader(cfg, split: str):
    """Return ``(dataset, dataloader)`` for ``split`` whose iteration order is ``schema_df`` order.

    ``meds_model_base.commands._runtime.split_dataset_and_loader`` takes the loader off the
    meds-torch-data ``Datamodule``, whose ``train_dataloader()`` is **hardcoded to ``shuffle=True``**
    (meds-torch-data's own class docstring: "The train dataloader shuffles so doesn't return stable
    outputs, but the others do not shuffle"). Lightning warns about a shuffling ``predict_dataloader``
    but does not turn it off.

    Building the loader from the *dataset* instead, with ``shuffle=False``, is the whole fix.
    """
    from meds_model_base.commands._runtime import SPLIT_ATTRS
    from meds_model_base.lightning import build_datamodule

    dataset = getattr(build_datamodule(cfg), SPLIT_ATTRS[split][0])
    loader = dataset.get_dataloader(
        batch_size=int(cfg.batch_size),
        num_workers=int(cfg.get("num_workers", 0) or 0),
        shuffle=False,
    )
    return dataset, loader


class OrderStableInferCommand(DefaultInferCommand):
    """``infer``, with every split's outputs actually aligned to the keys they are written against.

    **The bug this exists for.** ``_runtime.run_predict_step`` pairs ``dataset.schema_df`` (index order)
    with the outputs of ``trainer.predict(module, dataloaders=<loader>)`` (iteration order), on the stated
    assumption that the two agree — ``SPLIT_ATTRS`` is commented "Only the non-shuffling loaders are
    listed: prediction order must match ``schema_df`` order." For ``tuning`` and ``held_out`` that holds.
    For ``train`` it does not: ``Datamodule.train_dataloader()`` shuffles, so every training-split row of
    ``artifacts.parquet`` receives **another subject's** embedding, and a different one on every run.

    It is silent: the artifact still has the right keys, the right row count, the right width and a valid
    manifest. Measured on this repository's synthetic signal fixture (200 train / 40 tuning / 40 held-out
    subjects, ``SIGNAL//POS`` presence as the label), 5-fold cross-validated AUROC of a standardized
    logistic regression fitted on the materialized embeddings of each split:

    ============  =============================  ==========
    split         ``DefaultInferCommand``        this class
    ============  =============================  ==========
    ``train``     0.49 (chance)                  1.00
    ``tuning``    1.00                           1.00
    ``held_out``  1.00                           1.00
    ============  =============================  ==========

    The probe trains on ``train`` and validates on ``tuning``, so the effect is a probe fitted on
    scrambled features and rescued only by checkpoint selection against the clean validation split — which
    is why the learnability tier passed, and passed with a different score on every run.

    It is invisible to the conformance suite because ``test_end_to_end`` predicts ``splits=[held_out]``.

    This is a defect in the vendored contract, which ``copier update`` overwrites, so it is deliberately
    **not** patched there (see ``CLAUDE.md``). Filed upstream as
    `MEDS_model_template#21 <https://github.com/florian6973/MEDS_model_template/issues/21>`_. Delete this
    class and restore ``DefaultInferCommand`` once the contract builds a non-shuffling loader for every
    split.
    """

    def infer(self, cfg, module, index):
        import lightning.pytorch as L
        import polars as pl
        from meds_model_base.commands._runtime import KEYS, stack_outputs

        class _InferStepAdapter(L.LightningModule):
            """Present ``infer_step`` as ``predict_step``; Lightning's predict loop calls only the latter."""

            def __init__(self, inner):
                super().__init__()
                self.inner = inner

            def predict_step(self, batch, batch_idx: int = 0):
                return self.inner.infer_step(batch)

        driver = _InferStepAdapter(module)
        driver.eval()

        frames = []
        for split in index["split"].unique(maintain_order=True).to_list():
            dataset, dataloader = ordered_split_loader(cfg, split)
            keys = dataset.schema_df.select(KEYS)
            if not len(keys) or len(dataset) == 0:
                continue
            trainer = L.Trainer(
                accelerator="auto",
                devices=1,
                logger=False,
                enable_checkpointing=False,
                enable_progress_bar=False,
            )
            outputs = trainer.predict(driver, dataloaders=dataloader)
            frames.append(keys.with_columns(stack_outputs(outputs)))

        if not frames:
            raise RuntimeError(
                "The datamodule produced no rows for any requested split. Check that external_labels_dir "
                "and the tensorized cohort refer to the same subjects."
            )
        return pl.concat(frames, how="vertical_relaxed").unique(subset=KEYS, maintain_order=True)


class ScorablePredictCommand(ProbePredictCommand):
    """``predict``, plus an all-null ``predicted_boolean_value`` column so the output can be scored.

    A probe emits a probability, and that is all PORTER needs: §2.7.1 reports AUROC, which is
    threshold-free. `meds_evaluation.schema.PredictionSchema` agrees, and not merely by convention — both
    prediction fields are declared ``Optional`` and ``_validate_schema`` exists solely to enforce "at
    least one of the two must be present".

    `evaluate_binary_classification` nonetheless fetches the hard label unconditionally
    (`evaluate.py:54`), so a probability-only file passes validation and then crashes the scorer with
    `ColumnNotFoundError`. Reported upstream as
    `meds-evaluation#35 <https://github.com/kamilest/meds-evaluation/issues/35>`_.

    **The column is all-null, not thresholded.** `PredictionSchema.align` keeps a *declared* all-null
    column while dropping an *absent* one, and the scorer already handles the former::

        if predicted_values.is_null().all():
            predicted_values = None

    So an all-null column reaches exactly the branch the library was written for. Measured, on the same
    predictions:

    ==========================  =====================================================================
    ``predicted_boolean_value``  metrics produced
    ==========================  =====================================================================
    all-null (default)           ``roc_auc_score``, ``average_precision_score``, ``brier_score``,
                                 ``calibration_error``
    thresholded at 0.5           the same four, plus ``binary_accuracy`` and ``f1_score``
    ==========================  =====================================================================

    ``roc_auc_score`` is bit-identical between the two. The only difference is that thresholding *adds*
    accuracy and F1 — computed against a cutoff the model never chose. On an imbalanced cohort (the
    fluid-output task is 14.5% positive) those two numbers are actively misleading, and a reader has no
    way to tell they came from an arbitrary 0.5. Omitting them is more honest than inventing them.

    Set ``predict.decision_threshold=0.5`` to opt into the hard label when you genuinely want it — for
    instance with a cutoff chosen on the tuning split.
    """

    def predict(self, cfg, source, index):
        import polars as pl

        frame = super().predict(cfg, source, index)
        if "predicted_boolean_value" in frame.columns:
            return frame

        threshold = cfg.get("decision_threshold", DEFAULT_DECISION_THRESHOLD)
        if threshold is None:
            # Declared but null: kept by `align`, and the scorer turns it into `None` itself.
            label = pl.lit(None, dtype=pl.Boolean)
        else:
            label = pl.col("predicted_boolean_probability") > float(threshold)
        return frame.with_columns(label.alias("predicted_boolean_value"))


COMMANDS = {
    CommandName.preprocess_data: DefaultPreprocessDataCommand,
    CommandName.pretrain: DefaultPretrainCommand,
    CommandName.infer: OrderStableInferCommand,
    CommandName.supervised_train: ProbeTrainCommand,
    CommandName.predict: ScorablePredictCommand,
}
