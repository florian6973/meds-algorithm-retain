"""RETAIN's configuration surface.

Every architecture and optimisation field mirrors an option of the official `retain.py` command line
(`mp2893/retain` @ 9fd39c46), with the same default, so a run can be described by naming the upstream flags
it changed. The first block has no upstream counterpart: it is the MEDS adaptation, because the official
code starts from an already-tokenised list-of-list-of-list of visits and never has to decide what a visit
is. Those fields are marked `MEDS` below and are recorded as *adapted* in the implementation report.

Core `NNConfig` is not reused: RETAIN has two dropout rates (`keep_prob_emb`, `keep_prob_context`), four
explicit L2 penalties instead of one weight decay, and a default optimiser (Adadelta) without a learning
rate to set, so `capacity`, `learning_rate`, `weight_decay` and `dropout` would each be a field with no
effect.
"""

from dataclasses import dataclass

from meds_algorithm import AlgorithmConfig

SOLVERS = ("adadelta", "adam")


@dataclass(frozen=True)
class RETAINConfig(AlgorithmConfig):
    # -- MEDS adaptation: how a MEDS code stream becomes RETAIN's visit sequences -------------------
    #: Width of the absolute time grid that groups events into one visit. 1440 = calendar day, the
    #: closest generic analogue of the admission-level visits of `process_mimic.py`. 0 keeps every
    #: distinct timestamp as its own visit.
    visit_bucket_minutes: int = 1440
    #: Whether an event stamped exactly at the prediction time belongs to the history. MEDS-DEV tasks
    #: declare their input window with `index_timestamp: end` and `end_inclusive: True`, so the
    #: prediction time *is* the last instant of the granted input window; excluding it would discard
    #: data the task definition hands to the model. Set false for a task whose label-defining event can
    #: coincide with its own prediction time.
    include_events_at_prediction_time: bool = True
    #: Discard events older than this many days before the prediction time. `None` uses all history,
    #: which is what `start: null` in the demo task's input window asks for; the paper used an 18-month
    #: window for its own heart-failure cohort.
    lookback_days: float | None = None
    #: Keep at most this many of the most recent visits. `None` keeps all of them.
    max_visits: int | None = None
    #: Vocabulary filters. The vocabulary is learned from the training split only.
    min_code_count: int = 1
    max_vocab_size: int | None = None
    #: Drop the MEDS standard's two non-clinical subject markers, `MEDS_BIRTH` and `MEDS_DEATH`, from the
    #: code stream. Both the paper (section 4.1: visits are "diagnosis, medication and procedure codes")
    #: and `process_mimic.py` (sequences built from `DIAGNOSES_ICD` alone, `PATIENTS.csv` read only to
    #: derive the label) exclude demographics and outcome markers from the visit sequence. Keeping
    #: `MEDS_BIRTH` makes a birth record the first visit of essentially every sequence, decades before the
    #: rest, where it consumes one of RETAIN's visit-attention slots; keeping `MEDS_DEATH` puts the
    #: canonical outcome marker in the input space of any task whose window can reach it. These are
    #: MEDS-standard codes, not dataset predicates, so excluding them keeps this port predicate-free.
    exclude_meds_standard_markers: bool = True

    # -- Official `retain.py` options ---------------------------------------------------------------
    #: `--time_file`: upstream defaults to no time information, so this defaults to false.
    use_time: bool = False
    use_log_time: bool = True  # --use_log_time
    embed_size: int = 128  # --embed_size
    alpha_hidden_dim_size: int = 128  # --alpha_hidden_dim_size
    beta_hidden_dim_size: int = 128  # --beta_hidden_dim_size
    batch_size: int = 100  # --batch_size
    n_epochs: int = 10  # --n_epochs
    L2_output: float = 0.001  # --L2_output
    L2_emb: float = 0.001  # --L2_emb
    L2_alpha: float = 0.001  # --L2_alpha
    L2_beta: float = 0.001  # --L2_beta
    keep_prob_emb: float = 0.5  # --keep_prob_emb
    keep_prob_context: float = 0.5  # --keep_prob_context
    log_eps: float = 1e-8  # --log_eps
    solver: str = "adadelta"  # --solver

    # -- Runtime ------------------------------------------------------------------------------------
    # The seed is not configuration: it is the `seed` argument of every step (request.execution.seed).
    device: str = "auto"
    #: Only used when the task supplies no `tuning` labels, standing in for the fixed hold-out
    #: fractions of the official `load_data_simple`.
    validation_fraction: float = 0.1
    max_train_rows: int | None = None
    #: Per-key code contributions of Eq. (5). 0 disables the extra pass entirely.
    interpretation_top_k: int = 0

    def validate(self):
        super().validate()
        if self.solver not in SOLVERS:
            raise ValueError(f"solver must be one of {SOLVERS}, got {self.solver!r}")
        if self.visit_bucket_minutes < 0:
            raise ValueError("visit_bucket_minutes must be >= 0")
        if self.lookback_days is not None and self.lookback_days <= 0:
            raise ValueError("lookback_days must be positive or null")
        if not 0.0 < self.keep_prob_emb <= 1.0 or not 0.0 < self.keep_prob_context <= 1.0:
            raise ValueError("keep_prob_emb and keep_prob_context must lie in (0, 1]")
        for name in ("embed_size", "alpha_hidden_dim_size", "beta_hidden_dim_size", "batch_size", "n_epochs"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be >= 1")
        for name in ("L2_output", "L2_emb", "L2_alpha", "L2_beta"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be >= 0")
        if self.min_code_count < 1:
            raise ValueError("min_code_count must be >= 1")
        for name in ("max_visits", "max_vocab_size", "max_train_rows"):
            value = getattr(self, name)
            if value is not None and value < 1:
                raise ValueError(f"{name} must be >= 1 or null")
        if not 0.0 <= self.validation_fraction < 1.0:
            raise ValueError("validation_fraction must lie in [0, 1)")
        if self.interpretation_top_k < 0:
            raise ValueError("interpretation_top_k must be >= 0")
        if self.log_eps <= 0:
            raise ValueError("log_eps must be positive")

    @property
    def gru_input_size(self) -> int:
        """RETAIN concatenates the visit duration onto the GRU inputs only (`build_model`)."""
        return self.embed_size + (1 if self.use_time else 0)
