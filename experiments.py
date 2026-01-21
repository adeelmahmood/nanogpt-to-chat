from dataclasses import dataclass


@dataclass
class ExperimentConfig:
    name: str
    use_rope: bool
    use_rmsnorm: bool
    use_gqa: bool
    use_qk_norm: bool


EXPERIMENTS = {
    "baseline": ExperimentConfig(
        name="baseline",
        use_rope=False,
        use_rmsnorm=False,
        use_gqa=False,
        use_qk_norm=False,
    ),
    "rope": ExperimentConfig(
        name="rope",
        use_rope=True,
        use_rmsnorm=False,
        use_gqa=False,
        use_qk_norm=True,
    ),
    "rope_rms": ExperimentConfig(
        name="rope_rms",
        use_rope=True,
        use_rmsnorm=True,
        use_gqa=False,
        use_qk_norm=True,
    ),
    "rope_rms_gqa": ExperimentConfig(
        name="rope_rms_gqa",
        use_rope=True,
        use_rmsnorm=True,
        use_gqa=True,
        use_qk_norm=True,
    ),
    "latest": ExperimentConfig(
        name="latest",
        use_rope=True,
        use_rmsnorm=True,
        use_gqa=True,
        use_qk_norm=True,
    ),
}
