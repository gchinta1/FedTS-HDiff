"""Typed, flat experiment configuration.

One :class:`Config` fully describes one run (dataset x method x seed x variant).
YAML files are flat ``key: value`` mappings; unknown keys are rejected and every
enumerated value is validated.  ``--set key=value`` overrides are parsed with
YAML semantics, so ``--set drift_rounds=[100,175]`` works.

The *fingerprint* hashes the scientific content of a configuration (everything
except paths, device and cosmetic names); it decides whether a stored result can
be reused on resume.
"""

from __future__ import annotations

import dataclasses
import functools
import hashlib
import json
from dataclasses import dataclass, field, fields, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

PACKAGE_CONFIG_DIR = Path(__file__).resolve().parent / "configs"

DATASETS = ("mitbih", "ptbxl", "electricity", "metrla", "synthetic")
METHODS = (
    "fedavg",
    "fedprox",
    "fednova",
    "feddyn",
    "scaffold",
    "fedrep",
    "fedtdd",
    "fedts_hdiff",
    "cfl",
    "central",
)
MODELS = ("unet", "csdi", "lstm", "gan")
GROUPINGS = ("balance", "similarity", "kmeans", "random", "round_robin", "none")
DESCRIPTORS = ("label", "signal", "psd")
FORMS_SELECT = ("softmax", "linear")
FORMS_AGG = ("exp", "linear")
EDGE_WEIGHTING = ("homogeneity", "size")
SPLITS = ("time", "random")
SCHEDULES = ("linear", "quad")
SAMPLERS = ("ddim", "ddpm")
SPACINGS = ("trailing", "legacy")
OPT_STATE = ("persistent", "reset")
ASSIGN_ORDER = ("fixed", "shuffled")

# Fields that never change scientific results and are excluded from the fingerprint.
# eval_batch is scientific: ancestral (DDPM) sampling draws its step noise chunk by chunk.
NON_SCIENTIFIC = frozenset({"name", "data_root", "device", "save_checkpoint", "tags", "client_batch"})
# Fields added after the first campaign: omitted from the fingerprint while at these defaults, so
# configurations that do not use them keep their fingerprints (and completed roots stay valid).
FINGERPRINT_DEFAULT_OMIT: dict[str, Any] = {"edge_mix": 1.0, "personal_head": False}


@dataclass(frozen=True)
class Config:
    # ---- identity -----------------------------------------------------------------
    name: str = "run"
    tags: tuple[str, ...] = ()
    dataset: str = "synthetic"
    data_root: str = "data"
    seed: int = 1
    device: str = "auto"

    # ---- data ---------------------------------------------------------------------
    past_len: int = 64
    future_len: int = 64
    max_clients: int = 0  # 0 = every prepared client
    min_windows: int = 0  # extra load-time filter (prepare already applies the paper bounds)
    max_windows: int = 0  # 0 = no extra cap
    split: str = "time"
    split_seed: int = 123
    test_ratio: float = 0.2
    val_ratio: float = 0.0  # fraction of each client's training part held out for tuning
    # windows dropped between the training block and the next held-out block (time split) so no
    # training window contains a held-out forecast target; -1 = ceil(future_len / stride) - 1
    split_gap: int = -1

    # ---- model --------------------------------------------------------------------
    model: str = "unet"
    base_ch: int = 32
    time_dim: int = 128
    csdi_channels: int = 64
    csdi_layers: int = 4
    csdi_heads: int = 8
    lstm_hidden: int = 64
    lstm_layers: int = 2
    gan_noise_dim: int = 32
    gan_hidden: int = 64
    gan_n_critic: int = 5
    gan_gp: float = 10.0

    # ---- diffusion ----------------------------------------------------------------
    diffusion_steps: int = 500
    beta_start: float = 1e-4
    beta_end: float = 0.02
    beta_schedule: str = "linear"
    sampler: str = "ddim"
    ddim_steps: int = 200
    ddim_eta: float = 0.0
    ddim_spacing: str = "trailing"  # "legacy" reproduces the original truncated spacing

    # ---- federated training -------------------------------------------------------
    method: str = "fedts_hdiff"
    rounds: int = 200
    local_epochs: int = 4
    batch_size: int = 64
    lr: float = 1e-4
    weight_decay: float = 0.01  # torch.optim.AdamW default, as in the original code
    adam_beta1: float = 0.9
    adam_beta2: float = 0.999
    adam_eps: float = 1e-8
    client_opt_state: str = "persistent"  # the original keeps one AdamW per client across rounds
    participation: float = 1.0
    grad_clip: float = 0.0
    # clients trained simultaneously (U-Net diffusion only); 1 = one at a time.  Same random
    # draws either way, so results agree up to floating point (not part of the fingerprint).
    client_batch: int = 32

    # ---- hierarchy and grouping (FedTS-HDiff, hierarchical presets, CFL) ----------
    num_edges: int = 4
    edge_rounds: int = 1  # kappa: edge aggregations per global (cloud) aggregation
    grouping: str = "balance"
    regroup_every: int = 1  # 0 = static: assignment of the first round is kept
    descriptor: str = "label"
    assign_order: str = "fixed"
    select_form: str = "softmax"
    a_sel: float = 10.0
    b_sel: float = 0.0
    tau_sel: float = 1.0
    size_penalty: float = 1.0
    agg_form: str = "exp"
    alpha_agg: float = 1.0
    beta_agg: float = 1.0
    edge_weighting: str = "homogeneity"
    homogeneity_ref: str = "auto"  # auto: uniform for label descriptors (paper), mean otherwise
    # Group-level personalisation (needs static groups): each edge keeps its own model; at every cloud
    # round an edge starts from edge_mix * global + (1 - edge_mix) * its previous edge model, and its
    # clients are evaluated with that mixture.  1.0 = standard FedTS-HDiff (edges restart from global),
    # 0.0 = independent group models (clustered FL on the similarity partition).
    edge_mix: float = 1.0
    # Personalised hierarchy: every client keeps its own output head (FedRep-style local schedule:
    # fedrep_head_epochs on the head, then the body) and is evaluated with it; only the body is
    # aggregated (edges, then cloud), so the head never leaves the client.
    personal_head: bool = False

    # ---- baselines ----------------------------------------------------------------
    prox_mu: float = 0.01
    dyn_alpha: float = 0.01
    fedrep_head_epochs: int = 1
    fedtdd_lambda: float = 0.1
    fedtdd_syn_per_client: int = 32
    fedtdd_syn_ddim_steps: int = 25

    # ---- concept drift ------------------------------------------------------------
    drift_rounds: tuple[int, ...] = ()
    drift_fraction: float = 0.3
    drift_seed: int = 7

    # ---- evaluation ---------------------------------------------------------------
    eval_every: int = 10
    eval_curve_windows: int = 256
    eval_curve_samples: int = 50
    eval_final_windows: int = 1024
    eval_final_samples: int = 100
    eval_seed: int = 2024
    eval_split: str = "test"  # "val" for hyperparameter selection runs
    eval_extra_ddim: tuple[int, ...] = ()
    eval_batch: int = 16384
    diag_rounds: tuple[int, ...] = ()
    diag_windows: int = 256
    diag_random_partitions: int = 20
    save_checkpoint: bool = True

    # -------------------------------------------------------------------------------
    @property
    def window_len(self) -> int:
        return self.past_len + self.future_len

    def validate(self) -> "Config":
        def choice(key: str, allowed: Iterable[str]) -> None:
            val = getattr(self, key)
            if val not in allowed:
                raise ValueError(f"config {key}={val!r}: expected one of {sorted(allowed)}")

        choice("dataset", DATASETS)
        choice("method", METHODS)
        choice("model", MODELS)
        choice("grouping", GROUPINGS)
        choice("descriptor", DESCRIPTORS)
        choice("select_form", FORMS_SELECT)
        choice("agg_form", FORMS_AGG)
        choice("edge_weighting", EDGE_WEIGHTING)
        choice("split", SPLITS)
        choice("beta_schedule", SCHEDULES)
        choice("sampler", SAMPLERS)
        choice("ddim_spacing", SPACINGS)
        choice("client_opt_state", OPT_STATE)
        choice("assign_order", ASSIGN_ORDER)
        choice("eval_split", ("test", "val"))
        choice("homogeneity_ref", ("auto", "uniform", "mean"))
        if self.eval_split == "val" and self.val_ratio <= 0:
            raise ValueError("eval_split=val needs val_ratio > 0")
        positive = ("past_len", "future_len", "rounds", "local_epochs", "batch_size", "num_edges",
                    "edge_rounds", "diffusion_steps", "ddim_steps", "eval_final_windows",
                    "eval_final_samples", "eval_curve_windows", "eval_curve_samples")
        for key in positive:
            if getattr(self, key) < 1:
                raise ValueError(f"config {key} must be >= 1, got {getattr(self, key)}")
        if not 0.0 < self.participation <= 1.0:
            raise ValueError("participation must be in (0, 1]")
        if not 0.0 < self.test_ratio < 1.0 or not 0.0 <= self.val_ratio < 1.0:
            raise ValueError("test_ratio must be in (0,1) and val_ratio in [0,1)")
        if self.ddim_steps > self.diffusion_steps:
            raise ValueError("ddim_steps cannot exceed diffusion_steps")
        if self.model in ("lstm", "gan") and self.method not in ("fedavg", "fedts_hdiff", "central"):
            raise ValueError(f"model={self.model} is supported with fedavg, fedts_hdiff and central only")
        if self.method == "fedrep" and self.model != "unet":
            raise ValueError("fedrep splits the UNet into body/head and requires model=unet")
        if self.method == "fedtdd" and self.past_len != self.future_len:
            raise ValueError("fedtdd synthetic chaining requires past_len == future_len")
        if self.method == "fedrep" and self.fedrep_head_epochs >= self.local_epochs:
            raise ValueError("fedrep_head_epochs must be smaller than local_epochs")
        if self.method == "cfl" and self.participation < 1.0:
            raise ValueError("cfl is implemented for full participation")
        if self.method == "central" and self.drift_rounds:
            raise ValueError("central trains on a merged copy of the data that drift events do not update")
        if not 0.0 <= self.edge_mix <= 1.0:
            raise ValueError("edge_mix must be in [0, 1]")
        if self.edge_mix < 1.0 and (self.method != "fedts_hdiff" or self.regroup_every != 0 or self.diag_rounds):
            raise ValueError("edge_mix < 1 (group personalisation) needs method=fedts_hdiff, static groups "
                             "(regroup_every=0) and no gradient diagnostics")
        if self.personal_head:
            if self.method != "fedts_hdiff" or self.model != "unet" or self.edge_mix < 1.0 or self.diag_rounds:
                raise ValueError("personal_head needs method=fedts_hdiff, model=unet, edge_mix=1 and no "
                                 "gradient diagnostics")
            if self.fedrep_head_epochs >= self.local_epochs:
                raise ValueError("fedrep_head_epochs must be smaller than local_epochs")
        if self.diag_rounds and self.method in ("cfl", "fedrep", "central"):
            raise ValueError("gradient diagnostics are defined at a single global model (not cfl/fedrep/central)")
        if self.model == "unet" and self.future_len % 4:
            raise ValueError("the UNet pools twice: future_len must be divisible by 4")
        if self.past_len != self.future_len and self.model == "unet":
            raise ValueError("the UNet concatenates past and noisy future channel-wise: past_len must equal future_len")
        return self

    # -------------------------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        out = dataclasses.asdict(self)
        for k, v in out.items():
            if isinstance(v, tuple):
                out[k] = list(v)
        return out

    def scientific_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.to_dict().items()
                if k not in NON_SCIENTIFIC and not (k in FINGERPRINT_DEFAULT_OMIT and v == FINGERPRINT_DEFAULT_OMIT[k])}

    def fingerprint(self, extra: Mapping[str, Any] | None = None) -> str:
        payload = {"config": self.scientific_dict(), "extra": dict(extra or {})}
        blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def with_updates(self, **updates: Any) -> "Config":
        return replace(self, **_coerce(updates)).validate()


_FIELD_TYPES = {f.name: f for f in fields(Config)}


def _coerce(updates: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in updates.items():
        if key not in _FIELD_TYPES:
            raise KeyError(f"unknown config key {key!r}")
        default = _FIELD_TYPES[key].default
        if default is dataclasses.MISSING:  # pragma: no cover - every field has a default
            default = _FIELD_TYPES[key].default_factory()
        out[key] = _coerce_value(key, value, default)
    return out


def _coerce_value(key: str, value: Any, default: Any) -> Any:
    if isinstance(default, bool):
        if isinstance(value, bool):
            return value
        if isinstance(value, str) and value.lower() in ("true", "false"):
            return value.lower() == "true"
        raise TypeError(f"config {key} expects a bool, got {value!r}")
    if isinstance(default, tuple):
        if value is None:
            return ()
        if isinstance(value, (int, str)):
            value = [value]
        return tuple(value)
    if isinstance(default, int):
        if isinstance(value, float) and value.is_integer():
            value = int(value)
        if not isinstance(value, int) or isinstance(value, bool):
            raise TypeError(f"config {key} expects an int, got {value!r}")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise TypeError(f"config {key} expects a number, got {value!r}")
        return float(value)
    if isinstance(default, str):
        if not isinstance(value, str):
            raise TypeError(f"config {key} expects a string, got {value!r}")
        return value
    return value


def parse_overrides(items: Iterable[str]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for item in items:
        if "=" not in item:
            raise ValueError(f"--set expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = yaml.safe_load(raw)
    return out


def load_yaml(path: str | Path) -> dict[str, Any]:
    data = yaml.safe_load(Path(path).read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path}: a config file must be a flat mapping")
    return data


def dataset_defaults(dataset: str) -> dict[str, Any]:
    """Paper settings for ``dataset`` (``configs/<dataset>.yaml`` shipped with the package)."""
    return dict(_dataset_defaults_cached(dataset))


@functools.lru_cache(maxsize=None)
def _dataset_defaults_cached(dataset: str) -> dict[str, Any]:
    path = PACKAGE_CONFIG_DIR / f"{dataset}.yaml"
    return load_yaml(path) if path.exists() else {"dataset": dataset}


def build_config(
    base: Mapping[str, Any] | None = None,
    *,
    dataset: str | None = None,
    overrides: Mapping[str, Any] | None = None,
) -> Config:
    """Compose package dataset defaults <- base mapping <- overrides into a validated Config."""
    merged: dict[str, Any] = {}
    ds = dataset or (base or {}).get("dataset") or (overrides or {}).get("dataset")
    if ds:
        merged.update(dataset_defaults(ds))
    merged.update(base or {})
    merged.update(overrides or {})
    return Config().with_updates(**merged)


def load_config(path: str | Path | None, overrides: Iterable[str] = ()) -> Config:
    base = load_yaml(path) if path else {}
    return build_config(base, overrides=parse_overrides(overrides))


def save_config(cfg: Config, path: str | Path) -> None:
    from .utils import atomic_write_text

    atomic_write_text(path, yaml.safe_dump(cfg.to_dict(), sort_keys=True))
