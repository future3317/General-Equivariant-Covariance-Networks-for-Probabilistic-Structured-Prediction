"""Single semantic contract for dielectric model construction and inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import torch

# Direct ``python scripts/<entrypoint>.py`` execution must resolve this
# repository's ``models`` package before any parent-workspace module with the
# same name.  Keeping the bootstrap here makes every runtime consumer use the
# same import root without relying on an ambient PYTHONPATH.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from equivcompiler import (
    CenteredSpectralWindowCovariance,
    FeatureSpec,
    FullCovariance,
    SpectralWindowCovariance,
    plan_readout,
)
from models import EquivariantBackbone
from spd_maps import RepresentationMetricMap


RUN_SPEC_FILE = "run_spec.json"
RUN_SPEC_VERSION = 1
INFERENCE_CONTRACT_VERSION = 1


def inference_contract_from_args(args: argparse.Namespace, device: str | torch.device) -> dict[str, Any]:
    """Return the single precision/runtime contract used by all evaluators."""
    device_obj = torch.device(device)
    backbone_precision = getattr(args, "backbone_precision", "fp32")
    use_bf16 = backbone_precision == "bf16" and device_obj.type == "cuda"
    return {
        "version": INFERENCE_CONTRACT_VERSION,
        "device_type": device_obj.type,
        "backbone_precision": "bf16" if use_bf16 else "fp32",
        "operator_precision": "fp32",
        "autocast_dtype": "bfloat16" if use_bf16 else "none",
        "allow_tf32": bool(getattr(args, "allow_tf32", False)) if device_obj.type == "cuda" else False,
        "cudnn_benchmark": bool(device_obj.type == "cuda"),
    }


def inference_contract_hash(contract: Mapping[str, Any]) -> str:
    payload = json.dumps(dict(contract), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def source_provenance(repo_root: str | Path) -> dict[str, Any]:
    """Capture a complete source identity; dirty trees are explicit failures."""
    root = Path(repo_root)
    commit = subprocess.check_output(
        ["git", "-C", str(root), "rev-parse", "HEAD"], text=True
    ).strip()
    status = subprocess.check_output(
        ["git", "-C", str(root), "status", "--porcelain"], text=True
    )
    tracked = subprocess.check_output(
        ["git", "-C", str(root), "ls-files", "-s"], text=True
    )
    source_hash = hashlib.sha256(
        (commit + "\n" + status + "\n" + tracked).encode("utf-8")
    ).hexdigest()
    return {
        "commit": commit,
        "dirty": bool(status.strip()),
        "status_sha256": hashlib.sha256(status.encode("utf-8")).hexdigest(),
        "source_hash": source_hash,
    }


def dataset_provenance(data_dir: str | Path) -> dict[str, Any]:
    """Hash split metadata and index mappings used by the dielectric cache."""
    root = Path(data_dir)
    splits: dict[str, dict[str, str]] = {}
    for split in ("train", "val", "test"):
        graph_dir = root / f"{split}_graphs_full"
        files = [graph_dir / "metadata.json", graph_dir / "index_mapping.json"]
        splits[split] = {path.name: sha256_file(path) for path in files if path.is_file()}
    payload = json.dumps(splits, sort_keys=True, separators=(",", ":"))
    return {"splits": splits, "dataset_hash": hashlib.sha256(payload.encode("utf-8")).hexdigest()}


def checkpoint_chain_provenance(checkpoint_dirs: list[str | Path]) -> dict[str, Any]:
    entries = []
    for directory in checkpoint_dirs:
        path = Path(directory) / "best_model.pt"
        entries.append({"path": str(path), "sha256": sha256_file(path)})
    payload = json.dumps(entries, sort_keys=True, separators=(",", ":"))
    return {
        "checkpoints": entries,
        "checkpoint_chain_sha256": hashlib.sha256(payload.encode("utf-8")).hexdigest(),
    }


def compilation_record_with_hash(compilation: Mapping[str, Any]) -> dict[str, Any]:
    record = json.loads(json.dumps(dict(compilation)))
    payload = dict(record)
    payload["compatibility_hash"] = None
    record["compatibility_hash"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return record


def configure_inference_contract(contract: Mapping[str, Any]) -> None:
    """Apply the recorded backend flags before any model evaluation."""
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = bool(contract.get("allow_tf32", False))
        torch.backends.cudnn.allow_tf32 = bool(contract.get("allow_tf32", False))
        torch.backends.cudnn.benchmark = bool(contract.get("cudnn_benchmark", True))


def forward_dielectric(
    model,
    batch,
    *,
    target: torch.Tensor | None = None,
    return_scale: bool = False,
    contract: Mapping[str, Any] | None = None,
    faithful: bool = False,
    covariance_residual: torch.Tensor | None = None,
    pseudo_sqrt_covariance: torch.Tensor | None = None,
):
    """Run one prediction under the recorded BF16/FP32 contract.

    ``faithful=True`` is a training-only objective boundary: mean/trunk
    gradients use MSE while the covariance NLL uses detached features and a
    detached (optionally out-of-fold) residual.  Evaluation must keep the
    default probabilistic path so reported NLL remains the proper model score.
    """
    contract = contract or {"backbone_precision": "fp32"}
    use_bf16 = contract.get("backbone_precision") == "bf16" and batch.pos.device.type == "cuda"
    if faithful and pseudo_sqrt_covariance is not None:
        raise ValueError("faithful NLL and Wasserstein warm-up are distinct objectives")
    def _from_features(features, graph_batch):
        if pseudo_sqrt_covariance is not None:
            return model.forward_isotropic_wasserstein_from_features(
                features, graph_batch, pseudo_sqrt_covariance=pseudo_sqrt_covariance
            )
        if faithful:
            if target is None:
                raise ValueError("faithful inference requires target")
            return model.forward_faithful_from_features(
                features,
                graph_batch,
                target=target,
                covariance_residual=covariance_residual,
            )
        return model.forward_from_features(
            features, graph_batch, target=target, return_scale=return_scale
        )
    if not use_bf16:
        if faithful:
            node_features, graph_batch = model.backbone(batch)
            return _from_features(node_features, graph_batch)
        return model(batch, target=target, return_scale=return_scale)
    with torch.autocast(device_type=batch.pos.device.type, dtype=torch.bfloat16):
        node_features, graph_batch = model.backbone(batch)
    return _from_features(node_features.float(), graph_batch)


@dataclass(frozen=True)
class DielectricRunSpec:
    """All fields that determine the model's representation and likelihood."""

    hidden_dim: int
    lmax: int
    num_layers: int
    num_basis: int
    atom_features: str
    tp_backend: str
    cueq_method: str
    covariance_parameterization: str
    log_variance_min: float
    log_variance_max: float
    shape_min: float
    shape_max: float
    volume_min: float
    volume_max: float
    distribution: str
    student_t_dof: float
    representation_metric: str
    metric_scalar: float | None = None
    metric_l2: float | None = None

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "DielectricRunSpec":
        values = vars(args)
        required = {
            field
            for field in cls.__dataclass_fields__
            if field not in {"metric_scalar", "metric_l2"}
        }
        missing = sorted(required.difference(values))
        if missing:
            raise ValueError(f"run configuration is missing semantic fields: {missing}")
        return cls(**{field: values.get(field) for field in cls.__dataclass_fields__})

    @classmethod
    def from_dict(cls, values: Mapping[str, Any]) -> "DielectricRunSpec":
        return cls(**{field: values.get(field) for field in cls.__dataclass_fields__})

    def as_dict(self) -> dict[str, Any]:
        return asdict(self)


def _covariance(spec: DielectricRunSpec):
    if spec.covariance_parameterization == "matrix_exp":
        return FullCovariance()
    if spec.covariance_parameterization == "spectral_window":
        return SpectralWindowCovariance(spec.log_variance_min, spec.log_variance_max)
    if spec.covariance_parameterization == "centered_spectral_window":
        return CenteredSpectralWindowCovariance(
            spec.shape_min, spec.shape_max, spec.volume_min, spec.volume_max
        )
    raise ValueError(
        f"unsupported covariance parameterization: {spec.covariance_parameterization}"
    )


def build_dielectric_model(spec: DielectricRunSpec, device: str | torch.device):
    """Build exactly the model described by ``spec`` and return its compilation."""
    backbone = EquivariantBackbone(
        hidden_dim=spec.hidden_dim,
        lmax=spec.lmax,
        num_layers=spec.num_layers,
        atom_feature_dim=49,
        num_basis=spec.num_basis,
        atom_features=spec.atom_features,
        tp_backend=spec.tp_backend,
        cueq_method=spec.cueq_method,
    )
    plan = plan_readout(
        FeatureSpec.from_backbone(backbone),
        output="0e + 2e",
        covariance=_covariance(spec),
        distribution=spec.distribution,
        student_t_dof=spec.student_t_dof,
        output_scope="global",
    )
    model = plan.bind(backbone).to(device)
    if spec.representation_metric == "block_auto":
        if spec.metric_scalar is None or spec.metric_l2 is None:
            raise ValueError("block_auto requires saved metric_scalar and metric_l2")
        metric = torch.tensor(
            [spec.metric_scalar] + [spec.metric_l2] * 5,
            dtype=torch.float32,
            device=device,
        )
        model.spd_map = RepresentationMetricMap(model.spd_map, metric).to(device)
    elif spec.representation_metric != "none":
        raise ValueError(f"unsupported representation metric: {spec.representation_metric}")
    return model, plan.compilation


def write_run_spec(
    checkpoint_dir: str | Path,
    spec: DielectricRunSpec,
    *,
    compilation: dict[str, Any],
    training_stage: str,
    init_checkpoint: str | None,
    inference_contract: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
) -> None:
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    compilation_record = compilation_record_with_hash(compilation)
    payload = {
        "version": RUN_SPEC_VERSION,
        "model": spec.as_dict(),
        "compilation": compilation_record,
        "training_stage": training_stage,
        "init_checkpoint": init_checkpoint,
        "inference_contract": dict(inference_contract or {}),
        "inference_contract_hash": (
            inference_contract_hash(inference_contract)
            if inference_contract is not None
            else None
        ),
        "provenance": dict(provenance or {}),
    }
    (checkpoint_dir / RUN_SPEC_FILE).write_text(json.dumps(payload, indent=2))


def load_run_spec(checkpoint_dir: str | Path) -> DielectricRunSpec:
    path = Path(checkpoint_dir) / RUN_SPEC_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {RUN_SPEC_FILE}; run explicit migration before evaluation: {path}"
        )
    payload = json.loads(path.read_text())
    if payload.get("version") != RUN_SPEC_VERSION:
        raise ValueError(f"unsupported run spec version: {payload.get('version')}")
    return DielectricRunSpec.from_dict(payload["model"])


def load_run_record(checkpoint_dir: str | Path) -> dict[str, Any]:
    """Load the complete immutable run record, without compatibility fallbacks."""
    path = Path(checkpoint_dir) / RUN_SPEC_FILE
    if not path.is_file():
        raise FileNotFoundError(
            f"missing {RUN_SPEC_FILE}; run explicit migration before evaluation: {path}"
        )
    payload = json.loads(path.read_text())
    if payload.get("version") != RUN_SPEC_VERSION:
        raise ValueError(f"unsupported run spec version: {payload.get('version')}")
    if not isinstance(payload.get("model"), dict):
        raise ValueError(f"invalid model record in {path}")
    return payload


def load_dielectric_checkpoint(
    checkpoint_dir: str | Path,
    device: str | torch.device,
    *,
    filename: str = "best_model.pt",
):
    checkpoint_dir = Path(checkpoint_dir)
    spec = load_run_spec(checkpoint_dir)
    model, compilation = build_dielectric_model(spec, device)
    state_path = checkpoint_dir / filename
    if not state_path.is_file():
        raise FileNotFoundError(f"missing checkpoint: {state_path}")
    model.load_state_dict(torch.load(state_path, map_location=device))
    model.eval()
    return model, spec, compilation


def load_dielectric_data_args(checkpoint_dir: str | Path) -> argparse.Namespace:
    """Load non-semantic data-loader settings recorded beside a RunSpec."""
    path = Path(checkpoint_dir) / "args.json"
    if not path.is_file():
        raise FileNotFoundError(f"missing runtime data settings: {path}")
    values = json.loads(path.read_text())
    if not isinstance(values, dict):
        raise ValueError(f"invalid runtime data settings: {path}")
    return argparse.Namespace(**values)


@torch.inference_mode()
def collect_dielectric_predictions(
    model,
    dataloader,
    device: str | torch.device,
    *,
    inference_contract: Mapping[str, Any] | None = None,
):
    """Collect mean, target, and FP64-materialized scatter in one contract."""
    mean, scale, target_irreps, target_km = [], [], [], []
    for batch in dataloader:
        batch = batch.to(device)
        if batch.edge_index is None or batch.edge_index.numel() == 0:
            continue
        result = forward_dielectric(model, batch, contract=inference_contract)
        if model.spd_map is None:
            raise TypeError("probabilistic dielectric evaluation requires an SPD map")
        mean.append(result["mu"].double().cpu())
        scale.append(model.spd_map(result["params"].double()).cpu())
        target_irreps.append(batch.y_irreps.double().cpu())
        target_km.append(batch.y_km.double().cpu())
    if not mean:
        raise RuntimeError("no valid dielectric graphs were available for evaluation")
    return {
        "mu_irreps": torch.cat(mean),
        "scale_irreps": torch.cat(scale),
        "y_irreps": torch.cat(target_irreps),
        "y_km": torch.cat(target_km),
    }
