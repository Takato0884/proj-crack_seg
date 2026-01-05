from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from ultralytics import YOLO
from typing import Optional
import os
import json


@dataclass(frozen=True)
class ExpConfig:
    # data / models
    data_path: str                      # e.g., "crack-seg.yaml"
    pretrained_model: str               # e.g., "yolo11n-seg.pt"

    # experiment management
    runs_dir: str = "runs"              # root output dir
    project: str = "crack_seg"          # project name under runs_dir
    name: str = "exp01"                 # experiment name (unique per run)
    exist_ok: bool = False              # overwrite existing exp dir if True

    # train params
    epochs: int = 100
    batch: int = 16
    patience: int = 20
    seed: int = 0
    imgsz: int = 640

    # Weights & Biases (optional)
    use_wandb: bool = False                 # enable W&B logging
    wandb_project: Optional[str] = None     # defaults to cfg.project if None
    wandb_entity: Optional[str] = None      # your W&B team/org; None for personal
    wandb_run_name: Optional[str] = None    # defaults to cfg.name if None
    wandb_mode: Optional[str] = None        # e.g., "offline" to avoid network


def main(cfg: ExpConfig):
    """
    Train -> (optional val) -> test only (no external eval module).
    Test runs on the dataset 'test' split defined in data yaml.
    """
    base_dir = Path(__file__).resolve().parent  # src/

    # Resolve paths robustly (relative to repo root = src/..)
    repo_root = base_dir.parent

    data_path = (repo_root / cfg.data_path).resolve() if not Path(cfg.data_path).is_absolute() else Path(cfg.data_path)
    runs_root = (repo_root / cfg.runs_dir).resolve() if not Path(cfg.runs_dir).is_absolute() else Path(cfg.runs_dir)

    if not data_path.exists():
        raise FileNotFoundError(f"data_path not found: {data_path}")

    # 1) Optional W&B init
    wandb_run = None
    if cfg.use_wandb:
        try:
            import wandb  # type: ignore
            # Respect an explicit mode if provided (e.g., offline)
            if cfg.wandb_mode:
                os.environ["WANDB_MODE"] = cfg.wandb_mode

            wandb_run = wandb.init(
                project=(cfg.wandb_project or cfg.project),
                entity=cfg.wandb_entity,
                name=(cfg.wandb_run_name or cfg.name),
                config={
                    "task": "segment",
                    "data_path": cfg.data_path,
                    "pretrained_model": cfg.pretrained_model,
                    "epochs": cfg.epochs,
                    "batch": cfg.batch,
                    "patience": cfg.patience,
                    "seed": cfg.seed,
                    "imgsz": cfg.imgsz,
                },
                reinit=True,
            )
            # Use 'epoch' as the global step for all metrics
            try:
                wandb.define_metric('epoch')
                wandb.define_metric('*', step_metric='epoch')
            except Exception:
                pass
        except Exception as e:
            print(f"[wandb] init failed: {e}. Continuing without W&B.")
            wandb_run = None

    # 2) Train
    model = YOLO(cfg.pretrained_model)

    # If W&B is active, attach a callback to log epoch-level metrics
    if wandb_run is not None:
        def _log_epoch_metrics(trainer):
            try:
                import wandb  # type: ignore
                # Ultralytics stores metrics in trainer.validator.metrics
                metrics = getattr(getattr(trainer, 'validator', None), 'metrics', None)

                # Prepare default values to avoid KeyErrors
                # Use phase-prefixed keys for clarity in W&B
                m = {
                    "val/mAP50(B)": None,
                    "val/mAP50(M)": None,
                    "val/mAP50-95(B)": None,
                    "val/mAP50-95(M)": None,
                    "val/precision(B)": None,
                    "val/precision(M)": None,
                    "val/recall(B)": None,
                    "val/recall(M)": None,
                    "epoch": getattr(trainer, 'epoch', None),
                    "phase": "val",
                }

                # Prefer results_dict for robust key mapping
                if metrics is not None:
                    rd = getattr(metrics, 'results_dict', None)
                    if isinstance(rd, dict):
                        # Map known keys by stripping 'metrics/' and prefixing 'val/'
                        for key, val in rd.items():
                            out_key = key
                            if isinstance(out_key, str) and out_key.startswith('metrics/'):
                                out_key = out_key[len('metrics/') :]
                            # Only log numeric values
                            if isinstance(val, (int, float)):
                                m[f"val/{out_key}"] = val

                # Log validation metrics at the same epoch step; commit to finalize the step
                e = getattr(trainer, 'epoch', None)
                if e is not None:
                    wandb.log(m, step=int(e), commit=True)
                else:
                    wandb.log(m, commit=True)
            except Exception as e:
                print(f"[wandb] epoch metrics logging failed: {e}")

        # Maintain running sums for train metrics between callbacks
        _train_accum = {"loss_sum": 0.0, "count": 0}

        def _to_scalar_loss(x):
            """Convert various loss representations to a scalar float safely."""
            try:
                # Handle torch.Tensor without importing torch explicitly
                if hasattr(x, 'numel') and callable(getattr(x, 'numel')):
                    n = x.numel()
                    if n == 0:
                        return None
                    if n == 1:
                        return float(x.item())
                    # Multi-element tensor: use mean to get a scalar
                    if hasattr(x, 'mean'):
                        return float(x.mean().item())
                if hasattr(x, 'item'):
                    return float(x.item())
                if isinstance(x, (int, float)):
                    return float(x)
            except Exception:
                return None
            return None

        def _log_train_batch(trainer):
            """Accumulate training loss per batch for averaging at epoch end."""
            try:
                # Try several common attributes
                val = _to_scalar_loss(getattr(trainer, 'tloss', None))
                if val is None:
                    val = _to_scalar_loss(getattr(trainer, 'loss', None))
                if val is None:
                    # Sum components if available
                    losses = getattr(trainer, 'loss_items', None)
                    s = None
                    if isinstance(losses, (list, tuple)):
                        parts = [
                            _to_scalar_loss(v) for v in losses
                        ]
                        parts = [p for p in parts if p is not None]
                        if parts:
                            s = float(sum(parts))
                    elif isinstance(losses, dict):
                        parts = [
                            _to_scalar_loss(v) for v in losses.values()
                        ]
                        parts = [p for p in parts if p is not None]
                        if parts:
                            s = float(sum(parts))
                    val = s

                if val is not None:
                    _train_accum["loss_sum"] += float(val)
                    _train_accum["count"] += 1
            except Exception:
                pass

        def _log_train_epoch(trainer):
            """Log training losses and LR at end of each epoch."""
            try:
                import wandb  # type: ignore
                # Average loss from accumulator; fallback to tloss if available
                avg_loss = None
                if _train_accum["count"] > 0:
                    avg_loss = _train_accum["loss_sum"] / _train_accum["count"]
                else:
                    tloss = getattr(trainer, 'tloss', None)
                    if hasattr(tloss, 'item'):
                        avg_loss = float(tloss.item())
                    elif isinstance(tloss, (int, float)):
                        avg_loss = float(tloss)

                # Reset accumulator for next epoch
                _train_accum["loss_sum"] = 0.0
                _train_accum["count"] = 0

                # Per-component losses if available
                losses = getattr(trainer, 'loss_items', None)

                # Learning rate
                lr = None
                try:
                    # optimizer param groups lr
                    opt = getattr(trainer, 'optimizer', None)
                    if opt and hasattr(opt, 'param_groups') and opt.param_groups:
                        lr = opt.param_groups[0].get('lr', None)
                except Exception:
                    pass

                e = getattr(trainer, 'epoch', None)
                payload = {
                    'epoch': e,
                    'phase': 'train',
                    'train/avg_loss': avg_loss,
                }

                # If per-component losses exist, try to name them
                if isinstance(losses, (list, tuple)):
                    for i, v in enumerate(losses):
                        key = f'train/loss_{i}'
                        from_scalar = None
                        try:
                            from_scalar = _to_scalar_loss(v)
                        except Exception:
                            from_scalar = None
                        payload[key] = from_scalar
                elif isinstance(losses, dict):
                    for k, v in losses.items():
                        try:
                            payload[f'train/{k}'] = _to_scalar_loss(v)
                        except Exception:
                            payload[f'train/{k}'] = None

                # Log training metrics first without committing the step; val callback will commit
                if e is not None:
                    wandb.log(payload, step=int(e), commit=False)
                else:
                    wandb.log(payload, commit=False)
            except Exception as e:
                print(f"[wandb] train epoch logging failed: {e}")

        # Register on-fit-epoch-end callback
        try:
            model.add_callback('on_fit_epoch_end', _log_epoch_metrics)
            model.add_callback('on_train_batch_end', _log_train_batch)
            model.add_callback('on_train_epoch_end', _log_train_epoch)
        except Exception as e:
            print(f"[wandb] failed to add epoch callback: {e}")
    model.train(
        data=str(data_path),
        task="segment",
        epochs=cfg.epochs,
        batch=cfg.batch,
        patience=cfg.patience,
        seed=cfg.seed,
        imgsz=cfg.imgsz,
        project=str(runs_root / cfg.project),  # runs/crack_seg
        name=cfg.name,                         # exp01
        exist_ok=cfg.exist_ok,
    )

    # Resolve best.pt from this training run
    save_dir = Path(model.trainer.save_dir)         # runs/crack_seg/segment/exp01
    best_pt = save_dir / "weights" / "best.pt"
    if not best_pt.exists():
        raise FileNotFoundError(f"best.pt not found: {best_pt}")

    # 3) Test only (split='test' in your crack-seg.yaml)
    best_model = YOLO(str(best_pt))
    test_metrics = best_model.val(
        task="segment",
        split="test",
        project=str(runs_root / cfg.project),
        name=f"{cfg.name}_test",
        exist_ok=True,
    )

    # 4) Log to W&B if enabled and available
    if wandb_run is not None:
        try:
            # Log summary metrics from validation (test split)
            # Ultralytics returns a dict-like object; ensure JSON-serializable
            tm = getattr(test_metrics, 'results_dict', None)
            raw_metrics = tm if isinstance(tm, dict) else dict(test_metrics or {})
            # Prefix numeric metric keys with 'test/' for clarity
            serializable_metrics = {}
            for k, v in raw_metrics.items():
                # Keep only numeric metrics prefixed; non-metrics (strings/paths) remain as-is
                if isinstance(v, (int, float)):
                    # Drop leading 'metrics/' if present
                    key = k
                    if isinstance(key, str) and key.startswith('metrics/'):
                        key = key[len('metrics/') :]
                    serializable_metrics[f"test/{key}"] = v
                else:
                    serializable_metrics[k] = v
            # Add some paths and phase info
            serializable_metrics.update({
                "train_dir": str(save_dir),
                "best_pt": str(best_pt),
                "phase": "test",
            })
            # Log metrics without explicit step; test runs once and doesn't need epoch coupling
            import wandb  # type: ignore
            wandb.log(serializable_metrics, commit=True)

            # Upload the best model as an artifact
            art = wandb.Artifact(
                name=f"{(cfg.wandb_run_name or cfg.name)}-best-pt",
                type="model",
                metadata={"framework": "ultralytics", "task": "segment"},
            )
            art.add_file(str(best_pt))
            wandb.log_artifact(art)

            # Finish the run to flush events
            wandb_run.finish()
        except Exception as e:
            print(f"[wandb] logging failed: {e}.")

    return {
        "train_dir": str(save_dir),
        "best_pt": str(best_pt),
        "test_metrics": test_metrics,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Train and test YOLO segmentation")
    parser.add_argument("--data-path", default="datasets/crack-seg/crack-seg.yaml", help="Path to data.yaml (e.g., datasets/crack-seg/crack-seg.yaml or datasets/subset_kanazawa/labels/data.yaml)")
    parser.add_argument("--pretrained-model", default="weight/yolo11n-seg.pt", help="Path to pretrained model weights (e.g., weight/yolo11n-seg.pt)")
    parser.add_argument("--runs-dir", default="runs", help="Root runs directory")
    parser.add_argument("--project", default="crack_seg", help="Project name under runs")
    parser.add_argument("--name", default="exp01", help="Experiment name")
    parser.add_argument("--exist-ok", action="store_true", help="Overwrite existing experiment directory")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--use-wandb", action="store_true", help="Enable Weights & Biases logging")
    parser.add_argument("--wandb-mode", default=None, help="W&B mode, e.g., offline")
    args = parser.parse_args()

    cfg = ExpConfig(
        data_path=args.data_path,
        pretrained_model=args.pretrained_model,
        runs_dir=args.runs_dir,
        project=args.project,
        name=args.name,
        exist_ok=args.exist_ok,
        epochs=args.epochs,
        batch=args.batch,
        patience=args.patience,
        seed=args.seed,
        imgsz=args.imgsz,
        use_wandb=args.use_wandb,
        wandb_mode=args.wandb_mode,
    )
    out = main(cfg)
    print("train_dir:", out["train_dir"])
    print("best_pt:", out["best_pt"])
