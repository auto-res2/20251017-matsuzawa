import os
import sys
import time
from pathlib import Path
from typing import Dict, Any, List, Optional, Sequence

import hydra
import optuna
from hydra.utils import get_original_cwd
from omegaconf import OmegaConf

import torch
from torch import nn
from sklearn.metrics import accuracy_score, f1_score
from transformers import get_linear_schedule_with_warmup
import wandb

# ---------------------------------------------------------------------------
# Local imports (executed with -m from repository root)
# ---------------------------------------------------------------------------
from src.preprocess import build_dataloaders
from src.model import build_model

# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _set_seed(seed: int = 42):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _save_cfg(cfg, results_dir: Path):
    """Persist the (flattened) Hydra config for reproduction/evaluation."""
    results_dir.mkdir(parents=True, exist_ok=True)
    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        OmegaConf.save(config=cfg, f=str(cfg_path))


# ---------------------------------------------------------------------------
# Core training / evaluation routine
# ---------------------------------------------------------------------------

def _train_eval(cfg, trial: Optional[optuna.Trial] = None, verbose: bool = True) -> Dict[str, Any]:
    """Complete training & validation loop. Returns best metrics recorded."""

    trial_mode: bool = bool(cfg.get("trial_mode", False))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --------------------- Data & Model -----------------------------------
    train_loader, val_loader, _, tokenizer = build_dataloaders(cfg, trial_mode=trial_mode)
    model = build_model(cfg, tokenizer).to(device)

    # --------------------- Optimizer & LR schedule ------------------------
    opt_map = {"adamw": torch.optim.AdamW, "sgd": torch.optim.SGD}
    optim_cls = opt_map.get(str(cfg.training.optimizer).lower())
    if optim_cls is None:
        raise ValueError(f"Unsupported optimizer {cfg.training.optimizer}")

    optimizer = optim_cls(model.parameters(), lr=cfg.training.learning_rate, weight_decay=cfg.training.weight_decay)
    total_steps = len(train_loader) * (1 if trial_mode else cfg.training.epochs)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=cfg.training.warmup_steps,
        num_training_steps=max(total_steps, 1),
    )
    criterion = nn.CrossEntropyLoss()

    # --------------------- WandB ------------------------------------------
    use_wandb = (trial is None) and (cfg.wandb.mode != "disabled") and (not trial_mode)
    if use_wandb:
        run = wandb.init(
            entity=cfg.wandb.entity,
            project=cfg.wandb.project,
            id=cfg.run_id,
            resume="allow",
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        if verbose:
            print(f"WandB URL: {wandb.run.get_url()}")
    else:
        run = None

    # --------------------- Training loop ----------------------------------
    epochs = 1 if trial_mode else (cfg.optuna.get("max_epochs", cfg.training.epochs) if trial is not None else cfg.training.epochs)
    best_val_acc, best_val_f1 = 0.0, 0.0

    global_step = 0
    for epoch in range(1, epochs + 1):
        model.train()
        epoch_losses: List[float] = []
        epoch_true: List[int] = []
        epoch_pred: List[int] = []

        for step, batch in enumerate(train_loader):
            if trial_mode and step > 1:
                break
            labels = batch.pop("labels").to(device)
            for k in batch:
                batch[k] = batch[k].to(device)
            outputs = model(batch)
            loss = criterion(outputs, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.training.max_grad_norm)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

            epoch_losses.append(loss.item())
            epoch_pred.extend(outputs.argmax(dim=-1).detach().cpu().tolist())
            epoch_true.extend(labels.detach().cpu().tolist())
            global_step += 1

        train_loss = float(sum(epoch_losses) / max(1, len(epoch_losses)))
        train_acc = accuracy_score(epoch_true, epoch_pred)
        train_f1 = f1_score(epoch_true, epoch_pred, average="weighted")

        # --------------------- Validation ---------------------------------
        model.eval()
        val_losses: List[float] = []
        val_true: List[int] = []
        val_pred: List[int] = []
        start_inf = time.time()
        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                if trial_mode and step > 1:
                    break
                labels = batch.pop("labels").to(device)
                for k in batch:
                    batch[k] = batch[k].to(device)
                outputs = model(batch)
                val_losses.append(criterion(outputs, labels).item())
                val_pred.extend(outputs.argmax(dim=-1).detach().cpu().tolist())
                val_true.extend(labels.detach().cpu().tolist())
        inference_time = (time.time() - start_inf) / max(1, len(val_loader))
        val_loss = float(sum(val_losses) / max(1, len(val_losses)))
        val_acc = accuracy_score(val_true, val_pred)
        val_f1 = f1_score(val_true, val_pred, average="weighted")

        best_val_acc = max(best_val_acc, val_acc)
        best_val_f1 = max(best_val_f1, val_f1)

        log_dict = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "train_acc": train_acc,
            "val_acc": val_acc,
            "train_f1": train_f1,
            "val_f1": val_f1,
            "best_val_acc": best_val_acc,
            "best_val_f1": best_val_f1,
            "inference_time": inference_time,
        }
        if use_wandb:
            wandb.log(log_dict, step=epoch)
        if verbose:
            print(f"Epoch {epoch}: {log_dict}")

    if run is not None:
        wandb.finish()

    return {"best_val_acc": best_val_acc, "best_val_f1": best_val_f1}


# ---------------------------------------------------------------------------
# Optuna helpers
# ---------------------------------------------------------------------------

def _apply_optuna_suggestions(cfg, trial: optuna.Trial, param_map: Dict[str, Sequence[str]]):
    """Mutate `cfg` in-place with values suggested by Optuna."""
    for param_name, space in cfg.optuna.search_space.items():
        stype = str(space.type)
        if stype == "loguniform":
            val = trial.suggest_float(param_name, float(space.low), float(space.high), log=True)
        elif stype == "uniform":
            val = trial.suggest_float(param_name, float(space.low), float(space.high))
        elif stype == "categorical":
            val = trial.suggest_categorical(param_name, list(space.choices))
        elif stype == "int":
            val = trial.suggest_int(param_name, int(space.low), int(space.high), step=int(space.get("step", 1)))
        else:
            raise ValueError(f"Unsupported search space type: {space.type}")

        # propagate into cfg along mapped path (create intermediates if missing)
        ptr = cfg
        for key in param_map[param_name][:-1]:
            if key not in ptr:
                ptr[key] = {}
            ptr = ptr[key]
        ptr[param_map[param_name][-1]] = val


# ---------------------------------------------------------------------------
# Hydra entrypoint ----------------------------------------------------------
# ---------------------------------------------------------------------------

@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):  # noqa: C901
    """Hydra-managed entrypoint. Handles Optuna & final training."""

    _set_seed()

    # ---------------------------------------------------------------------
    # Flatten configuration: merge run-specific subtree into root namespace
    # ---------------------------------------------------------------------
    OmegaConf.set_struct(cfg, False)
    flat_cfg = OmegaConf.merge(cfg, cfg.run)  # later keys (run) overwrite root duplicates
    if cfg.get("trial_mode", False):
        flat_cfg.trial_mode = True
        flat_cfg.wandb.mode = "disabled"
    if "results_dir" in cfg:
        flat_cfg.results_dir = cfg.results_dir

    # results directory ----------------------------------------------------
    results_dir = Path(flat_cfg.results_dir).expanduser().absolute()
    _save_cfg(flat_cfg, results_dir)

    # --------------------- Hyper-parameter search -------------------------
    if flat_cfg.get("optuna") and int(flat_cfg.optuna.n_trials) > 0 and not flat_cfg.get("trial_mode", False):
        param_map = {
            "learning_rate": ["training", "learning_rate"],
            "batch_size": ["training", "batch_size"],
            "weight_decay": ["training", "weight_decay"],
            "adapter_reduction_factor": ["model", "task_adapters", "reduction_factor"],
        }

        study_path = results_dir / "optuna_study.db"
        study = optuna.create_study(
            study_name=str(flat_cfg.run_id),
            direction=str(flat_cfg.optuna.direction),
            storage=f"sqlite:///{study_path}",
            load_if_exists=True,
        )

        def objective(trial: optuna.Trial):
            trial_cfg = OmegaConf.create(OmegaConf.to_container(flat_cfg, resolve=True))
            _apply_optuna_suggestions(trial_cfg, trial, param_map)
            trial_cfg.wandb.mode = "disabled"  # disable WandB inside optimisation trials
            trial_cfg.optuna.max_epochs = flat_cfg.optuna.get("max_epochs", 3)
            metrics = _train_eval(trial_cfg, trial=trial, verbose=False)
            return metrics["best_val_acc"]

        study.optimize(objective, n_trials=int(flat_cfg.optuna.n_trials), gc_after_trial=True)
        print(f"Optuna best params: {study.best_params}")

        # Update flattened cfg with best params for final training ----------
        for key, val in study.best_params.items():
            if key == "adapter_reduction_factor":
                flat_cfg.model.task_adapters.reduction_factor = val
            elif key in {"learning_rate", "batch_size", "weight_decay"}:
                flat_cfg.training[key] = val

    # --------------------- Final full training ---------------------------
    _train_eval(flat_cfg, trial=None, verbose=True)


if __name__ == "__main__":
    main()