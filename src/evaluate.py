"""Independent evaluation & figure generation for the collected WandB runs.

Fixes: ensure all metrics referenced in plotting exist. When a run does not
contain ``train_acc`` or other expected columns they are skipped gracefully.
"""

import json
import sys
from pathlib import Path
from typing import Dict, List

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb

FIGURE_DIR = "images"
DATA_DIR = "wandb_data"

# ---------------------------------------------------------------------------
# CLI helper
# ---------------------------------------------------------------------------

def _parse_results_dir(argv) -> Path:
    """Extract results_dir either from key=value or first positional argument."""
    for arg in argv[1:]:
        if arg.startswith("results_dir="):
            return Path(arg.split("=", 1)[1]).expanduser().absolute()
    if len(argv) > 1 and not argv[1].startswith("-"):
        return Path(argv[1]).expanduser().absolute()
    raise ValueError("results_dir argument is required (e.g., results_dir=outputs/)")


# ---------------------------------------------------------------------------
# WandB helpers
# ---------------------------------------------------------------------------

def _fetch_runs(entity: str, project: str):
    api = wandb.Api()
    return api.runs(f"{entity}/{project}")


def _export_metrics(runs, out_dir: Path):
    """Save per-run history CSV & summary JSON. Returns summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)
    summaries: Dict[str, Dict] = {}

    needed_cols: List[str] = [
        "epoch",
        "best_val_acc",
        "best_val_f1",
        "train_loss",
        "val_loss",
        "train_acc",
        "val_acc",
        "train_f1",
        "val_f1",
        "inference_time",
    ]
    for run in runs:
        df = run.history(keys=needed_cols, pandas=True)
        df.to_csv(out_dir / f"run_{run.name}_metrics.csv", index=False)

        # Summary per run ---------------------------------------------------
        summaries[run.name] = {
            "best_val_acc": float(df["best_val_acc"].dropna().max()) if "best_val_acc" in df else None,
            "best_val_f1": float(df["best_val_f1"].dropna().max()) if "best_val_f1" in df else None,
        }
    with open(out_dir / "summary.json", "w") as f:
        json.dump(summaries, f, indent=2)
    return summaries


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_learning_curves(runs, dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    for run in runs:
        df = run.history(keys=["epoch", "train_loss", "val_loss", "train_acc", "val_acc"], pandas=True)
        if df.empty or "epoch" not in df:
            continue

        # Loss curve -------------------------------------------------------
        fig, ax = plt.subplots(figsize=(6, 4))
        if "train_loss" in df and not df["train_loss"].isnull().all():
            ax.plot(df["epoch"], df["train_loss"], label="Train Loss")
        if "val_loss" in df and not df["val_loss"].isnull().all():
            ax.plot(df["epoch"], df["val_loss"], label="Val Loss")
        ax.set_title(f"Loss Curve – {run.name}")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss")
        ax.legend()
        fig.tight_layout()
        fname = dest / f"loss_curve_{run.name}.pdf"
        fig.savefig(fname)
        print(fname.name)
        plt.close(fig)

        # Accuracy curve ---------------------------------------------------
        if ("train_acc" in df and not df["train_acc"].isnull().all()) or (
            "val_acc" in df and not df["val_acc"].isnull().all()
        ):
            fig, ax = plt.subplots(figsize=(6, 4))
            if "train_acc" in df and not df["train_acc"].isnull().all():
                ax.plot(df["epoch"], df["train_acc"], label="Train Acc")
            if "val_acc" in df and not df["val_acc"].isnull().all():
                ax.plot(df["epoch"], df["val_acc"], label="Val Acc")
            ax.set_title(f"Accuracy Curve – {run.name}")
            ax.set_xlabel("Epoch")
            ax.set_ylabel("Accuracy")
            ax.legend()
            fig.tight_layout()
            fname = dest / f"acc_curve_{run.name}.pdf"
            fig.savefig(fname)
            print(fname.name)
            plt.close(fig)


def _plot_comparison(summary: Dict[str, Dict], dest: Path):
    dest.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame.from_dict(summary, orient="index").reset_index().rename(columns={"index": "run_id"})
    if df["best_val_acc"].isnull().all():
        return
    fig, ax = plt.subplots(figsize=(max(6, len(df) * 0.8), 4))
    sns.barplot(data=df, x="run_id", y="best_val_acc", ax=ax, palette="viridis")
    ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha="right")
    ax.set_title("Best Validation Accuracy Across Runs")
    for p in ax.patches:
        ax.annotate(f"{p.get_height():.2f}", (p.get_x() + p.get_width() / 2.0, p.get_height()),
                    ha="center", va="center", xytext=(0, 5), textcoords="offset points")
    fig.tight_layout()
    fname = dest / "comparison_best_val_acc.pdf"
    fig.savefig(fname)
    print(fname.name)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Improvement computation
# ---------------------------------------------------------------------------

def _compute_improvement(summary: Dict[str, Dict]):
    records: Dict[str, Dict[str, float]] = {}
    for run_id, metrics in summary.items():
        # heuristic: dataset name is last token after last '-'
        dataset_name = run_id.split("-")[-1]
        method = "proposed" if run_id.startswith("proposed") else "baseline"
        records.setdefault(dataset_name, {})[method] = metrics["best_val_acc"]

    improvement = {
        ds: {
            **vals,
            "improvement_rate": ((vals.get("proposed") - vals.get("baseline")) / vals.get("baseline")) if vals.get("baseline") else None,
        }
        for ds, vals in records.items()
    }
    return improvement


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    results_dir = _parse_results_dir(sys.argv)
    cfg_path = results_dir / "config.yaml"
    if not cfg_path.exists():
        raise FileNotFoundError(f"Config file not found at {cfg_path}")

    import yaml
    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    entity = cfg["wandb"]["entity"]
    project = cfg["wandb"]["project"]

    runs = _fetch_runs(entity, project)
    summary = _export_metrics(runs, results_dir / DATA_DIR)

    _plot_learning_curves(runs, results_dir / FIGURE_DIR)
    _plot_comparison(summary, results_dir / FIGURE_DIR)

    improvement = _compute_improvement(summary)
    with open(results_dir / DATA_DIR / "improvement.json", "w") as f:
        json.dump(improvement, f, indent=2)

    print("Figures and data exported successfully.")
