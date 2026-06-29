from __future__ import annotations

from pathlib import Path

import pandas as pd

from campaign_optimizer.io import read_table


def write_basic_plots(iter_dir: Path) -> list[Path]:
    """Write optional static plots from reduced optimizer tables.

    This is best-effort. Plotting failures should not block optimizer artifact
    creation.
    """
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return []

    outputs: list[Path] = []
    plot_dir = iter_dir / "outputs" / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)

    objective_path = iter_dir / "inputs" / "objective_table.csv"

    if objective_path.exists():
        df = read_table(objective_path)

        for score in ["score_guiding_v1", "score_beamlike_v1", "score_transverse_v1"]:
            if score not in df.columns:
                continue

            values = pd.to_numeric(df[score], errors="coerce").dropna()

            if values.empty:
                continue

            fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
            ax.hist(values.to_numpy(), bins=min(20, max(5, len(values) // 5)))
            ax.set_xlabel(score)
            ax.set_ylabel("count")
            ax.set_title(f"Observed {score}")

            path = plot_dir / f"observed_{score}.png"
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            outputs.append(path)

    rec_path = iter_dir / "outputs" / "recommended_candidates.tsv"

    if rec_path.exists():
        rec = read_table(rec_path, sep="\t")

        if "rank" in rec.columns and "acquisition_value" in rec.columns:
            fig, ax = plt.subplots(figsize=(6, 4), dpi=140)
            ax.plot(
                pd.to_numeric(rec["rank"]),
                pd.to_numeric(rec["acquisition_value"], errors="coerce"),
                marker="o",
            )
            ax.set_xlabel("rank")
            ax.set_ylabel("acquisition_value")
            ax.set_title("Recommended candidates")

            path = plot_dir / "recommended_candidates_rank.png"
            fig.tight_layout()
            fig.savefig(path)
            plt.close(fig)
            outputs.append(path)

    return outputs
