#!/usr/bin/env python3
"""Plot the three figures the paper's argument rests on.

    python scripts/make_figures.py            # writes figures/*.png and *.pdf

`PAPER_OUTLINE.md` §9 lists ten figures. Three of them -- 4, 5 and 7 -- carry
claims that nothing else in the paper carries, and all three had their data
committed and none of them plotted. This plots exactly those three, from the
committed artifacts and nothing else, so a clone reproduces them without the
game's content, a checkpoint or a GPU:

* **Figure 4**, the generalisation curve, from `learning_curve.csv`. The claim
  is that most of the learning is early -- 82% of held-out tracks finished by
  257k gradient steps and roughly flat after -- which is what makes 300k the
  recommended budget rather than 1.22M.
* **Figure 5**, out-of-support distance along the lap, from the per-track
  violation series. The claim is §6.2's: official tracks leave the policy's
  observation support **at the first frame**, and unseen tracks from the
  training distribution do not. The control band is the point of the figure,
  not decoration -- without it a reader cannot tell a tight band from a real
  distance.
* **Figure 7**, Summer 1 progress against gradient steps, from the separator
  runs. The claim is that the two arms disagree depending on when you look:
  the pretrained arm is further round the lap through the middle of training
  and reaches its first lap later.

Deliberately plain: one panel per claim, no styling that a greyscale print
would lose, and every series labelled with the artifact it came from. Both PNG
(for reading) and PDF (for LaTeX) are written.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "figures"
GEN = ROOT / "arms" / "generalisation"
SUPPORT = GEN / "support"
SEP = ROOT / "arms" / "separator"

OFFICIAL = ("Winter_5", "Desert_3", "Summer_1", "Summer_5")


def number(value: str) -> float:
    """A CSV cell as a float, with blanks becoming NaN rather than an error.

    `learning_curve.csv` leaves a cell empty where the metric had no value at
    that row -- a greedy fraction before the first greedy episode, a median
    finish time before the first finish. Dropping those rows would shift every
    later point leftward on the x axis; NaN leaves a gap in the line, which is
    what actually happened.
    """
    return float(value) if value not in ("", None) else float("nan")


def save(fig, stem: str) -> None:
    OUT.mkdir(exist_ok=True)
    for suffix in ("png", "pdf"):
        path = OUT / f"{stem}.{suffix}"
        fig.savefig(path, dpi=200, bbox_inches="tight")
        print(f"  wrote {path.relative_to(ROOT)}")
    plt.close(fig)


def figure_1() -> None:
    """The architecture, and where each oracle attaches.

    Drawn rather than photographed: the point of the figure is which box each
    artifact comes from -- the game's own module and its captured geometry on
    one side, our code on the other -- and where the two validation checks
    attach, which a screenshot could not show.
    """
    fig, ax = plt.subplots(figsize=(7.2, 4.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(-0.7, 6.4)
    ax.axis("off")

    def box(x, y, w, h, text, face, edge="0.3", size=8.5, weight="normal"):
        ax.add_patch(plt.Rectangle((x, y), w, h, facecolor=face,
                                   edgecolor=edge, lw=1.1, zorder=2))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=size, zorder=3, weight=weight)

    def arrow(x1, y1, x2, y2, style="-|>", colour="0.35", dashed=False):
        ax.annotate("", xy=(x2, y2), xytext=(x1, y1), zorder=1,
                    arrowprops=dict(arrowstyle=style, color=colour, lw=1.1,
                                    linestyle="--" if dashed else "-",
                                    shrinkA=2, shrinkB=2))

    GAME, OURS, CHECK = "#dfe8f3", "#eef0f2", "#f6e7e2"

    ax.text(2.3, 6.15, "from the shipped client", fontsize=8.5,
            style="italic", color="0.35", ha="center")
    box(0.3, 5.1, 4.0, 0.8, "app.asar  (PolyTrack 0.6.2)", GAME, size=9,
        weight="bold")
    box(0.3, 4.0, 1.9, 0.8, "physics\n.wasm", GAME)
    box(2.4, 4.0, 1.9, 0.8, "collision geometry\n(captured)", GAME)
    arrow(1.25, 5.1, 1.25, 4.8)
    arrow(3.35, 5.1, 3.35, 4.8)

    ax.text(7.6, 6.15, "track codes", fontsize=8.5, style="italic",
            color="0.35", ha="center")
    box(5.6, 5.1, 1.9, 0.8, "official\ntracks", GAME)
    box(7.7, 5.1, 2.0, 0.8, "generator\n(route by\nconstruction)", OURS, size=7.5)

    box(0.3, 2.6, 4.0, 0.9, "wasmtime host in Python\n(no browser, no IPC)",
        OURS, size=9, weight="bold")
    arrow(2.3, 4.0, 2.3, 3.5)

    box(5.6, 2.6, 4.1, 0.9, "PolyTrackEnv\n62-float observation, 9 actions",
        OURS, size=9, weight="bold")
    arrow(4.3, 3.05, 5.6, 3.05)
    arrow(6.55, 5.1, 6.55, 3.5)
    arrow(8.7, 5.1, 8.7, 3.5)

    box(5.6, 1.2, 4.1, 0.8, "IQN policy  +  guided beam search", OURS, size=9)
    arrow(7.65, 2.6, 7.65, 2.0)
    arrow(7.0, 2.0, 7.0, 2.6)

    box(0.3, 1.2, 4.0, 0.8, "acceptance gate\n10 human laps, 5 tracks", CHECK,
        size=8.5)
    box(0.3, 0.1, 4.0, 0.8, "the client's own verifier\n(shipped worker, Node)",
        CHECK, size=8.5)
    arrow(2.3, 2.6, 2.3, 2.0)
    arrow(4.3, 1.2, 5.6, 1.6, dashed=True)
    arrow(5.6, 1.35, 4.3, 0.65, dashed=True)

    ax.text(2.3, -0.42, "both checks execute the same physics module",
            fontsize=7.5, style="italic", color="0.45", ha="center")

    save(fig, "fig1_architecture")


def figure_4() -> None:
    """Held-out finish rate and greedy lap fraction against gradient steps."""
    rows = list(csv.DictReader((GEN / "learning_curve.csv").open()))
    steps = np.array([number(r["gradient_steps"]) for r in rows]) / 1000.0
    finish = np.array([number(r["finish_rate"]) for r in rows]) * 100.0
    greedy = np.array([number(r["greedy_fraction"]) for r in rows]) * 100.0

    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ax.plot(steps, greedy, lw=1.2, color="0.55", label="greedy lap fraction")
    ax.plot(steps, finish, lw=1.8, color="C0", label="finish rate")
    # The dashed line is the three-seed budget the paper recommends, and
    # the annotation is why: the curve is already flat well before it.
    ax.axvline(300, color="C3", lw=1.0, ls="--")
    ax.annotate("300k: the three-seed budget\n(82% already by 257k)",
                xy=(300, 86), xytext=(380, 52), fontsize=8, color="C3",
                arrowprops=dict(arrowstyle="->", color="C3", lw=0.8))
    ax.set_xlabel("gradient steps (thousands)")
    ax.set_ylabel("per cent")
    ax.set_ylim(0, 105)
    ax.set_xlim(0, steps.max())
    ax.set_title("Figure 4  Generalisation to 16 held-out generated tracks",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.text(0.01, -0.06, "arms/generalisation/learning_curve.csv, 121 points, "
             "one 1.22M-step run", fontsize=6.5, color="0.4")
    save(fig, "fig4_generalisation_curve")


def figure_5() -> None:
    """Features out of support against distance along the lap."""
    # Against elapsed *steps*, not arc. An attempt that stalls spends many
    # frames at nearly the same arc, so an arc axis compresses the whole
    # stalled tail against the right-hand edge and reads as a late spike for
    # every track -- which is an artifact of the axis, not of the policy.
    def series(name: str):
        rows = list(csv.DictReader((SUPPORT / f"violations_{name}.csv").open()))
        oos = np.array([int(r["features_out_of_support"]) for r in rows])
        return np.linspace(0, 100, len(oos)), oos

    control = sorted(
        p.stem.removeprefix("violations_")
        for p in SUPPORT.glob("violations_Generated_*.csv")
    )

    fig, ax = plt.subplots(figsize=(6.4, 3.8))

    # The control band: 16 unseen tracks from the TRAINING distribution, all
    # finished. Plotted as the envelope of their per-frame counts, because the
    # claim is that the whole family sits inside the band.
    grid = np.linspace(0, 100, 200)
    stack = np.array([
        np.interp(grid, *series(name)) for name in control
    ])
    ax.fill_between(grid, stack.min(axis=0), stack.max(axis=0),
                    color="0.75", alpha=0.6, lw=0,
                    label=f"control: {len(stack)} unseen generated tracks, "
                          "all finished (mean 0.05)")

    for name, colour in zip(OFFICIAL, ("C0", "C1", "C2", "C3")):
        steps, oos = series(name)
        half = oos[: len(oos) // 2].mean()
        ax.plot(steps, oos, lw=1.4, color=colour,
                label=f"{name.replace('_', ' ')}: mean {oos.mean():.1f}, "
                      f"first half {half:.1f}")

    ax.set_xlabel("elapsed share of the attempt, per cent of its control steps")
    ax.set_ylabel("features out of support, of 62")
    ax.set_xlim(0, 100)
    ax.set_ylim(bottom=0)
    ax.set_title("Figure 5  Official tracks sit outside the policy's training "
                 "support", fontsize=10)
    ax.legend(fontsize=7, loc="upper left", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.text(0.01, -0.07, "arms/generalisation/support/violations_*.csv, pass 2. "
             "Band = 0.1st-99.9th percentile of each feature over 24,545 "
             "training frames.", fontsize=6.5, color="0.4")
    save(fig, "fig5_out_of_support")


def figure_7() -> None:
    """Summer 1 progress against gradient steps, both arms, three seeds."""
    arms = {
        "scratch": ("sep-g099-1", "xlong-scratch-2", "xlong-scratch-3"),
        "pretrained": ("xlong-pre-1", "xlong-pre-2", "xlong-pre-3"),
    }
    first_lap = {"scratch": (121, 234, 252), "pretrained": (284, 246, 305)}

    fig, ax = plt.subplots(figsize=(6.4, 3.8))
    for arm, runs, colour in (("scratch", arms["scratch"], "C0"),
                              ("pretrained", arms["pretrained"], "C1")):
        for index, run in enumerate(runs):
            history = json.loads((SEP / run / "history.json").read_text())
            steps = np.array([r["gradient_steps"] for r in history]) / 1000.0
            frac = np.array([r["greedy_fraction"] for r in history]) * 100.0
            # Binned at 10k steps rather than smoothed: the raw column is
            # one near-greedy collector's recent episodes, so it is noisy by
            # construction, and a bin keeps the x axis honest about where each
            # point came from.
            edges = np.arange(0, 410, 10)
            mids = (edges[:-1] + edges[1:]) / 2
            binned = np.array([
                frac[(steps >= lo) & (steps < hi)].mean()
                if ((steps >= lo) & (steps < hi)).any() else np.nan
                for lo, hi in zip(edges[:-1], edges[1:])
            ])
            ax.plot(mids, binned, lw=1.1, color=colour, alpha=0.85,
                    label=f"{arm} (3 seeds)" if index == 0 else None)
        for step in first_lap[arm]:
            ax.plot([step], [2], marker="D", ms=4.5, color=colour,
                    clip_on=False)

    ax.set_xlabel("gradient steps (thousands)")
    ax.set_ylabel("share of the lap reached, per cent")
    ax.set_xlim(0, 400)
    ax.set_ylim(0, 105)
    ax.set_title("Figure 7  Summer 1: an official track is learnable, and the "
                 "arms disagree", fontsize=10)
    ax.legend(fontsize=8, loc="lower right", frameon=False)
    ax.grid(alpha=0.25, lw=0.5)
    fig.text(0.01, -0.07, "arms/separator/*/history.json, 400k steps, gamma 0.99. "
             "Diamonds mark each run's first completed lap. Curves are "
             "binned at 10k steps.", fontsize=6.5, color="0.4")
    save(fig, "fig7_summer1_progress")


def main() -> int:
    print(f"\n  writing to {OUT.relative_to(ROOT)}/\n")
    figure_1()
    figure_4()
    figure_5()
    figure_7()
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
