"""Figures for the report (PNG and PDF in outputs/results/figures/). Called by analyze.py; needs matplotlib.

Colours: a colour-blind-checked categorical palette in a fixed order. Gender is deliberately not coded
blue/pink, which would echo the very stereotype H2 measures. Every number is also in the tables.
"""
from __future__ import annotations

from collections import Counter, defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from common import LLMS, RESULTS, SYSTEMS, human_baseline, register_group, wilson  # noqa: E402

BLUE, ORANGE, AQUA, YELLOW, VIOLET = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#6250d6"
GRAY, INK, INK2, MUTED, GRID, AXIS, SURFACE = "#c3c2b7", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7", "#ffffff"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"], "font.size": 8.5,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.spines.top": False, "axes.spines.right": False,
    "xtick.color": MUTED, "ytick.color": INK2, "xtick.labelcolor": INK2, "axes.titlesize": 9,
    "axes.titleweight": "bold", "axes.titlecolor": INK, "legend.frameon": False, "savefig.dpi": 300,
})
OUT = RESULTS / "figures"


def save(fig, name: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "pdf"):
        fig.savefig(OUT / f"{name}.{ext}", bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)
    print(f"figure -> {OUT / name}.png")


def stacked_panel(ax, names, shares, labels, colours, title):
    """100% stacked horizontal bars with a white gap between segments and direct labels on big segments."""
    for i, name in enumerate(names):
        left = 0.0
        for share_, colour in zip(shares[i], colours):
            if share_ > 0:
                ax.barh(i, share_, left=left, color=colour, edgecolor=SURFACE, linewidth=1.2, height=0.7)
                if share_ >= 0.08:
                    text_colour = INK if colour in (YELLOW, GRAY, AQUA) else "#ffffff"
                    ax.text(left + share_ / 2, i, f"{100 * share_:.0f}", ha="center", va="center",
                            fontsize=7, color=text_colour)
            left += share_
    ax.set_yticks(range(len(names)), names)
    if not ax.yaxis_inverted():   # the panels share the y axis: invert it once, first row on top
        ax.invert_yaxis()
    ax.set_xlim(0, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100%"])
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_title(title, loc="left", pad=20)
    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for c in colours]
    ax.legend(handles, labels, ncol=len(labels), loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5,
              handlelength=1, columnspacing=1, borderaxespad=0.2)


def fig_h1(rows: list[dict]) -> None:
    core = [r for r in rows if r["split"] == "core" and r["condition"] == "none"]
    groups = [(SYSTEMS[s], [r for r in core if r["system"] == s]) for s in SYSTEMS]
    groups = [("Humans (Tatoeba)", human_baseline())] + [(n, g) for n, g in groups if g]
    names = [n for n, _ in groups]
    reg_shares, gen_shares = [], []
    for _, g in groups:
        n = len(g)
        reg = Counter(register_group(r["register"]) for r in g)
        gen = Counter(r["gender"] for r in g)
        reg_shares.append([reg["tu"] / n, reg["tum"] / n, reg["aap"] / n, 1 - (reg["tu"] + reg["tum"] + reg["aap"]) / n])
        gen_shares.append([gen["M"] / n, gen["F"] / n, 1 - (gen["M"] + gen["F"]) / n])
    fig, (a, b) = plt.subplots(1, 2, figsize=(7.0, 2.9), sharey=True, gridspec_kw={"wspace": 0.08})
    stacked_panel(a, names, reg_shares, ["tu", "tum", "aap", "other"], [BLUE, ORANGE, AQUA, GRAY], "Register")
    stacked_panel(b, names, gen_shares, ["masculine", "feminine", "no gender marked"], [VIOLET, YELLOW, GRAY], "Gender")
    for ax in (a, b):
        ax.axhline(0.5, color=AXIS, linewidth=0.8)   # the human reference row above, the systems below
    save(fig, "h1_defaults")


def fig_h2(rows: list[dict], items: dict) -> None:
    frames: dict[str, dict[str, str]] = defaultdict(dict)
    for it in items.values():
        if it.get("frame_id"):
            frames[it["frame_id"]][it["cue"]] = it["id"]
    plain = {(r["system"], r["id"]): r for r in rows if r["condition"] == "none"}
    cues = [("neutral", "neutral", BLUE), ("fem_stereotype", "female-stereotyped", ORANGE),
            ("masc_stereotype", "male-stereotyped", AQUA)]
    systems = [s for s in SYSTEMS if any((s, f["fem_stereotype"]) in plain for f in frames.values())]
    fig, ax = plt.subplots(figsize=(5.2, 0.32 * len(systems) + 0.9))
    for j, (cue, label, colour) in enumerate(cues):
        for i, s in enumerate(systems):
            got = [plain[(s, f[cue])] for f in frames.values() if (s, f.get(cue)) in plain]
            k, n = sum(r["gender"] == "F" for r in got), len(got)
            lo, hi = wilson(k, n)
            y = i + (j - 1) * 0.22
            ax.plot([lo, hi], [y, y], color=colour, linewidth=1.2, alpha=0.6, solid_capstyle="round")
            ax.plot(k / n, y, "o", color=colour, markersize=5, markeredgecolor=SURFACE, markeredgewidth=0.8,
                    label=label if i == 0 else None)
    ax.set_yticks(range(len(systems)), [SYSTEMS[s] for s in systems])
    ax.invert_yaxis()
    ax.set_xlim(-0.02, 1)
    ax.set_xticks([0, 0.25, 0.5, 0.75, 1], ["0", "25", "50", "75", "100%"])
    ax.grid(axis="x", color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0)
    ax.spines["left"].set_visible(False)
    ax.set_xlabel(f"feminine outputs (95% CI), {len(frames)} frames")
    ax.legend(ncol=3, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5, handletextpad=0.3)
    save(fig, "h2_stereotypes")


def fig_h3(rows: list[dict]) -> None:
    llm = [r for r in rows if r["system"] in LLMS and r["split"] == "core"]
    by = defaultdict(list)
    for r in llm:
        by[(r["system"], r["condition"])].append(r)
    present = [s for s in LLMS if by[(s, "register")] and by[(s, "gender")] and by[(s, "both")]]
    if not present:
        print("(H3 figure skipped: no LLM has all three conditions yet)")
        return
    tests = [("Register as requested", "register", lambda r: r["register"] == r["requested_register"]),
             ("Gender as requested", "gender", lambda r: r["gender"] == r["requested_gender"])]
    # (share, low, high) per panel, LLM and condition; both panels then share one x scale
    stats = {}
    for title, alone, ok in tests:
        for s in present:
            for cond in (alone, "both"):
                sub = by[(s, cond)]
                k, n = sum(map(ok, sub)), len(sub)
                stats[(title, s, cond)] = (k / n, *wilson(k, n))
    xmin = max(0.0, min(v[1] for v in stats.values()) - 0.03)
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 0.5 * len(present) + 1.0), sharey=True, gridspec_kw={"wspace": 0.1})
    for ax, (title, alone, ok) in zip(axes, tests):
        for i, s in enumerate(present):
            for offset, cond, colour, label in ((-0.14, alone, BLUE, "asked alone"),
                                                (0.14, "both", ORANGE, "asked together with the other attribute")):
                p, lo, hi = stats[(title, s, cond)]
                ax.plot([lo, hi], [i + offset] * 2, color=colour, linewidth=1.2, alpha=0.55, solid_capstyle="round")
                ax.plot(p, i + offset, "o", color=colour, markersize=5.5, markeredgecolor=SURFACE,
                        markeredgewidth=0.8, label=label if i == 0 else None, zorder=3)
            change = 100 * (stats[(title, s, "both")][0] - stats[(title, s, alone)][0])
            ax.text(1.005, i, f"{change:+.1f}".replace("-", "−"), transform=ax.get_yaxis_transform(),
                    va="center", ha="left", fontsize=7.5, color=INK2)
        ax.set_title(title, loc="left", pad=20)
        ax.set_xlim(xmin, 1.0)
        ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(0.05))
        ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(lambda x, _: f"{100 * x:.0f}%"))
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(present)), [SYSTEMS[s] for s in present])
    axes[0].set_ylim(len(present) - 0.5, -0.5)   # first LLM on top
    axes[0].legend(ncol=2, loc="lower left", bbox_to_anchor=(0, 1.0), fontsize=7.5, handletextpad=0.3,
                   borderaxespad=0.2)
    axes[1].text(1.0, 1.0, "change\n(points)", transform=axes[1].transAxes, ha="left", va="bottom",
                 fontsize=7, color=MUTED)
    save(fig, "h3_interference")


def fig_h4(pairs: list[dict], duels: list[dict], scorer, metrics: list[str]) -> None:
    """One panel per metric (different scales never share an axis): the mean score of a translation that is
    perfect except for the gender / register, next to real MT outputs that have both right."""
    from statistics import mean
    rows = [("wrong gender only", "gender", ORANGE), ("wrong register only", "register", ORANGE),
            ("both wrong", "both", ORANGE)]
    fig, axes = plt.subplots(1, len(metrics), figsize=(2.35 * len(metrics), 1.9), sharey=True,
                             gridspec_kw={"wspace": 0.12})
    axes = [axes] if len(metrics) == 1 else axes
    labels = [r[0] for r in rows] + ["right attributes,\nown wording (MT)"]
    for ax, m in zip(axes, metrics):
        values = [mean(scorer[p["key"]][m] for p in pairs if p["error"] == e) for _, e, _ in rows]
        values.append(mean(scorer[d["output"]][m] for d in duels if d["error"] == "gender"))
        colours = [c for *_, c in rows] + [BLUE]
        ax.barh(range(len(values)), values, color=colours, height=0.6, edgecolor=SURFACE, linewidth=1)
        for i, v in enumerate(values):
            ax.text(v + 1.5, i, f"{v:.0f}", va="center", ha="left", fontsize=7, color=INK2)
        ax.set_xlim(0, 112)
        ax.set_xticks([0, 50, 100])
        ax.set_title(m, loc="left")
        ax.grid(axis="x", color=GRID, linewidth=0.6)
        ax.set_axisbelow(True)
        ax.tick_params(axis="y", length=0)
        ax.spines["left"].set_visible(False)
    axes[0].set_yticks(range(len(labels)), labels)
    axes[0].set_ylim(len(labels) - 0.5, -0.5)
    fig.text(0.5, -0.12, "mean sentence score against the reference (a perfect translation scores 100)",
             ha="center", fontsize=7.5, color=INK2)
    save(fig, "h4_metrics")


def main(rows: list[dict], items: dict) -> None:
    fig_h1(rows)
    fig_h2(rows, items)
    fig_h3(rows)
