"""
pnn_viz.py — Photonic Neural Network Visualization Suite
=========================================================
Call  save_all_figures(...)  after training to produce the full figure set.

Figures produced
----------------
  01_architecture.png       — Waveguide mesh architecture with learned phase heatmaps
  02_training_curves.png    — Loss + accuracy per layer across epochs
  03_confusion_matrix.png   — Per-class confusion matrix (test set, vote predictions)
  04_feature_space.png      — 2-D PCA projection of final-layer features, coloured by class
  05_cosine_similarity.png  — Mean cosine similarity to each anchor per true class (heatmap)
  06_energy_conservation.png— Per-layer output intensity distribution (energy check)
  07_phase_rose.png         — Polar histogram of learned theta / phi phases per layer
  08_composite.png          — Paper-ready composite of key panels
"""
import sys
sys.path.append("../")
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from sklearn.decomposition import PCA
import tensorflow as tf


# ── Palette ──────────────────────────────────────────────────────────────────
# ── Dark theme (default) ────────────────────────────────────────────────────
BG       = "#05070f"
PANEL    = "#0b0f1e"
GRID     = "#1a1f35"
ACCENT   = "#00d4ff"
ACCENT2  = "#ff6b35"
WHITE    = "#e8eeff"
DIM      = "#4a5280"

CLASS_COLORS = ["#00d4ff","#ff00fb","#0800ff","#ffcc00","#d0ff00","#f50505", "#ff6b35", "#a8ff3e", "#ff3e8a", "#c97bff"]

_DARK_THEME  = dict(BG=BG, PANEL=PANEL, GRID=GRID, ACCENT=ACCENT,
                    ACCENT2=ACCENT2, WHITE=WHITE, DIM=DIM)
_LIGHT_THEME = dict(
    BG      = "#ffffff",
    PANEL   = "#f4f6fb",
    GRID    = "#d0d6e8",
    ACCENT  = "#0077bb",   # strong blue — readable on white
    ACCENT2 = "#cc4400",   # dark orange
    WHITE   = "#111111",   # "white" role → dark text on light bg
    DIM     = "#555577",
)

import sys as _sys

@tf.function
def majority_vote_tiebreak(preds_early, last_scores):
    C         = tf.shape(last_scores)[1]
    votes = tf.reduce_sum(tf.one_hot(preds_early, depth=C, dtype=tf.int32), axis=1)
    max_count = tf.reduce_max(votes, axis=1, keepdims=True)
    tied      = tf.equal(votes, max_count)
    n_tied    = tf.reduce_sum(tf.cast(tied, tf.int32), axis=1)
    vote_winner = tf.argmax(votes, axis=1, output_type=tf.int32)
    neg_inf      = tf.constant(-1e9, dtype=last_scores.dtype)
    masked_scores = tf.where(tied, last_scores, neg_inf)
    tie_winner   = tf.argmax(masked_scores, axis=1, output_type=tf.int32)
    return tf.where(n_tied > 1, tie_winner, vote_winner)


# @tf.function(reduce_retracing=True)
def forward_all_cos_sims(model, xb, refs):
    L = len(model.all_layers)
    x = xb
    cos_sims_list = []

    for i, layer in enumerate(model.all_layers):
        if i < L - 1:
            x_nl = layer(x, training=False)
            
            E_lin = layer.forward_linear(x)          # call ONCE, reuse
            ff = tf.math.real(E_lin * tf.math.conj(E_lin))
        else:
            ff   = layer.forward_ff(x)
            x_nl = x

        norm    = tf.math.l2_normalize(ff, axis=1)
        cos_sim = tf.matmul(norm, refs[i], transpose_b=True)
        cos_sims_list.append(cos_sim)

        if i < L - 1:
            x = x_nl

    cos_sims    = tf.stack(cos_sims_list)
    preds_early = tf.argmax(cos_sims[:-1], axis=2, output_type=tf.int32)
    preds_early = tf.transpose(preds_early, perm=[1, 0])
    last_scores = cos_sims[-1]
    vote_pred   = majority_vote_tiebreak(preds_early, last_scores)
    return cos_sims, vote_pred


def set_theme(theme="dark"):
    """Switch all global colour constants to 'dark' or 'light'."""
    t = _DARK_THEME if theme == "dark" else _LIGHT_THEME
    mod = _sys.modules[__name__]
    for k, v in t.items():
        globals()[k] = v
        setattr(mod, k, v)
CLASS_NAMES  = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

_PHOTON = LinearSegmentedColormap.from_list(
    "photon", ["#05070f", "#001833", "#0057a8", "#00d4ff", "#ffffff"], N=256
)
_PHASE = LinearSegmentedColormap.from_list(
    "phase", ["#ff3e8a", "#1a0533", "#05070f", "#001833", "#00d4ff"], N=256
)

def _fig(w=12, h=8):
    fig = plt.figure(figsize=(w, h), facecolor=BG)
    return fig

def _ax(fig, *args, **kw):
    ax = fig.add_subplot(*args, **kw)
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=WHITE, labelsize=14)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    return ax

def _style_ax(ax):
    ax.set_facecolor(PANEL)
    for spine in ax.spines.values():
        spine.set_edgecolor(GRID)
    ax.tick_params(colors=WHITE, labelsize=14)
    ax.xaxis.label.set_color(WHITE)
    ax.yaxis.label.set_color(WHITE)
    ax.title.set_color(WHITE)
    return ax

def _label(ax, txt, size=11, color=WHITE, x=0.01, y=0.97):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=size,
            color=color, va="top", ha="left", fontfamily="monospace")


def _save(fig, path, dpi=120):
    # Cap figure size so renderer never exceeds ~8000px on longest side.
    # MemoryError: std::bad_alloc means the canvas is too large.
    MAX_PX = 8000
    w, h   = fig.get_size_inches()
    max_in = max(w, h)
    if max_in * dpi > MAX_PX:
        dpi = int(MAX_PX / max_in)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=BG)
    plt.close(fig)
    print(f"  ✓  saved → {path}  ({int(w*dpi)}×{int(h*dpi)}px @ {dpi}dpi)")


def _normalise_history(history, n_layers):
    """
    Accept either:
      - a plain list  [v0, v1, ...]          → one curve, treated as layer 0
      - a dict        {0: [...], 1: [...]}    → one curve per layer key
      - a list of lists [[...], [...]]        → one curve per inner list

    Always returns a dict {layer_idx: [values]}.
    """
    if isinstance(history, dict):
        return history
    if isinstance(history, list):
        # list of scalars → single layer
        if len(history) == 0:
            return {0: []}
        if not isinstance(history[0], (list, np.ndarray)):
            return {0: history}
        # list of lists → one per layer
        return {i: h for i, h in enumerate(history)}
    return {0: list(history)}


# ═══════════════════════════════════════════════════════════════════════════════
# 01  Architecture
# ═══════════════════════════════════════════════════════════════════════════════
def capture_phases(model):
    """Snapshot current theta/phi for all PNN layers. Call BEFORE training."""
    return [
        {"theta": layer.mesh.theta.numpy().reshape(-1).copy(),
         "phi":   layer.mesh.phi.numpy().reshape(-1).copy()}
        for layer in model.all_layers
    ]


def capture_transfer_matrices(model):
    """
    Snapshot theta/phi phases for every layer before training.
    Used by plot_light_propagation for true column-by-column MZI propagation.
    Returns list of {"theta": np.array, "phi": np.array} dicts.
    """
    return [
        {"theta": layer.mesh.theta.numpy().reshape(-1).copy(),
         "phi":   layer.mesh.phi.numpy().reshape(-1).copy()}
        for layer in model.all_layers
    ]


def plot_architecture(model, out_path="01_architecture.png", phases_before=None):
    """
    Clements mesh heatmaps.  Square 25×25 grid.
    When phases_before is given: [Before | After | Δ] side-by-side per phase.
    Distribution plots are handled by plot_phase_distributions() separately.
    """
    import copy
    N_PORTS  = model.all_layers[0].n_ports
    N_MCOLS  = model.all_layers[0].n_ports
    N_ROWS   = model.all_layers[0].n_ports   # full 25×25 grid; row 24 is intentionally empty
                    # (wg 24 is always the bottom of a pair, never a top-wg)
    MZI_EVEN = list(range(0, N_PORTS - 1, 2))   # top-wg indices: 0,2,...,22
    MZI_ODD  = list(range(1, N_PORTS - 1, 2))   # top-wg indices: 1,3,...,23
    VMIN_T, VMAX_T = 0.0, np.pi
    VMIN_P, VMAX_P = 0.0, 2 * np.pi

    def _make_cmap(base):
        cm = copy.copy(plt.cm.get_cmap(base))
        cm.set_bad(color=PANEL)
        return cm
    cmap_t    = _make_cmap("viridis")
    cmap_p    = _make_cmap("plasma")
    cmap_diff = _make_cmap("RdBu_r")

    def build_grid(phases):
        # Full 25×25 grid; row 24 stays NaN (wg 24 is never a top-wg)
        grid = np.full((N_ROWS, N_MCOLS), np.nan)
        idx  = 0
        for col in range(N_MCOLS):
            pairs = MZI_EVEN if col % 2 == 0 else MZI_ODD
            for wg in pairs:
                if idx < len(phases):
                    grid[wg, col] = phases[idx]
                idx += 1
        return grid

    compare = phases_before is not None
    n_pnn   = len(model.all_layers)

    # Each PNN layer gets 2 rows (θ row, φ row).
    # Each row has either [heat] or [before | after | diff], all square cells.
    # We give each heatmap panel a fixed cell_size inches so cells stay square
    # regardless of n_pnn.
    CELL   = 0.55          # inches per MZI cell — larger cells
    HM_W   = N_MCOLS * CELL   # ~13.75 in
    HM_H   = N_PORTS * CELL   # ~13.75 in
    PAD_H  = 4.5           # vertical padding per row (titles, labels, annotation)
    PAD_W  = 3.5           # per heatmap panel (colorbar + margins)

    if compare:
        n_heat_cols = 3      # before, after, diff
        fig_w = n_heat_cols * (HM_W + PAD_W) + 2.0
    else:
        n_heat_cols = 1
        fig_w = HM_W + PAD_W + 2.0

    row_h = HM_H + PAD_H
    fig_h = n_pnn * 2 * row_h + 2.0   # 2 phase rows per PNN layer

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    title = (f"PNN Mesh  —  Before vs After Training  (Clements, {model.all_layers[0].n_ports}{model.all_layers[0].n_ports})"
             if compare else
             f"PNN Mesh  —  Trained Phases  (Clements, {model.all_layers[0].n_ports}×{model.all_layers[0].n_ports})")
    fig.suptitle(title, color=WHITE, fontsize=88, fontfamily="monospace",
                 fontweight="bold", y=1.002)

    n_rows = n_pnn * 2
    if compare:
        wratios = [HM_W, HM_W, HM_W]
        ncols   = 3
    else:
        wratios = [HM_W]
        ncols   = 1

    gs = gridspec.GridSpec(n_rows, ncols, figure=fig,
                           width_ratios=wratios,
                           hspace=(PAD_H + 0.5) / row_h,
                           wspace=(PAD_W + 0.5) / HM_W,
                           left=0.05, right=0.95,
                           top=0.97, bottom=0.02)

    CB_TICKS = {
        "θ": ([0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
              ["0", "π/4", "π/2", "3π/4", "π"]),
        "φ": ([0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
              ["0", "π/2", "π", "3π/2", "2π"]),
    }

    FSIZE_TITLE = 80
    FSIZE_LABEL = 68
    FSIZE_TICK  = 60
    FSIZE_CB    = 60
    FSIZE_ANNOT = 52

    def _heatmap(ax, grid, cmap, vmin, vmax, title_txt, title_color,
                 show_ylab=True, show_xlab=True):
        _style_ax(ax)
        im = ax.imshow(np.ma.masked_invalid(grid), cmap=cmap,
                       vmin=vmin, vmax=vmax,
                       aspect="equal", interpolation="nearest", origin="upper")
        ax.set_xticks(np.arange(-0.5, N_MCOLS, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, N_ROWS,  1), minor=True)
        ax.grid(which="minor", color=BG, linewidth=0.6)
        ax.tick_params(which="minor", left=False, bottom=False)
        ax.set_xticks(range(0, N_MCOLS, 4))
        ax.set_xticklabels(range(0, N_MCOLS, 4) if show_xlab else [],
                           fontsize=FSIZE_TICK, color=WHITE)
        ax.set_yticks(range(0, N_ROWS, 4))
        ax.set_yticklabels(range(0, N_ROWS, 4) if show_ylab else [],
                           fontsize=FSIZE_TICK, color=WHITE)
        if show_xlab:
            ax.set_xlabel("Mesh column", fontsize=FSIZE_LABEL, color=WHITE, labelpad=4)
        if show_ylab:
            ax.set_ylabel("Waveguide", fontsize=FSIZE_LABEL, color=WHITE, labelpad=4)
        ax.set_title(title_txt, color=title_color, fontsize=FSIZE_TITLE,
                     fontfamily="monospace", fontweight="bold", pad=9)
        return im

    def _colorbar(fig, im, ax, label, ticks, ticklabels, color):
        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02, shrink=1.0)
        cb.set_label(label, color=color, fontsize=FSIZE_CB, labelpad=6)
        cb.set_ticks(ticks)
        cb.set_ticklabels(ticklabels)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE, fontsize=FSIZE_CB)
        cb.ax.yaxis.set_tick_params(color=WHITE, length=8, width=2)
        cb.outline.set_edgecolor(GRID)

    for li, layer in enumerate(model.all_layers):
        theta_a = np.clip(layer.mesh.theta.numpy().reshape(-1), VMIN_T, VMAX_T)
        phi_a   = np.clip(layer.mesh.phi.numpy().reshape(-1),   VMIN_P, VMAX_P)
        if compare:
            theta_b = np.clip(phases_before[li]["theta"], VMIN_T, VMAX_T)
            phi_b   = np.clip(phases_before[li]["phi"],   VMIN_P, VMAX_P)

        specs = [
            ("θ", cmap_t, VMIN_T, VMAX_T, ACCENT,
             theta_a, theta_b if compare else None),
            ("φ", cmap_p, VMIN_P, VMAX_P, ACCENT2,
             phi_a,   phi_b   if compare else None),
        ]

        for pi, (lbl, cmap, vmin, vmax, color, arr_a, arr_b) in enumerate(specs):
            row    = li * 2 + pi
            ticks, tlabels = CB_TICKS[lbl]
            ga     = build_grid(arr_a)
            gb     = build_grid(arr_b) if compare else None

            if compare:
                # col 0 — Before
                ax0 = fig.add_subplot(gs[row, 0])
                im0 = _heatmap(ax0, gb, cmap, vmin, vmax,
                               f"L{li+1}  {lbl}  — init",
                               DIM, show_ylab=True)
                _colorbar(fig, im0, ax0, f"{lbl} (rad)", ticks, tlabels, WHITE)

                # col 1 — After
                ax1 = fig.add_subplot(gs[row, 1])
                im1 = _heatmap(ax1, ga, cmap, vmin, vmax,
                               f"L{li+1}  {lbl}  — trained",
                               color, show_ylab=False)
                _colorbar(fig, im1, ax1, f"{lbl} (rad)", ticks, tlabels, color)

                # col 2 — Diff
                diff = ga - gb
                dlim = max(float(np.nanpercentile(np.abs(diff), 98)), 0.05)
                ax2  = fig.add_subplot(gs[row, 2])
                _style_ax(ax2)
                im2  = ax2.imshow(np.ma.masked_invalid(diff),
                                  cmap=cmap_diff, vmin=-dlim, vmax=dlim,
                                  aspect="equal", interpolation="nearest",
                                  origin="upper")
                ax2.set_xticks(np.arange(-0.5, N_MCOLS, 1), minor=True)
                ax2.set_yticks(np.arange(-0.5, N_ROWS,  1), minor=True)
                ax2.grid(which="minor", color=BG, linewidth=0.6)
                ax2.tick_params(which="minor", left=False, bottom=False)
                ax2.set_xticks(range(0, N_MCOLS, 4))
                ax2.set_xticklabels(range(0, N_MCOLS, 4), fontsize=FSIZE_TICK, color=WHITE)
                ax2.set_yticks([])
                ax2.set_xlabel("Mesh column", fontsize=FSIZE_LABEL, color=WHITE, labelpad=4)
                ax2.set_title(f"L{li+1}  Δ{lbl}  — (trained − init)",
                              color="#a8ff3e", fontsize=FSIZE_TITLE,
                              fontfamily="monospace", fontweight="bold", pad=9)
                n_tot     = int((~np.isnan(diff)).sum())
                n_changed = int(np.sum(np.abs(diff[~np.isnan(diff)]) > 0.05))
                # Place annotation outside the axes, below the x-axis label
                ax2.text(0.5, -0.22,
                         f"|Δ| > 0.05 rad:  {n_changed} / {n_tot}  MZIs",
                         transform=ax2.transAxes, fontsize=FSIZE_ANNOT,
                         color="#a8ff3e", ha="center", va="top",
                         
                         fontfamily="monospace",
                         bbox=dict(boxstyle="round,pad=0.4",
                                   facecolor=PANEL, edgecolor="#a8ff3e", alpha=0.9))
                cb2 = fig.colorbar(im2, ax=ax2, fraction=0.046, pad=0.02, shrink=1.0)
                cb2.set_label("Δ rad", color="#a8ff3e", fontsize=FSIZE_CB, labelpad=6)
                plt.setp(cb2.ax.yaxis.get_ticklabels(), color=WHITE, fontsize=FSIZE_CB)
                cb2.ax.yaxis.set_tick_params(color=WHITE, length=8, width=2)
                cb2.outline.set_edgecolor(GRID)

            else:
                ax0 = fig.add_subplot(gs[row, 0])
                im0 = _heatmap(ax0, ga, cmap, vmin, vmax,
                               f"L{li+1}  {lbl}  (trained)", color)
                _colorbar(fig, im0, ax0, f"{lbl} (rad)", ticks, tlabels, color)

    _save(fig, out_path)


def plot_phase_distributions(model, out_path="01b_phase_distributions.png",
                             phases_before=None):
    """
    Clean phase distribution plot.

    Layout per PNN layer column:
      - Top subplot    : θ histogram
      - Middle subplot : φ histogram
      - Bottom text box: stats table (μ, σ for init + trained)

    Single shared legend anchored at the very top of the figure.
    No text inside the histogram panels.
    """
    n_pnn   = len(model.all_layers)
    compare = phases_before is not None

    VMIN_T, VMAX_T = 0.0, np.pi
    VMIN_P, VMAX_P = 0.0, 2 * np.pi
    BINS = 28

    FSIZE_TITLE  = 72
    FSIZE_LABEL  = 60
    FSIZE_TICK   = 52
    FSIZE_STATS  = 52
    FSIZE_LEGEND = 52
    FSIZE_SUPER  = 80

    # Figure: tall enough for 2 histo rows + stats row + legend
    col_w  = 26.0         # inches per layer column
    fig_w  = col_w * n_pnn + 3.0
    fig_h  = 60.0         # fixed tall height

    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)

    title = ("Phase Distributions  —  Before vs After Training"
             if compare else "Phase Distributions  —  Trained")
    fig.suptitle(title, color=WHITE, fontsize=FSIZE_SUPER,
                 fontfamily="monospace", fontweight="bold", y=0.975)

    # 3 rows: θ hist | φ hist | stats table
    # height_ratios: histograms tall, stats row shorter
    gs = gridspec.GridSpec(3, n_pnn, figure=fig,
                           height_ratios=[5, 5, 3],
                           hspace=0.45, wspace=0.32,
                           left=0.08, right=0.97,
                           top=0.92, bottom=0.18)

    PARAMS = [
        ("θ", "internal phase", ACCENT,  VMIN_T, VMAX_T,
         [0, np.pi/4, np.pi/2, 3*np.pi/4, np.pi],
         ["0", "π/4", "π/2", "3π/4", "π"]),
        ("φ", "external phase", ACCENT2, VMIN_P, VMAX_P,
         [0, np.pi/2, np.pi, 3*np.pi/2, 2*np.pi],
         ["0", "π/2", "π", "3π/2", "2π"]),
    ]

    # Collect handles for shared legend (only need to do once)
    legend_handles = None

    for li, layer in enumerate(model.all_layers):
        arr_trained = [
            np.clip(layer.mesh.theta.numpy().reshape(-1), VMIN_T, VMAX_T),
            np.clip(layer.mesh.phi.numpy().reshape(-1),   VMIN_P, VMAX_P),
        ]
        arr_init = [
            np.clip(phases_before[li]["theta"], VMIN_T, VMAX_T),
            np.clip(phases_before[li]["phi"],   VMIN_P, VMAX_P),
        ] if compare else [None, None]

        stats_lines = []   # collect stats text for bottom panel

        for pi, (sym, name, color, vmin, vmax, ticks, tlabels) in enumerate(PARAMS):
            ax = fig.add_subplot(gs[pi, li])
            _style_ax(ax)
            ax.tick_params(axis='both', labelsize=FSIZE_TICK, colors=WHITE)

            bins    = np.linspace(vmin, vmax, BINS + 1)
            trained = arr_trained[pi]
            init    = arr_init[pi]

            if compare and init is not None:
                h1, _, p1 = ax.hist(init, bins=bins, density=True,
                        histtype="stepfilled", color="#6a7aaa",
                        alpha=0.50, zorder=2, label="Init")
                ax.hist(init, bins=bins, density=True,
                        histtype="step", color="#aabbee",
                        linewidth=2.5, alpha=1.0, zorder=3)
                h2, _, p2 = ax.hist(trained, bins=bins, density=True,
                        histtype="stepfilled", color=color,
                        alpha=0.55, zorder=4, label="Trained")
                ax.hist(trained, bins=bins, density=True,
                        histtype="step", color=color,
                        linewidth=2.5, alpha=1.0, zorder=5)

                # Mean lines only — no text inside plot
                mu_i = float(np.mean(init))
                mu_t = float(np.mean(trained))
                ax.axvline(mu_i, color="#aabbee", lw=2.0, linestyle="--", zorder=6)
                ax.axvline(mu_t, color=color,     lw=2.5, linestyle="--", zorder=7)

                # Collect stats for bottom table
                si, st = float(np.std(init)), float(np.std(trained))
                stats_lines.append(
                    (sym, color,
                     f"init    μ={mu_i:.3f}  σ={si:.3f}",
                     f"trained μ={mu_t:.3f}  σ={st:.3f}")
                )

                if legend_handles is None and li == 0 and pi == 0:
                    import matplotlib.patches as mpatches
                    legend_handles = [
                        mpatches.Patch(color="#6a7aaa", alpha=0.8, label="Init (random)"),
                        mpatches.Patch(color=ACCENT,   alpha=0.8, label="Trained  θ"),
                        mpatches.Patch(color=ACCENT2,  alpha=0.8, label="Trained  φ"),
                    ]
            else:
                ax.hist(trained, bins=bins, density=True,
                        histtype="stepfilled", color=color, alpha=0.6, zorder=2)
                ax.hist(trained, bins=bins, density=True,
                        histtype="step", color=color, linewidth=2.5, zorder=3)
                mu_t = float(np.mean(trained))
                ax.axvline(mu_t, color=color, lw=2.5, linestyle="--", zorder=4)
                st = float(np.std(trained))
                stats_lines.append(
                    (sym, color, None,
                     f"trained μ={mu_t:.3f}  σ={st:.3f}")
                )

            ax.set_xlim(vmin, vmax)
            ax.set_xticks(ticks)
            ax.set_xticklabels(tlabels, fontsize=FSIZE_TICK, color=WHITE)
            ax.tick_params(axis="y", labelsize=FSIZE_TICK, colors=WHITE)
            ax.set_xlabel(f"{sym}  ({name})", fontsize=FSIZE_LABEL,
                          color=WHITE, labelpad=6)
            ax.set_ylabel("Density" if li == 0 else "",
                          fontsize=FSIZE_LABEL, color=WHITE, labelpad=6)
            ax.set_title(f"Layer {li+1}  —  {sym}",
                         color=color, fontsize=FSIZE_TITLE,
                         fontfamily="monospace", fontweight="bold", pad=10)
            ax.grid(color=GRID, lw=0.8, alpha=0.6, axis="y")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)

        # ── Stats table in row 2 ──────────────────────────────────────────────
        ax_s = fig.add_subplot(gs[2, li])
        ax_s.set_facecolor(PANEL)
        ax_s.axis("off")
        for spine in ax_s.spines.values():
            spine.set_visible(False)

        # Draw a clean table: header + one row per phase param
        row_y  = [0.85, 0.42]
        header = ["", "init  μ  /  σ", "trained  μ  /  σ"]
        col_x  = [0.01, 0.28, 0.65]

        # header row
        for cx, hdr in zip(col_x, header):
            ax_s.text(cx, 1.05, hdr, transform=ax_s.transAxes,
                      fontsize=FSIZE_STATS - 2, color=WHITE,
                      fontfamily="monospace", fontweight="bold", va="top")

        ax_s.plot([0.0, 1.0], [0.98, 0.98], color=GRID, lw=1.5,
                  transform=ax_s.transAxes, clip_on=False)

        for (sym, color, init_str, trained_str), ry in zip(stats_lines, row_y):
            # symbol
            ax_s.text(col_x[0], ry, sym, transform=ax_s.transAxes,
                      fontsize=FSIZE_STATS, color=color,
                      fontfamily="monospace", fontweight="bold", va="top")
            # init stats
            if init_str:
                ax_s.text(col_x[1], ry, init_str.replace("init    ", ""),
                          transform=ax_s.transAxes,
                          fontsize=FSIZE_STATS, color="#aabbee",
                          fontfamily="monospace", va="top")
            # trained stats
            ax_s.text(col_x[2], ry, trained_str.replace("trained ", ""),
                      transform=ax_s.transAxes,
                      fontsize=FSIZE_STATS, color=color,
                      fontfamily="monospace", va="top")

            ax_s.plot([0.0, 1.0], [ry - 0.35, ry - 0.35], color=GRID, lw=0.8,
                      transform=ax_s.transAxes, clip_on=False)

    # ── Shared legend anchored at top-centre of figure ────────────────────────
    if legend_handles is not None:
        leg = fig.legend(handles=legend_handles,
                         loc="lower center", ncol=3,
                         fontsize=FSIZE_LEGEND,
                         facecolor=PANEL, edgecolor=GRID,
                         bbox_to_anchor=(0.5, 0.155),
                         framealpha=0.95,
                         handleheight=2.5, handlelength=3.0,
                         borderpad=1.2, labelspacing=1.0)
        for t in leg.get_texts():
            t.set_color(WHITE)
        leg.get_frame().set_linewidth(2.0)

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 02  Training curves
# ═══════════════════════════════════════════════════════════════════════════════
def plot_training_curves(acc_history, loss_history, out_path="02_training_curves.png"):
    n_layers = len(model.all_layers) if False else None  # unused — derived from histories

    # Normalise both histories to dicts {layer_idx: [values]}
    acc_dict  = _normalise_history(acc_history,  n_layers=1)
    loss_dict = _normalise_history(loss_history, n_layers=1)
    n_curves  = max(len(acc_dict), len(loss_dict))

    fig = _fig(w=12, h=4.5)
    fig.suptitle("Training History", color=WHITE, fontsize=16,
                 fontfamily="monospace", y=1.01)

    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32,
                           left=0.07, right=0.97, top=0.88, bottom=0.12)

    ax_l = fig.add_subplot(gs[0, 0]); _style_ax(ax_l)
    ax_l.set_title("Loss", color=WHITE, fontsize=18, fontfamily="monospace")
    ax_l.set_xlabel("Epoch"); ax_l.set_ylabel("Loss")
    ax_l.yaxis.label.set_color(WHITE); ax_l.xaxis.label.set_color(WHITE)

    ax_a = fig.add_subplot(gs[0, 1]); _style_ax(ax_a)
    ax_a.set_title("Accuracy", color=WHITE, fontsize=18, fontfamily="monospace")
    ax_a.set_xlabel("Epoch"); ax_a.set_ylabel("Accuracy")
    ax_a.yaxis.label.set_color(WHITE); ax_a.xaxis.label.set_color(WHITE)
    ax_a.set_ylim(0, 1)

    layer_colors = ([ACCENT, ACCENT2, "#a8ff3e", "#ff3e8a", "#c97bff",
                      "#ffd93d", "#ff6b6b", "#6bcb77"])

    all_keys = sorted(set(list(loss_dict.keys()) + list(acc_dict.keys())))
    for li in all_keys:
        c    = layer_colors[li % len(layer_colors)]
        lbl  = f"Layer {li+1}"

        if li in loss_dict and len(loss_dict[li]) > 0:
            eps = range(1, len(loss_dict[li]) + 1)
            ax_l.plot(eps, loss_dict[li], color=c, lw=2, label=lbl)
            ax_l.fill_between(eps, loss_dict[li], alpha=0.12, color=c)

        if li in acc_dict and len(acc_dict[li]) > 0:
            eps = range(1, len(acc_dict[li]) + 1)
            ax_a.plot(eps, acc_dict[li], color=c, lw=2, label=lbl)
            ax_a.fill_between(eps, acc_dict[li], alpha=0.12, color=c)
            ax_a.axhline(acc_dict[li][-1], color=c, lw=0.6,
                         linestyle="--", alpha=0.5)

    for ax in (ax_l, ax_a):
        ax.grid(color=GRID, linewidth=0.5, alpha=0.6)
        leg = ax.legend(fontsize=14, facecolor=PANEL, edgecolor=GRID)
        for t in leg.get_texts(): t.set_color(WHITE)

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 03  Confusion matrix
# ═══════════════════════════════════════════════════════════════════════════════
def plot_confusion_matrix(model, n_classes, x_test, y_test, refs, batch_size=256,
                          out_path="03_confusion_matrix.png"):
    if refs is None:
        print(f"  –  skipping {out_path}  (refs=None)")
        return

    ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
          .batch(batch_size).prefetch(tf.data.AUTOTUNE))

    all_true, all_pred = [], []
    for xb, yb in ds:
        true = tf.argmax(yb, axis=1, output_type=tf.int32).numpy()
        _, vp = forward_all_cos_sims(model, xb, refs)
        all_true.extend(true)
        all_pred.extend(vp.numpy())

    C = n_classes# len(CLASS_NAMES[:n_classes])
    class_names = [str(i) for i in range(C)]

    cm = np.zeros((C, C), dtype=int)
    for t, p in zip(all_true, all_pred):
        cm[t, p] += 1

    cm_norm = cm.astype(float) / (cm.sum(axis=1, keepdims=True) + 1e-9)

    # Scale figure and font sizes with number of classes
    cell      = max(1.1, 9.0 / C)        # inches per cell
    fig_size  = C * cell + 2.0
    fsize_ann = max(7,  int(18 - C))     # annotation inside cells
    fsize_lbl = max(8,  int(22 - C))     # axis tick labels
    fsize_ttl = max(10, int(24 - C))     # title

    fig = _fig(w=fig_size, h=fig_size)
    fig.suptitle("Confusion Matrix (Vote, Test Set)",
                 color=WHITE, fontsize=26, fontfamily="monospace", y=0.97)
    ax = fig.add_subplot(111); _style_ax(ax)

    im = ax.imshow(cm_norm, cmap=_PHOTON, vmin=0, vmax=1)

    for i in range(C):
        for j in range(C):
            v    = cm[i, j]
            pct  = cm_norm[i, j]
            col  = "white" if pct < 0.5 else BG
            ax.text(j, i, f"{v}\n{pct:.0%}", ha="center", va="center",
                    fontsize=fsize_ann, color=col, fontfamily="monospace",
                    fontweight="bold" if i == j else "normal")

    ax.set_xticks(range(C)); ax.set_yticks(range(C))
    ax.set_xticklabels([f"Pred {c}" for c in class_names],
                       color=WHITE, fontsize=fsize_lbl)
    ax.set_yticklabels([f"True {c}" for c in class_names],
                       color=WHITE, fontsize=fsize_lbl)
    ax.set_xlabel("Predicted class", fontsize=fsize_lbl, color=WHITE, labelpad=12)
    ax.set_ylabel("True class",      fontsize=fsize_lbl, color=WHITE, labelpad=12)

    cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
    cb.set_label("Recall", color=WHITE, fontsize=fsize_lbl)
    plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE, fontsize=fsize_lbl - 2)

    total_correct = np.trace(cm)
    ax.set_title(f"Overall accuracy: {total_correct / cm.sum():.1%}",
                 color=ACCENT, fontsize=fsize_ttl, fontfamily="monospace")

    _save(fig, out_path)

# ═══════════════════════════════════════════════════════════════════════════════
# 04  Feature space (PCA) — all layers in one figure
# ═══════════════════════════════════════════════════════════════════════════════
def plot_feature_space(model, n_classes, x_test, y_test, refs, layer_idx=-1,
                       out_path="04_feature_space.png"):
    """Show PCA feature space for every layer side-by-side in one figure."""

    n_layers = len(model.all_layers)
    ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
          .batch(512).prefetch(tf.data.AUTOTUNE))

    # Collect labels once
    all_labels = []
    for _, yb in ds:
        all_labels.extend(tf.argmax(yb, axis=1).numpy())
    labels = np.array(all_labels)

    # Collect features for every layer
    all_feats = [[] for _ in range(n_layers)]
    for xb, _ in (tf.data.Dataset.from_tensor_slices((x_test, y_test))
                  .batch(512).prefetch(tf.data.AUTOTUNE)):
        for li in range(n_layers):
            f = model.forward_ff(xb, li, training=False)
            f = tf.math.l2_normalize(f, axis=1)
            all_feats[li].append(f.numpy())

    panel_w = 9
    fig_w   = panel_w * n_layers + 1
    fig_h   = 8
    fig = plt.figure(figsize=(fig_w, fig_h), facecolor=BG)
    fig.suptitle("Feature Space — PCA per Layer",
                 color=WHITE, fontsize=22, fontfamily="monospace",
                 fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(1, n_layers, figure=fig,
                           wspace=0.30,
                           left=0.06, right=0.97,
                           top=0.92, bottom=0.10)

    for li in range(n_layers):
        feats  = np.concatenate(all_feats[li], axis=0)
        pca    = PCA(n_components=2)
        coords = pca.fit_transform(feats)
        

        ax = fig.add_subplot(gs[li]); _style_ax(ax)

        for ci, (name, col) in enumerate(zip(CLASS_NAMES[:n_classes], CLASS_COLORS[:n_classes])):
            mask = labels == ci
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       c=col, s=8, alpha=0.35, linewidths=0,
                       label=f"Digit {name}")
        if refs is not None:
            anc    = pca.transform(refs[li].numpy())
            for ci, (name, col) in enumerate(zip(CLASS_NAMES[:n_classes], CLASS_COLORS[:n_classes])):
                ax.scatter(*anc[ci], marker="*", s=350, color=col,
                        edgecolors="white", linewidths=0.8, zorder=5)
                ax.annotate(f"  [{name}]", anc[ci],
                            color=col, fontsize=16, fontfamily="monospace",
                            fontweight="bold")

        ax.set_xlabel(f"PC1  ({pca.explained_variance_ratio_[0]:.1%} var)",
                      fontsize=15, color=WHITE)
        ax.set_ylabel(f"PC2  ({pca.explained_variance_ratio_[1]:.1%} var)",
                      fontsize=15, color=WHITE)
        ax.set_title(f"Layer {li+1}", color=ACCENT, fontsize=18,
                     fontfamily="monospace", fontweight="bold")
        ax.grid(color=GRID, linewidth=0.4, alpha=0.5)
        ax.tick_params(colors=WHITE, labelsize=13)

        # Legend only on the last panel to save space
        if li == n_layers - 1:
            leg = ax.legend(fontsize=14, facecolor=PANEL, edgecolor=GRID,
                            loc="upper right", markerscale=2)
            for t in leg.get_texts(): t.set_color(WHITE)

        _label(ax, "★ = simplex anchor", size=18, color=WHITE, x=0.02, y=0.04)

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 05  Cosine-similarity heatmap
# ═══════════════════════════════════════════════════════════════════════════════
def plot_cosine_heatmap(model, n_classes, x_test, y_test, refs,
                        out_path="05_cosine_similarity.png"):

    n_layers = len(model.all_layers)
    C        = n_classes# len(CLASS_NAMES)

    ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
          .batch(512).prefetch(tf.data.AUTOTUNE))

    # mean_cos[layer, true_class, anchor_class]
    mean_cos = np.zeros((n_layers, C, C))
    counts   = np.zeros(C, dtype=int)

    for xb, yb in ds:
        true = tf.argmax(yb, axis=1).numpy()
        for li in range(n_layers):
            out = model.forward_ff(xb, li, training=False)
            out = tf.math.l2_normalize(out, axis=1)
            cs  = tf.matmul(out, refs[li], transpose_b=True).numpy()  # (B, C)
            for ci in range(C):
                mask = true == ci
                if mask.any():
                    mean_cos[li, ci] += cs[mask].sum(axis=0)
        for ci in range(C):
            counts[ci] += (true == ci).sum()

    for ci in range(C):
        if counts[ci] > 0:
            mean_cos[:, ci, :] /= counts[ci]

    fig = _fig(w=5 * n_layers + 1, h=5)
    fig.suptitle("Mean Cosine Similarity  (true class × anchor)",
                 color=WHITE, fontsize=20, fontfamily="monospace", y=1.01)

    gs = gridspec.GridSpec(1, n_layers, figure=fig, wspace=0.35)

    for li in range(n_layers):
        ax = fig.add_subplot(gs[li]); _style_ax(ax)
        im = ax.imshow(mean_cos[li], cmap=_PHOTON, vmin=-1, vmax=1,
                       aspect="equal")

        for i in range(C):
            for j in range(C):
                v   = mean_cos[li, i, j]
                col = "white" if v < 0 else BG
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=14, color=col, fontfamily="monospace")

        ax.set_xticks(range(C)); ax.set_xticklabels(CLASS_NAMES[:n_classes], color=WHITE, fontsize=14)
        ax.set_yticks(range(C)); ax.set_yticklabels(CLASS_NAMES[:n_classes], color=WHITE, fontsize=14)
        ax.set_xlabel("Anchor class", fontsize=16)
        ax.set_ylabel("True class" if li == 0 else "", fontsize=16)
        ax.set_title(f"Layer {li+1}", color=ACCENT, fontsize=18, fontfamily="monospace")

        cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cb.set_label("cos sim", color=WHITE, fontsize=16)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE, fontsize=14)

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 06  Energy conservation
# ═══════════════════════════════════════════════════════════════════════════════
def plot_energy_conservation(model, x_test, out_path="06_energy_conservation.png"):

    n_layers = len(model.all_layers)
    N_SAMPLE = 512
    BINS     = 50
    X_LIM    = (0.0, 1.5)

    # Colours matching the reference image
    COLOR_MESH = ACCENT          # cyan  — mesh-only row
    COLOR_FULL = "#ff8c00"       # orange — full-layer row

    # Row labels (rotated, on the left spine)
    ROW_LABELS = [
        "Mesh only  (unitary check)",
        "Mesh + EO nonlinearity  (full layer)",
    ]
    ROW_COLORS = [ACCENT, ACCENT2]

    fig_w = max(10, 5.5 * n_layers + 1.5)
    fig = plt.figure(figsize=(fig_w, 7.5), facecolor=BG)
    fig.suptitle("Energy Conservation per Layer",
                 color=WHITE, fontsize=20, fontfamily="monospace",
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(
        2, n_layers, figure=fig,
        hspace=0.52, wspace=0.32,
        left=0.13, right=0.97,
        top=0.90, bottom=0.10,
    )

    sample = tf.convert_to_tensor(x_test[:N_SAMPLE])
    x_cur  = sample   # propagated state (with NL) for sequential feeding

    for li, layer in enumerate(model.all_layers):
        E_in = tf.reduce_sum(tf.abs(x_cur) ** 2, axis=1).numpy()

        # ── mesh-only ratio ───────────────────────────────────────────────
        E_mesh  = tf.reduce_sum(tf.abs(layer.forward_linear(x_cur)) ** 2,
                                axis=1).numpy()
        r_mesh  = E_mesh / (E_in + 1e-12)

        # ── full-layer ratio (mesh + nonlinearity) ────────────────────────
        E_full  = tf.reduce_sum(tf.abs(layer(x_cur, training=False)) ** 2,
                                axis=1).numpy()
        r_full  = E_full / (E_in + 1e-12)

        def _draw(ax, ratio, color, row_idx):
            _style_ax(ax)

            # histogram
            r_std = float(ratio.std())
            if r_std < 1e-6:
                # degenerate spike — draw a single bar
                ax.bar([float(ratio.mean())], [len(ratio)],
                       width=0.01, color=color, alpha=0.85, edgecolor=PANEL,
                       zorder=3)
            else:
                ax.hist(ratio, bins=BINS, range=X_LIM,
                        color=color, alpha=0.80, edgecolor=PANEL,
                        linewidth=0.3, zorder=3)

            # reference lines
            ax.axvline(1.0, color=WHITE, lw=1.4, linestyle="--",
                       alpha=0.7, label="ideal = 1", zorder=4)
            ax.axvline(float(ratio.mean()), color="#a8ff3e", lw=1.8,
                       linestyle="-", zorder=5,
                       label=f"mean = {ratio.mean():.4f}")

            ax.set_xlim(*X_LIM)
            ax.set_xlabel("|E_out|² / |E_in|²", fontsize=12, color=WHITE)
            ax.set_ylabel("Count" if li == 0 else "", fontsize=12, color=WHITE)
            ax.grid(color=GRID, linewidth=0.4, alpha=0.5, zorder=0)

            # layer title only on top row
            if row_idx == 0:
                ax.set_title(f"Layer {li+1}", color=ACCENT, fontsize=15,
                             fontfamily="monospace", fontweight="bold")

            # legend (ideal + mean)
            leg = ax.legend(fontsize=11, facecolor=PANEL, edgecolor=GRID,
                            loc="upper left")
            for t in leg.get_texts():
                t.set_color(WHITE)

            # ── annotation box: energy loss % + sample count ──────────────
            loss_pct  = max(0.0, 1.0 - float(ratio.mean())) * 100.0
            n_samples = len(ratio)
            ann_color = "#ff4444" if loss_pct > 5.0 else "#a8ff3e"
            ax.text(0.98, 0.97,
                    f"loss = {loss_pct:.1f}%\nn = {n_samples} samples",
                    transform=ax.transAxes,
                    fontsize=11, color=WHITE, fontfamily="monospace",
                    va="top", ha="right",
                    bbox=dict(boxstyle="round,pad=0.4",
                              facecolor=ann_color,
                              edgecolor=ann_color,
                              alpha=0.85))

        ax_top = fig.add_subplot(gs[0, li])
        ax_bot = fig.add_subplot(gs[1, li])
        _draw(ax_top, r_mesh, COLOR_MESH, row_idx=0)
        _draw(ax_bot, r_full, COLOR_FULL, row_idx=1)

        # advance propagated state through the full layer for the next layer
        x_cur = layer(x_cur, training=False)

    # ── rotated row labels on the far left ───────────────────────────────────
    for row_idx, (label, color) in enumerate(zip(ROW_LABELS, ROW_COLORS)):
        # place in axes-fraction coordinates of the leftmost column
        ax_ref = fig.add_subplot(gs[row_idx, 0])
        ax_ref.set_visible(False)   # invisible — just for transform
        fig.text(
            0.005,
            ax_ref.get_position().y0 + ax_ref.get_position().height / 2,
            label,
            color=color, fontsize=10, fontfamily="monospace",
            fontweight="bold", rotation=90,
            va="center", ha="center",
        )

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 07  Phase rose (polar histogram)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_phase_rose(model, out_path="07_phase_rose.png", phases_before=None):

    n_layers = len(model.all_layers)
    compare  = phases_before is not None
    bins     = 36
    cell     = 3

    # Always 2 cols (θ | φ), n_layers rows.
    # When compare=True: init and trained are overlaid on the SAME axes.
    fig = plt.figure(figsize=(2 * cell, n_layers * cell), facecolor=BG)

    title = ("Learned Phase Distribution  —  Before vs After Training  (polar)"
             if compare else "Learned Phase Distribution (polar)")
    fig.suptitle(title, color=WHITE, fontsize=16, fontfamily="monospace", y=1.02)

    def _draw_rose(ax, phases, color, alpha, label):
        counts, bin_edges = np.histogram(phases, bins=bins, range=(0, 2 * np.pi))
        widths  = np.diff(bin_edges)
        centers = bin_edges[:-1] + widths / 2
        ax.bar(centers, counts, width=widths * 0.9,
               color=color, alpha=alpha, edgecolor=PANEL, label=label)

    for li, layer in enumerate(model.all_layers):
        theta_trained = layer.mesh.theta.numpy().reshape(-1) % (2 * np.pi)
        phi_trained   = layer.mesh.phi.numpy().reshape(-1)   % (2 * np.pi)

        for ki, (trained, sym, color) in enumerate([
            (theta_trained, "θ", ACCENT),
            (phi_trained,   "φ", ACCENT2),
        ]):
            ax = fig.add_subplot(n_layers, 2, li * 2 + ki + 1,
                                 projection="polar")
            ax.set_facecolor(PANEL)
            ax.tick_params(colors=WHITE, labelsize=10)
            ax.spines["polar"].set_edgecolor(GRID)
            ax.set_yticks([])
            ax.grid(color=GRID, linewidth=0.4, alpha=0.5)

            if compare:
                init = phases_before[li][sym.replace("θ","theta").replace("φ","phi")] % (2 * np.pi)
                # init: dim, trained: bright — overlaid on same axes
                _draw_rose(ax, init,    color, 0.30, "init")
                _draw_rose(ax, trained, color, 0.85, "trained")
                ax.set_title(f"L{li+1} {sym}", color=color, fontsize=12,
                             fontfamily="monospace", pad=10)
                # small legend inside the polar axes
                leg = ax.legend(fontsize=8, loc="lower left",
                                bbox_to_anchor=(0.0, -0.12),
                                facecolor=PANEL, edgecolor=GRID,
                                framealpha=0.8)
                for t in leg.get_texts():
                    t.set_color(WHITE)
            else:
                _draw_rose(ax, trained, color, 0.75, "trained")
                name = "theta" if sym == "θ" else "phi"
                ax.set_title(f"L{li+1} {sym} ({name})",
                             color=color, fontsize=12,
                             fontfamily="monospace", pad=10)

    fig.tight_layout()
    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 06b  Per-waveguide output power distribution
# ═══════════════════════════════════════════════════════════════════════════════
def plot_waveguide_power(model,n_classes, x_test, y_test,
                         out_path="06b_waveguide_power.png"):
    """
    For each layer, shows the mean output power |E_out[wg]|² per waveguide,
    broken down by true class.  Reveals which waveguides carry signal for
    each class after the mesh transformation.

    Layout: n_layers rows × 1 column, each panel is a grouped bar chart
    (waveguide index on x, mean power on y, one bar per class).
    """
    n_layers = len(model.all_layers)
    C        = n_classes# len(CLASS_NAMES)
    N_WG      = int(model.all_layers[0].n_ports)
    ds = (tf.data.Dataset.from_tensor_slices((x_test, y_test))
          .batch(512).prefetch(tf.data.AUTOTUNE))

    # power_sum[layer, class, wg], count[class]
    power_sum = np.zeros((n_layers, C, N_WG))
    counts    = np.zeros(C, dtype=int)

    for xb, yb in ds:
        true = tf.argmax(yb, axis=1).numpy()
        x = xb
        for li, layer in enumerate(model.all_layers):
            out   = layer.forward_linear(x)              # (B, N_WG) complex
            power = tf.abs(out) ** 2                     # (B, N_WG) real
            for ci in range(C):
                mask = true == ci
                if mask.any():
                    power_sum[li, ci] += power.numpy()[mask].sum(axis=0)
            x = layer(x, training=False)
        for ci in range(C):
            counts[ci] += (true == ci).sum()

    # normalise
    mean_power = np.zeros_like(power_sum)
    for ci in range(C):
        if counts[ci] > 0:
            mean_power[:, ci, :] = power_sum[:, ci, :] / counts[ci]

    # ── figure ────────────────────────────────────────────────────────────────
    panel_h = 4.5
    fig = plt.figure(figsize=(14, panel_h * n_layers + 1.5), facecolor=BG)
    fig.suptitle("Per-Waveguide Output Power  (mean |E|² by class)",
                 color=WHITE, fontsize=18, fontfamily="monospace",
                 fontweight="bold", y=1.01)

    gs = gridspec.GridSpec(n_layers, 1, figure=fig,
                           hspace=0.55,
                           left=0.07, right=0.97,
                           top=0.94, bottom=0.07)

    bar_w   = 0.8 / C          # width of each class bar within a waveguide slot
    wg_idx  = np.arange(N_WG)

    for li in range(n_layers):
        ax = fig.add_subplot(gs[li]); _style_ax(ax)


        for ci, (name, col) in enumerate(zip(CLASS_NAMES[:n_classes], CLASS_COLORS[:n_classes])):
            offset = (ci - C / 2 + 0.5) * bar_w
            ax.bar(wg_idx + offset, mean_power[li, ci],
                   width=bar_w * 0.92, color=col, alpha=0.85,
                   label=f"Digit {name}", zorder=3)

        # mean across all classes (grey line)
        overall = mean_power[li].mean(axis=0)
        ax.plot(wg_idx, overall, color=WHITE, lw=1.5,
                linestyle="--", alpha=0.6, label="mean (all)", zorder=4)

        ax.set_xlim(-0.6, N_WG - 0.4)
        ax.set_xticks(wg_idx[::2])
        ax.set_xticklabels(wg_idx[::2], fontsize=12, color=WHITE)
        ax.tick_params(axis="y", labelsize=12, colors=WHITE)
        ax.set_xlabel("Waveguide index", fontsize=13, color=WHITE)
        ax.set_ylabel("Mean |E|²",       fontsize=13, color=WHITE)
        ax.set_title(f"Layer {li+1}", color=ACCENT, fontsize=16,
                     fontfamily="monospace", fontweight="bold")
        ax.grid(color=GRID, lw=0.5, alpha=0.5, axis="y", zorder=0)

        if li == 0:
            leg = ax.legend(fontsize=12, facecolor=PANEL, edgecolor=GRID,
                            loc="upper right", ncol=C)
            for t in leg.get_texts(): t.set_color(WHITE)

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# 06c  Light propagation through the mesh (transfer matrix visualisation)
# ═══════════════════════════════════════════════════════════════════════════════
def plot_light_propagation(model, out_path="06c_light_propagation.png",
                           matrices_before=None):
    """
    Layout: n_layers rows × 2 cols (Untrained | Trained) per layer.
    Always shows both panels — untrained uses matrices_before if provided,
    otherwise re-uses a fresh random identity as a reference.
    """
    n_layers = len(model.all_layers)
    N        = model.all_layers[0].n_ports
    N_COLS   = model.all_layers[0].n_ports
    src_wg   = N // 2

    fire_cmap = LinearSegmentedColormap.from_list(
        "fire", ["#000000", "#8b0000", "#ff0000", "#ff8c00", "#ffff00"], N=256
    )

    def _mzi_matrix(theta, phi):
        """2×2 MZI transfer matrix (beamsplitter + phase)."""
        c, s = np.cos(theta / 2), np.sin(theta / 2)
        return np.array([[c, 1j*s], [1j*s, c]], dtype=np.complex128) * np.exp(1j * phi)

    def _col_unitary(thetas, phis, col):
        """Assemble N×N unitary for one mesh column from per-MZI (θ,φ)."""
        U     = np.eye(N, dtype=np.complex128)
        pairs = list(range(0, N-1, 2)) if col % 2 == 0 else list(range(1, N-1, 2))
        for k, wg in enumerate(pairs):
            m = _mzi_matrix(thetas[k], phis[k])
            U[wg,   wg]   = m[0, 0];  U[wg,   wg+1] = m[0, 1]
            U[wg+1, wg]   = m[1, 0];  U[wg+1, wg+1] = m[1, 1]
        return U

    def _propagation(theta_flat, phi_flat):
        """Step through each MZI column, recording |amplitude|² per waveguide."""
        e = np.zeros(N, dtype=np.complex128)
        e[src_wg] = 1.0
        prop = np.zeros((N, N_COLS + 1))
        prop[:, 0] = np.abs(e) ** 2
        idx = 0
        for col in range(N_COLS):
            pairs = list(range(0, N-1, 2)) if col%2==0 else list(range(1, N-1, 2))
            n_mzi = len(pairs)
            U = _col_unitary(theta_flat[idx:idx+n_mzi],
                             phi_flat[idx:idx+n_mzi], col)
            e = U @ e
            prop[:, col+1] = np.abs(e) ** 2
            idx += n_mzi
        return prop

    # 2 rows × 1 col: top = untrained, bottom = trained (all layers concatenated)
    cell_h = 6
    fig = plt.figure(figsize=(max(14, 6 * n_layers), cell_h * 2 + 1.5),
                     facecolor=BG)
    fig.suptitle("Light Propagation — All Layers  (Untrained vs Trained)",
                 color=WHITE, fontsize=16, fontfamily="monospace",
                 fontweight="bold", y=0.98)

    gs = gridspec.GridSpec(2, 1, figure=fig,
                           hspace=0.42,
                           left=0.06, right=0.96,
                           top=0.90, bottom=0.07)

    def _propagation_all_layers(phases_list):
        """
        Propagate through ALL layers sequentially.
        phases_list: list of {"theta": array, "phi": array} per layer.
        Returns prop of shape (N, n_layers * (N_COLS+1)) — one block per layer.
        Each column of prop = waveguide powers at that mesh depth.
        """
        e    = np.zeros(N, dtype=np.complex128)
        e[src_wg] = 1.0
        cols_all = []
        for phases in phases_list:
            theta_flat = phases["theta"]
            phi_flat   = phases["phi"]
            idx = 0
            # record state entering this layer
            cols_all.append(np.abs(e) ** 2)
            for col in range(N_COLS):
                pairs = list(range(0, N-1, 2)) if col%2==0 else list(range(1, N-1, 2))
                n_mzi = len(pairs)
                U = _col_unitary(theta_flat[idx:idx+n_mzi],
                                 phi_flat[idx:idx+n_mzi], col)
                e = U @ e
                cols_all.append(np.abs(e) ** 2)
                idx += n_mzi
        return np.stack(cols_all, axis=1)  # (N, n_layers*(N_COLS+1))

    # Build phase lists for trained and untrained
    trained_phases = [{"theta": layer.mesh.theta.numpy().reshape(-1).astype(np.float64),
                        "phi":  layer.mesh.phi.numpy().reshape(-1).astype(np.float64)}
                      for layer in model.all_layers]

    if matrices_before is not None:
        before_phases = [{"theta": matrices_before[li]["theta"].astype(np.float64),
                          "phi":   matrices_before[li]["phi"].astype(np.float64)}
                         for li in range(n_layers)]
    else:
        rng_ = np.random.RandomState(0)
        before_phases = [{"theta": rng_.uniform(0, np.pi,    trained_phases[li]["theta"].shape),
                          "phi":   rng_.uniform(0, 2*np.pi,  trained_phases[li]["phi"].shape)}
                         for li in range(n_layers)]

    prop_before  = _propagation_all_layers(before_phases)
    prop_trained = _propagation_all_layers(trained_phases)

    total_cols = n_layers * (N_COLS + 1)

    # Layer boundary x positions for divider lines
    boundaries = [(li+1) * (N_COLS+1) for li in range(n_layers - 1)]
    # Label centres per layer
    centres = [(li * (N_COLS+1) + (N_COLS+1)//2) for li in range(n_layers)]

    panels = [
        (prop_before,  "Untrained", "#6a7aaa"),
        (prop_trained, "Trained",   ACCENT),
    ]

    for ci, (prop, lbl, title_col) in enumerate(panels):
        vmax = max(np.percentile(prop, 99.5), 1e-6)

        ax = fig.add_subplot(gs[ci])
        ax.set_facecolor("black")
        for spine in ax.spines.values():
            spine.set_edgecolor(GRID)

        im = ax.imshow(prop, aspect="auto", origin="upper",
                       cmap=fire_cmap, vmin=0, vmax=vmax,
                       interpolation="nearest",
                       extent=[0, total_cols, N - 0.5, -0.5])

        # Layer dividers and labels
        for bx in boundaries:
            ax.axvline(bx, color=WHITE, lw=1.5, linestyle="--", alpha=0.5)
        for li, cx in enumerate(centres):
            ax.text(cx, -0.8, f"Layer {li+1}",
                    color=WHITE, fontsize=12, fontfamily="monospace",
                    ha="center", va="bottom", transform=ax.get_xaxis_transform())

        ax.set_title(lbl, color=title_col, fontsize=16,
                     fontfamily="monospace", fontweight="bold", pad=10)
        ax.set_xlabel("Mesh column  (ℓ)  —  all layers", fontsize=13,
                      color=WHITE, labelpad=5)
        ax.set_ylabel("Waveguide  (n)" if ci == 0 else "",
                      fontsize=13, color=WHITE, labelpad=5)
        ax.tick_params(colors=WHITE, labelsize=11)

        # x ticks: reset per layer
        tick_pos = [li*(N_COLS+1) + c for li in range(n_layers)
                    for c in range(0, N_COLS+1, 5)]
        tick_lbl = [str(c) for li in range(n_layers)
                    for c in range(0, N_COLS+1, 5)]
        ax.set_xticks(tick_pos)
        ax.set_xticklabels(tick_lbl, fontsize=9, color=WHITE)

        cb = fig.colorbar(im, ax=ax, fraction=0.025, pad=0.02)
        cb.set_label("|amp|²", color=WHITE, fontsize=12)
        plt.setp(cb.ax.yaxis.get_ticklabels(), color=WHITE, fontsize=10)

        ax.axhline(src_wg, color="#00ff88", lw=1.2, linestyle=":",
                   alpha=0.8, label=f"Input wg {src_wg}")
        ax.legend(fontsize=11, facecolor=PANEL, edgecolor=GRID, loc="upper right")

    _save(fig, out_path)


# ═══════════════════════════════════════════════════════════════════════════════
# Public entry point
# ═══════════════════════════════════════════════════════════════════════════════
def _save_theme(model, 
                x_test, 
                y_test, 
                refs,
                n_classes,
                acc_history, 
                loss_history,
                train_acc, 
                test_acc,
                outdir, 
                phases_before, 
                suffix,
                matrices_before=None
                ):
    """Save all figures for the currently-active theme."""
    os.makedirs(outdir, exist_ok=True)
    p = lambda name: os.path.join(outdir, name.replace(".png", f"{suffix}.png"))

    plot_architecture(model,
                      out_path=p("01_architecture.png"),
                      phases_before=phases_before)

    plot_phase_distributions(model,
                             out_path=p("01b_phase_distributions.png"),
                             phases_before=phases_before)

    plot_training_curves(acc_history, loss_history,
                         out_path=p("02_training_curves.png"))

    plot_confusion_matrix(model, n_classes, x_test, y_test, refs,
                          out_path=p("03_confusion_matrix.png"))

    plot_feature_space(model, n_classes, x_test, y_test, refs,
                       out_path=p("04_feature_space.png"))

    if refs is not None:
        plot_cosine_heatmap(model,n_classes, x_test, y_test, refs,
                            out_path=p("05_cosine_similarity.png"))

    plot_energy_conservation(model, x_test,
                             out_path=p("06_energy_conservation.png"))
    
    plot_phase_rose(model,
                    out_path=p("07_phase_rose.png"),
                    phases_before=phases_before)

    plot_waveguide_power(model,n_classes, x_test, y_test,
                         out_path=p("06b_waveguide_power.png"))
    
    plot_light_propagation(model,
                           out_path=p("06c_light_propagation.png"),
                           matrices_before=matrices_before)



def save_all_figures(model, x_test, y_test, refs,n_classes,
                     acc_history, loss_history,
                     train_acc, test_acc,
                     outdir="pnn_outputs",
                     phases_before=None,
                     matrices_before = None
                     ):
    """
    Generate and save all figures in BOTH dark and light themes.

    Dark  → pnn_outputs/*_dark.png
    Light → pnn_outputs/*_light.png

    Parameters
    ----------
    model          : trained Photonicmodel
    x_test         : complex64 test features  (N, 25)
    y_test         : float32 one-hot labels   (N, 5)
    refs           : tf.Tensor simplex anchors (L, C, D)
    acc_history    : dict {layer_idx: [acc per epoch]}
    loss_history   : dict {layer_idx: [loss per epoch]}
    train_acc      : float — final vote train accuracy
    test_acc       : float — final vote test accuracy
    outdir         : directory to write PNGs into
    phases_before  : output of capture_phases() called before training
    """
    for theme in ("dark", "light"):
        print(f"\n{'='*55}")
        print(f"  Saving figures [{theme} theme] → {outdir}/")
        print(f"{'='*55}")
        set_theme(theme)
        _save_theme(model, x_test, y_test, refs, n_classes,
                    acc_history, loss_history,
                    train_acc, test_acc,
                    outdir, 
                    phases_before, 
                    suffix=f"_{theme}",
                    matrices_before=None
                    )

    # Restore dark theme as default
    set_theme("dark")
    print(f"\n  All figures saved.\n")
