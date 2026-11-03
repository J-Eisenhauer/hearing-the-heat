"""Figures of the manuscript (Figs. 1-6) and the Supplementary Material (Figs. S1-S9) from the result tables."""
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import colors
from matplotlib.colors import LinearSegmentedColormap, ListedColormap, BoundaryNorm, to_rgba
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.patches import ConnectionPatch, Patch, Rectangle
from matplotlib.path import Path as MplPath
from matplotlib.ticker import MaxNLocator, MultipleLocator

import models as M
import settings as S
from bootstrap import FIT_EXAMPLES
from loudness_control import FIG_EXAMPLES as LOUDNESS_EXAMPLES

for _font in sorted((S.ROOT / "fonts").glob("*.ttf")):          # Inter (SIL OFL, see fonts/OFL.txt)
    matplotlib.font_manager.fontManager.addfont(str(_font))

DEG = "\N{DEGREE SIGN}"
DASH = "\N{EN DASH}"
MINUS = "\N{MINUS SIGN}"
SQ = "\N{SUPERSCRIPT TWO}"

# one colour per model and per measure group; levels are shown by opacity
MODEL_COL = {"time-only": "#4D4D4D", "T-only": "#D6609E", "L_time": "#56B4E9", "L": "#0072B2", "NP": "#D55E00"}
MODEL_LABEL = {"time-only": "time only", "T-only": "temperature only", "L_time": "L$_{\\mathrm{time}}$", "L": "L",
               "NP": "NP"}
GROUP_COL = {"nonlinear": "#18A558", "conventional": "#7B3294"}
DATA_BAR = "0.35"
GREY = "#8c8c8c"
NAME = {"PE": "PE", "LZC78": "LZC78", "RSE4": "RSE$_4$", "RSR": "RSR", "ACI": "ACI", "AEI": "AEI",
        "LogBandEnergy": "Log band energy", "SpectralEntropy": "Spectral entropy",
        "SpectralFlatness": "Spectral flatness", "TemporalEntropy": "Temporal entropy"}
BANDS = [int(f) for f, _ in S.BANDS]
BAND_LABEL = [f"{f}-{2 * f} Hz" for f in BANDS]
FS_LABEL, FS_TICK, FS_LEGEND, FS_TITLE = 18, 16, 16, 20       # figures drawn at about 13 in, printed at ~0.5x
BASE = {"font.family": ["Inter", "DejaVu Sans"]}
LARGE = {**BASE, "font.size": FS_TICK, "axes.labelsize": FS_LABEL, "xtick.labelsize": FS_TICK,
         "ytick.labelsize": FS_TICK, "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "0.35",
         "text.color": "black"}
SMALL = {**BASE, "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False, "axes.linewidth": 0.6,
         "xtick.major.width": 0.6, "ytick.major.width": 0.6}
BAND_ALPHA = {50: 0.40, 80: 0.25, 95: 0.14}
BAR_WIDTH = {50: 5.0, 80: 3.0, 95: 1.3}
LEVELS = (50, 80, 95)
CORR_COLOURS = ["#08306b", "#2171b5", "#9ecae1", "#ffffff", "#a1d99b", "#238b45", "#00441b"]
CORR_LIMITS = [-1.0, -0.95, -0.75, -0.50, 0.50, 0.75, 0.95, 1.0]


def read(folder, name):
    return pd.read_csv(folder / name)


def short_band(f):
    return f"{f // 1024}k" if f >= 1024 else str(f)


def save(fig, name, **kw):
    """PNG for viewing and a vector PDF (TrueType fonts) for the article."""
    fig.savefig(S.FIGURES_DIR / name, **kw)
    with matplotlib.rc_context({"pdf.fonttype": 42}):
        fig.savefig((S.FIGURES_DIR / name).with_suffix(".pdf"), **{**kw, "dpi": 600})
    plt.close(fig)


# =============================================================================
# Bands of the Bayesian bootstrap (Figs. 1, 6 and S9)
# =============================================================================
def union_fill(ax, polygons, color, alpha, zorder=1, n=800):
    """Fill the union of polygons once, so that overlapping bands of one colour are not darker."""
    clean = [np.asarray(p, float) for p in polygons if len(p) >= 3 and np.all(np.isfinite(p))]
    pts = np.vstack(clean)
    (x0, y0), (x1, y1) = pts.min(0), pts.max(0)
    px, py = (x1 - x0) * 0.01 + 1e-9, (y1 - y0) * 0.01 + 1e-9
    xg, yg = np.linspace(x0 - px, x1 + px, n), np.linspace(y0 - py, y1 + py, n)
    inside = np.zeros((n, n), bool)
    for poly in clean:
        ix = np.where((xg >= poly[:, 0].min()) & (xg <= poly[:, 0].max()))[0]
        iy = np.where((yg >= poly[:, 1].min()) & (yg <= poly[:, 1].max()))[0]
        if len(ix) and len(iy):
            xx, yy = np.meshgrid(xg[ix], yg[iy])
            hit = MplPath(poly, closed=True).contains_points(np.column_stack([xx.ravel(), yy.ravel()]))
            inside[np.ix_(iy, ix)] |= hit.reshape(xx.shape)
    ax.contourf(xg, yg, inside.astype(float), levels=[0.5, 1.5], colors=[color], alpha=alpha, antialiased=True,
                zorder=zorder)


def ribbon(x, lo, hi):
    return np.vstack([np.column_stack([x, hi]), np.column_stack([x[::-1], lo[::-1]])])


def convex_hull(points):
    pts = sorted(map(tuple, points))
    cross = lambda o, a, b: (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])
    lower, upper = [], []
    for p in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    for p in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return np.array(lower[:-1] + upper[:-1])


def path_band(x, lo, hi, radius, n=48):
    """Band along a path in the given order: elliptic caps at the points joined by their convex hulls."""
    ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
    caps = [np.column_stack([xi + radius * np.cos(ang), (l + h) / 2 + max((h - l) / 2, 1e-6) * np.sin(ang)])
            for xi, l, h in zip(x, lo, hi)]
    return caps + [convex_hull(np.vstack([a, b])) for a, b in zip(caps[:-1], caps[1:])]


def bootstrap_row(axs, measure, fmin, with_np=False, phases=False, folder=None):
    """One row of Figs. 1 and 6: fit over time (left) and temperature (right) with bootstrap bands.
    phases=True (Figs. 1 and 6): heating occasions as filled and cooling occasions as open points, dotted line at the
    temperature peak in the time panel. axs[1] = None draws the time panel only (Fig. S9); folder: bootstrap results
    (default results/bootstrap)."""
    folder = folder or S.BOOTSTRAP_DIR
    t = read(folder, f"fit_bands_{measure}_{fmin}.csv")
    br = read(folder, f"branch_bands_{measure}_{fmin}.csv")
    col = MODEL_COL["L"]
    radius = 0.012 * (t.T_mean.max() - t.T_mean.min())
    for ax, x, time_panel in ((axs[0], t.time_days, True), (axs[1], t.T_mean, False)):
        if ax is None:
            continue
        for lv in LEVELS[::-1]:
            if time_panel:
                ax.fill_between(x, t[f"L_lo{lv}"], t[f"L_hi{lv}"], color=col, alpha=BAND_ALPHA[lv], lw=0, zorder=1)
                if with_np:
                    ax.fill_between(x, t[f"NP_lo{lv}"], t[f"NP_hi{lv}"], color=MODEL_COL["NP"], alpha=BAND_ALPHA[lv],
                                    lw=0, zorder=1)
            else:
                union_fill(ax, [ribbon(g["T"].to_numpy(), g[f"L_lo{lv}"].to_numpy(), g[f"L_hi{lv}"].to_numpy())
                                for _, g in br.groupby("phase", sort=False)], col, BAND_ALPHA[lv])
                if with_np:
                    union_fill(ax, path_band(x.to_numpy(), t[f"NP_lo{lv}"].to_numpy(), t[f"NP_hi{lv}"].to_numpy(),
                                             radius), MODEL_COL["NP"], BAND_ALPHA[lv])
        if time_panel:
            ax.plot(x, t.L, "-", color=col, lw=2.6, zorder=2)
            if with_np:
                ax.plot(x, t.NP, "-", color=MODEL_COL["NP"], lw=2.6, zorder=2)
        else:
            for _, g in br.groupby("phase", sort=False):
                ax.plot(g["T"], g.L, "-", color=col, lw=2.6, zorder=2)
            if with_np:
                ax.plot(x, t.NP, "-", color=MODEL_COL["NP"], lw=1.0, alpha=0.9, zorder=2)
                ax.plot(x, t.NP, "D", color=MODEL_COL["NP"], ms=3.2, mec="none", zorder=2)
        for lv in LEVELS[::-1]:
            ax.vlines(x, t[f"data_lo{lv}"], t[f"data_hi{lv}"], color=DATA_BAR, lw=BAR_WIDTH[lv], alpha=0.55, zorder=4,
                      capstyle="butt")
        if phases:
            cool = M.cooling(M.T_MEAN)[t.occasion.to_numpy() - 1] == 1
            for sel, face in ((~cool, "black"), (cool, "white")):
                ax.plot(x[sel], t.observed_mean[sel], "o", color="black", mfc=face, mec="black", mew=1.3, ms=6.5,
                        zorder=5)
            if time_panel:
                ax.axvline(t.time_days[t.T_mean.idxmax()], color="0.5", lw=1.0, ls=":", zorder=0)
        else:
            ax.plot(x, t.observed_mean, "o", color="black", ms=6, mec="none", zorder=5)
        ax.set_xlabel("time (days)" if time_panel else f"internal temperature ({DEG}C)")
        ax.set_ylabel("z-score" if time_panel else "")
        if not with_np:
            ax.set_ylim(top=min(ax.get_ylim()[1], 3.0))
    axs[0].xaxis.set_major_locator(MultipleLocator(2))
    axs[0].set_title(f"{NAME[measure]}, {fmin}{DASH}{2 * fmin} Hz", loc="left", fontsize=FS_TITLE, fontweight="bold")


def fig1_L_fit_examples():
    """RSR in three octave bands (FIT_EXAMPLES of bootstrap.py): fit of L over time and temperature."""
    fit = read(S.MODELS_DIR, "03_meanfit_metrics.csv")
    with plt.rc_context(LARGE):
        fig, axs = plt.subplots(len(FIT_EXAMPLES), 2, figsize=(13, 4.8 * len(FIT_EXAMPLES) + 1.6), squeeze=False)
        for r, (m, f) in enumerate(FIT_EXAMPLES):
            bootstrap_row(axs[r], m, f, phases=True)
            x = fit[(fit.measure == m) & (fit.fmin_hz == f) & (fit.model == "L")].iloc[0]
            axs[r, 0].legend([Line2D([], [], ls="")] * 2, [f"R{SQ} = {x.R2:.2f}", f"RMSE = {x.RMSE:.2f}"], loc="best",
                             handlelength=0, handletextpad=0, frameon=True, framealpha=0.85, edgecolor="none",
                             fontsize=FS_LEGEND - 1)
        col = MODEL_COL["L"]
        handles = [Line2D([], [], marker="o", ls="", color="black", ms=7, label="mean of 5 buckets, heating"),
                   Line2D([], [], marker="o", ls="", color="black", mfc="white", mew=1.3, ms=7,
                          label="mean of 5 buckets, cooling"),
                   Line2D([], [], color=col, lw=2.6, label="L fit")]
        handles += [Line2D([], [], color=DATA_BAR, lw=BAR_WIDTH[lv], alpha=0.55, label=f"data {lv} %") for lv in LEVELS]
        handles += [Patch(color=col, alpha=BAND_ALPHA[lv] + 0.1, label=f"fit {lv} %") for lv in LEVELS]
        fig.legend(handles=handles, loc="lower center", ncol=4, frameon=False, fontsize=FS_LEGEND - 1)
        fig.tight_layout(rect=(0, 0.10, 1, 1.0))
        save(fig, "Figure_1.png", dpi=200)


def fig6_np_vs_L():
    fit = read(S.MODELS_DIR, "03_meanfit_metrics.csv")
    per = read(S.MODELS_DIR, "02_cv_per_config.csv")
    m, f = "PE", 4096
    r2 = fit[(fit.measure == m) & (fit.fmin_hz == f)].set_index("model")
    ho = per[(per.measure == m) & (per.fmin_hz == f)].iloc[0]
    with plt.rc_context(LARGE):
        fig, axs = plt.subplots(1, 2, figsize=(13, 6.6), squeeze=False)
        bootstrap_row(axs[0], m, f, with_np=True, phases=True)
        axs[0, 0].legend([Line2D([], [], color=MODEL_COL["L"], lw=2.6), Line2D([], [], color=MODEL_COL["NP"], lw=2.6)],
                         [f"L: R{SQ} = {r2.R2['L']:.2f}, held-out = {ho.HO_L:.2f}",
                          f"NP: R{SQ} = {r2.R2['NP']:.2f} (edf {r2.parameters['NP']:.1f}), held-out = {ho.HO_NP:.2f}"],
                         loc="best", frameon=True, framealpha=0.85, edgecolor="none", fontsize=FS_LEGEND - 2)
        handles = [Line2D([], [], marker="o", ls="", color="black", ms=7, label="mean of 5 buckets, heating"),
                   Line2D([], [], marker="o", ls="", color="black", mfc="white", mew=1.3, ms=7,
                          label="mean of 5 buckets, cooling")]
        handles += [Line2D([], [], color=DATA_BAR, lw=BAR_WIDTH[lv], alpha=0.55, label=f"data {lv} %") for lv in LEVELS]
        handles += [Patch(color="0.5", alpha=0.35, label="fit 50/80/95 %")]
        fig.legend(handles=handles, loc="lower center", ncol=3, frameon=False, fontsize=FS_LEGEND)
        fig.tight_layout(rect=(0, 0.13, 1, 1.0))
        save(fig, "Figure_6.png", dpi=200)


# =============================================================================
# Fig. 2: temperature surrogates
# =============================================================================
def fig2_temperature_surrogates():
    occ = read(S.MODELS_DIR, "00_occasions.csv")
    surr = pd.read_csv(S.MODELS_DIR / "07_temperature_surrogates.csv.gz", usecols=["median_skill_L_nonlinear"])
    summary = read(S.MODELS_DIR, "01_group_summary.csv")
    observed = summary[(summary.group == "nonlinear") & (summary.model == "L")].HO_median.iloc[0]
    rng = np.random.default_rng(3)
    with plt.rc_context({**SMALL, "font.size": 13, "axes.labelsize": 14, "xtick.labelsize": 12.5,
                         "ytick.labelsize": 12.5, "legend.fontsize": 13, "text.color": "black"}):
        fig, axs = plt.subplots(1, 2, figsize=(9.0, 3.7), gridspec_kw={"width_ratios": [1.1, 1]})
        ax = axs[0]
        for _ in range(12):
            ax.plot(occ.time_days, M.iaaft(occ.T_mean.to_numpy(), rng), color=GREY, lw=0.7, alpha=0.7, zorder=1)
        for b in S.BUCKETS:
            ax.plot(occ.time_days, occ[f"T_{b}"], color=MODEL_COL["L"], lw=0.9, alpha=0.85, zorder=2)
        ax.plot(occ.time_days, occ.T_mean, color="black", lw=2.2, zorder=3)
        ax.set_xlabel("time (days)")
        ax.xaxis.set_major_locator(MultipleLocator(2))
        ax.set_ylabel(f"internal temperature ({DEG}C)")
        ax = axs[1]
        ax.hist(surr.median_skill_L_nonlinear, bins=60, color=GREY, alpha=0.8)
        ax.axvline(observed, color=MODEL_COL["L"], lw=2.5)
        ax.legend(handles=[Patch(fc=MODEL_COL["L"], label="observed")], loc="upper left", frameon=False,
                  handlelength=1.8)
        ax.set_xlabel("median held-out skill of L")
        ax.set_ylabel("surrogates")
        fig.tight_layout()
        save(fig, "Figure_2.png", dpi=300)


# =============================================================================
# Fig. 3 and Fig. S4: skill and R2 by band as colour tables
# =============================================================================
def model_long_table():
    per = read(S.MODELS_DIR, "02_cv_per_config.csv")
    fit = read(S.MODELS_DIR, "03_meanfit_metrics.csv")
    rows = []
    for m in S.MODELS + ["NP"]:
        x = per[["family", "measure", "fmin_hz"]].assign(model=m, HO=per[f"HO_{m}"])
        rows.append(x.merge(fit[fit.model == m][["measure", "fmin_hz", "R2"]], on=["measure", "fmin_hz"]))
    return pd.concat(rows)


def colour_table(ax, d, metric, how, models, f_cell, f_head, f_band, f_title, band_labels=True):
    """Best or median configuration of each group per band; shading ranks the cells within the table."""
    agg = (lambda s: s.max()) if how == "Best" else (lambda s: s.median())
    tab = pd.DataFrame({(g, m): [agg(d[(d.family == g) & (d.model == m) & (d.fmin_hz == b)][metric]) for b in BANDS]
                        for g in ("nonlinear", "conventional") for m in models}, index=BANDS)
    v = tab.to_numpy().ravel()
    rank = (-v).argsort().argsort() / (v.size - 1)
    cls = np.where(rank <= 0.10, 0, np.where(rank <= 0.30, 1, np.where(rank <= 0.50, 2, 3))).reshape(tab.shape)
    alpha = [1.0, 0.50, 0.20, 0.0]
    label = {"time-only": "time", "T-only": "T", "L_time": "L$_{\\mathrm{time}}$", "L": "L", "NP": "NP"}
    nrow, ncol = tab.shape
    for i in range(nrow):
        for j in range(ncol):
            ax.add_patch(Rectangle((j, nrow - 1 - i), 1, 1, fc=to_rgba(GROUP_COL[tab.columns[j][0]], alpha[cls[i, j]]),
                                   ec="0.75", lw=0.6))
            ax.text(j + 0.5, nrow - 1 - i + 0.5, f"{round(tab.iat[i, j], 2) + 0.0:.2f}", ha="center", va="center",
                    fontsize=f_cell, fontweight="bold", color="white" if cls[i, j] == 0 else "black")
    for j in range(ncol):
        ax.text(j + 0.5, nrow + 0.25, label[tab.columns[j][1]], ha="center", va="bottom", fontsize=f_head)
    for gi, g in enumerate(("nonlinear", "conventional")):
        ax.text(gi * len(models) + len(models) / 2, nrow + 1.05, g, ha="center", va="bottom", fontsize=f_head,
                color=GROUP_COL[g], fontweight="bold")
    ax.plot([len(models)] * 2, [0, nrow], color="black", lw=1.6)
    ax.add_patch(Rectangle((0, 0), ncol, nrow, fc="none", ec="black", lw=1.2))
    if band_labels:
        for i, b in enumerate(BANDS):
            ax.text(-0.12, nrow - 1 - i + 0.5, f"{b}-{2 * b} Hz", ha="right", va="center", fontsize=f_band)
    ax.set_xlim(-2.6 if band_labels else -0.1, ncol + 0.05)
    ax.set_ylim(-0.1, nrow + 2.3)
    ax.axis("off")
    ax.text(0, nrow + 2.2, f"{how} across measures", ha="left", va="top", fontsize=f_title, fontweight="bold")


def rank_legend(fontsize):
    alpha = [1.0, 0.50, 0.20, 0.0]
    handles = [tuple(Patch(fc=to_rgba(GROUP_COL[g], a), ec="0.6") for g in GROUP_COL) for a in alpha]
    return dict(handles=handles, labels=["best 10 %", "next 20 %", "next 20 %", "remaining 50 %"],
                handler_map={tuple: HandlerTuple(ndivide=None, pad=0)}, handlelength=2.6, fontsize=fontsize)


def fig3_heldout_by_band():
    d = model_long_table()
    with plt.rc_context(BASE):
        fig, axs = plt.subplots(1, 2, figsize=(15, 7.4))
        for ax, how in zip(axs, ("Best", "Median")):
            colour_table(ax, d, "HO", how, ["L", "NP"], 17, 18, 17, 19, band_labels=(how == "Best"))
        fig.legend(**rank_legend(17), loc="lower center", ncol=4, frameon=False, title="rank class within each table",
                   title_fontsize=17)
        fig.subplots_adjust(left=0.01, right=0.995, top=0.99, bottom=0.17, wspace=0.0)
        axs[0].set_position([0.01, 0.17, 0.585, 0.82])
        axs[1].set_position([0.60, 0.17, 0.395, 0.82])
        save(fig, "Figure_3.png", dpi=200)


def figS4_r2_by_band():
    d = model_long_table()
    with plt.rc_context(BASE):
        fig, axs = plt.subplots(2, 1, figsize=(7.0, 8.4))
        for ax, how in zip(axs, ("Best", "Median")):
            colour_table(ax, d, "R2", how, S.MODELS + ["NP"], 9.5, 11, 10.5, 12)
        fig.legend(**rank_legend(10.5), loc="lower center", ncol=4, frameon=False,
                   title="rank class within each table", title_fontsize=10.5)
        fig.tight_layout(rect=(0, 0.06, 1, 1.0), h_pad=0.6)
        save(fig, "Figure_S4.png", dpi=200)


# =============================================================================
# Figs. 4 and S7: coefficients of L
# =============================================================================
COEFS = [(1, "Cooling-phase\noffset", "z", "cooling offset $b$ (z)"),
         (2, "Heating\ntemp. slope", f"z/{DEG}C", f"heating slope $d$ (z/{DEG}C)"),
         (3, "Cooling change\nin temp. slope", f"z/{DEG}C", f"slope change $e$ (z/{DEG}C)")]


def nice_ceiling(x):
    candidates = sorted(step * 10.0 ** e for step in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8) for e in range(-4, 3))
    return next((c for c in candidates if c >= x), x)


def violins(ax, values_by_band, colour, lw95, lw50, dot, dot_lw):
    for pos, v in enumerate(values_by_band):
        if np.ptp(v) > np.finfo(float).eps:
            body = ax.violinplot([v], positions=[pos], widths=0.7, showmeans=False, showmedians=False,
                                 showextrema=False, bw_method=0.35)["bodies"][0]
            body.set_facecolor(colour)
            body.set_edgecolor(colour)
            body.set_alpha(0.6)
            body.set_linewidth(0.8)
            body.set_zorder(1)
        q025, q25, med, q75, q975 = np.percentile(v, [2.5, 25, 50, 75, 97.5])
        ax.plot([pos, pos], [q025, q975], color="black", lw=lw95, solid_capstyle="butt", zorder=5)
        ax.plot([pos, pos], [q25, q75], color="black", lw=lw50, solid_capstyle="butt", zorder=6)
        ax.scatter(pos, med, s=dot, color="white", edgecolor="black", linewidth=dot_lw, zorder=7)


def coefficient_figures():
    draws = np.load(S.BOOTSTRAP_DIR / "L_coefficient_draws.npy")
    cfg = M.load_values(read(S.MEASURES_DIR, "measures_by_recording.csv.gz"))[1]
    k_of = {(m, f): k for k, (m, f) in enumerate(zip(cfg.measure, cfg.fmin_hz))}
    by_band = lambda m, col: [draws[k_of[(m, f)], :, col].astype(float) for f in BANDS]
    bound = lambda v: max(abs(np.percentile(v, 2.5)), abs(np.percentile(v, 97.5)))
    largest = {(m, col): max(bound(v) for v in by_band(m, col)) for m in S.MEASURES for col, *_ in COEFS}
    row_half = {col: nice_ceiling(1.05 * max(largest[(m, col)] for m in S.NONLINEAR)) for col, *_ in COEFS}

    with plt.rc_context(BASE):
        # Fig. S7: all measures (rows) by coefficient (columns)
        W, H, fs_t, fs_l = 7.2, 9.6, 7.5, 8.5
        fig, axes = plt.subplots(len(S.MEASURES), len(COEFS), figsize=(W, H), squeeze=False, sharex=True)
        fig.subplots_adjust(left=1.0 / W, right=1 - 0.08 / W, top=1 - 0.30 / H, bottom=1.05 / H, hspace=0.18,
                            wspace=0.28)
        for r, m in enumerate(S.MEASURES):
            for c, (col, _, _, title) in enumerate(COEFS):
                ax = axes[r, c]
                violins(ax, by_band(m, col), GROUP_COL[S.FAMILY[m]], 0.6, 1.6, 6, 0.4)
                ax.set_ylim(-row_half[col], row_half[col])
                ax.yaxis.set_major_locator(MaxNLocator(nbins=3, symmetric=True))
                ax.axhline(0, color="black", linewidth=0.9, zorder=4)
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=fs_t, length=2, pad=1.5)
                ax.set_xlim(-0.5, len(BANDS) - 0.5)
                if r == 0:
                    ax.set_title(title, fontsize=fs_l)
                if c == 0:
                    ax.set_ylabel("\n".join(NAME[m].rsplit(" ", 1)), fontsize=fs_l)
                if r == len(S.MEASURES) - 1:
                    ax.set_xticks(np.arange(len(BANDS)))
                    ax.set_xticklabels([short_band(f) for f in BANDS], fontsize=fs_t)
        handles = [Patch(facecolor=GROUP_COL[g], alpha=0.6, edgecolor=GROUP_COL[g], label=g) for g in GROUP_COL]
        handles += [Line2D([0], [0], color="black", linewidth=0.6, label="95% interval"),
                    Line2D([0], [0], color="black", linewidth=1.6, label="50% interval"),
                    Line2D([0], [0], marker="o", linestyle="none", markersize=3.5, markerfacecolor="white",
                           markeredgecolor="black", label="median")]
        fig.text(0.5 + 0.46 / W, 0.62 / H, "octave band, lower edge (Hz)", ha="center", va="center", fontsize=fs_l)
        fig.legend(handles=handles, loc="lower center", ncol=len(handles), bbox_to_anchor=(0.5 + 0.46 / W, 0.04 / H),
                   fontsize=fs_l, frameon=False, handlelength=1.6, columnspacing=1.2)
        save(fig, "Figure_S7.png", dpi=300)

        # Fig. 4: LZC78 and RSR
        examples = ("LZC78", "RSR")
        font, panel_h, gap_x, gap_y = 20, 1.7, 1.1, 0.35
        panel_w = 1.5 * panel_h
        left, right, bottom, top = 1.45, 0.25, 1.15, 1.6
        width = left + len(COEFS) * panel_w + (len(COEFS) - 1) * gap_x + right
        height = bottom + len(examples) * panel_h + (len(examples) - 1) * gap_y + top
        fig = plt.figure(figsize=(width, height))
        for r, m in enumerate(examples):
            for c, (col, short, unit, _) in enumerate(COEFS):
                ax = fig.add_axes([(left + c * (panel_w + gap_x)) / width,
                                   (bottom + (len(examples) - 1 - r) * (panel_h + gap_y)) / height,
                                   panel_w / width, panel_h / height])
                violins(ax, by_band(m, col), GROUP_COL[S.FAMILY[m]], 1.0, 3.2, 16, 0.6)
                h = nice_ceiling(1.05 * max(largest[(x, col)] for x in examples))
                ax.set_ylim(-h, h)
                ax.axhline(0, color="black", linewidth=1.6, zorder=4)
                ax.set_xlim(-0.5, len(BANDS) - 0.5)
                ax.yaxis.set_major_locator(MaxNLocator(nbins=4, symmetric=True))
                ax.spines["top"].set_visible(False)
                ax.spines["right"].set_visible(False)
                ax.tick_params(labelsize=font - 1.5)
                ax.set_xticks(np.arange(len(BANDS)))
                ax.set_xticklabels([s if i % 3 == 0 else "" for i, s in enumerate(short_band(f) for f in BANDS)]
                                   if r == len(examples) - 1 else [], fontsize=font - 1)
                if r == 0:
                    ax.set_title(short + f"\n[{unit}]", fontsize=font)
                if c == 0:
                    ax.set_ylabel(NAME[m], fontsize=font + 2, fontweight="bold", labelpad=8)
        handles = [Patch(facecolor=GROUP_COL["nonlinear"], alpha=0.6, edgecolor=GROUP_COL["nonlinear"],
                         label="Bayesian bootstrap distribution"),
                   Line2D([0], [0], color="black", linewidth=1.0, label="95% interval"),
                   Line2D([0], [0], color="black", linewidth=3.2, label="50% interval"),
                   Line2D([0], [0], marker="o", linestyle="none", markersize=7, markerfacecolor="white",
                          markeredgecolor="black", label="Median")]
        fig.legend(handles=handles, loc="lower center", ncol=4, bbox_to_anchor=(0.5, 1.0), fontsize=font - 1,
                   frameon=False)
        fig.text(0.5, 0.15 / height, "Octave band, lower edge [Hz]", ha="center", va="bottom", fontsize=font)
        save(fig, "Figure_4.png", dpi=300, bbox_inches="tight")


# =============================================================================
# Fig. 5: correlation matrix
# =============================================================================
def fig5_correlation_matrix():
    order = ["LZC78", "PE", "RSE4", "RSR", "LogBandEnergy", "ACI", "TemporalEntropy", "SpectralEntropy",
             "SpectralFlatness", "AEI"]
    rho = pd.read_csv(S.MODELS_DIR / "11_correlation_matrix.csv", index_col=0)
    keys = [f"{m} | {b}" for m in order for b in BAND_LABEL]
    values = rho.loc[keys, keys].to_numpy(float)
    labels = [(NAME[m], b) for m in order for b in BAND_LABEL]
    n, boundary, fs = len(keys), 4 * len(BANDS), 23
    cmap = ListedColormap(CORR_COLOURS)
    norm = BoundaryNorm(CORR_LIMITS, cmap.N, clip=True)
    cmap.set_bad("white")
    block = lambda m: (labels.index((NAME[m], BAND_LABEL[0])), labels.index((NAME[m], BAND_LABEL[-1])) + 1)
    with plt.rc_context(BASE):
        fig, ax = plt.subplots(figsize=(15, 14), constrained_layout=True)
        shown = values.copy()
        shown[np.triu_indices(n, 1)] = np.nan
        image = ax.imshow(shown, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
        centres = [np.mean(range(*block(m))) for m in order]
        ax.set_xticks(centres, [NAME[m] for m in order], rotation=45, ha="right", fontsize=fs)
        ax.set_yticks(centres, [NAME[m] for m in order], fontsize=fs)
        for pos in [block(m)[0] for m in order[1:]]:
            ax.plot([-0.5, pos - 0.5], [pos - 0.5, pos - 0.5], color="black", linewidth=0.8)
            ax.plot([pos - 0.5, pos - 0.5], [pos - 0.5, n - 0.5], color="black", linewidth=0.8)
        xs, ys = [-0.5], [-0.5]
        for i in range(n):
            xs += [i + 0.5, i + 0.5]
            ys += [i - 0.5, i + 0.5]
        ax.plot(xs, ys, color="black", linewidth=0.8)
        ax.plot([-0.5, boundary - 0.5], [boundary - 0.5, boundary - 0.5], color="black", linewidth=4.5)
        ax.plot([boundary - 0.5, boundary - 0.5], [boundary - 0.5, n - 0.5], color="black", linewidth=4.5)
        for side in ("top", "right"):
            ax.spines[side].set_visible(False)
        # two enlarged blocks show how the matrix is read
        for (rm, cm), rect in ((("ACI", "ACI"), (0.77, 0.42, 0.20, 0.20)), (("PE", "LZC78"), (0.50, 0.70, 0.24, 0.24))):
            rows, cols = block(rm), block(cm)
            sub = values[rows[0]:rows[1], cols[0]:cols[1]].copy()
            if rm == cm:
                sub[np.triu_indices(sub.shape[0], 1)] = np.nan
            inset = ax.inset_axes(rect)
            inset.imshow(sub, cmap=cmap, norm=norm, aspect="equal", interpolation="nearest")
            ticks, names = np.arange(sub.shape[0]), [short_band(f) for f in BANDS]
            inset.xaxis.tick_top()
            inset.yaxis.tick_right()
            inset.set_xticks(ticks, names, rotation=90, fontsize=fs - 3)
            inset.set_yticks(ticks, names, fontsize=fs - 3)
            inset.tick_params(length=2, pad=2)
            for i in range(sub.shape[0]):
                inset.add_patch(Rectangle((i - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.2))
            for spine in inset.spines.values():
                spine.set_edgecolor(MODEL_COL["NP"])
                spine.set_linewidth(2.5)
            title = (f"{rm} \N{MULTIPLICATION SIGN} {cm}\n(one measure)" if rm == cm
                     else f"{rm} (rows) \N{MULTIPLICATION SIGN}\n{cm} (columns)")
            inset.text(-0.08, 1.0, title, transform=inset.transAxes, ha="right", va="top", fontsize=fs - 2)
            ax.add_patch(Rectangle((cols[0] - 0.5, rows[0] - 0.5), cols[1] - cols[0], rows[1] - rows[0], fill=False,
                                   edgecolor=MODEL_COL["NP"], linewidth=3.0, zorder=6))
            ax.add_artist(ConnectionPatch(xyA=(cols[1] - 0.5, rows[0] - 0.5), coordsA=ax.transData,
                                          xyB=(-0.5, sub.shape[0] - 0.5), coordsB=inset.transData,
                                          color=MODEL_COL["NP"], linewidth=1.8))
        ax.set_xlim(-0.5, n - 0.5)
        ax.set_ylim(n - 0.5, -0.5)
        cb = fig.colorbar(image, ax=ax, boundaries=CORR_LIMITS, ticks=CORR_LIMITS, shrink=0.78, spacing="uniform")
        cb.ax.set_yticklabels([f"{v:.2f}".replace("-", MINUS) for v in CORR_LIMITS])
        cb.set_label("Spearman correlation", fontsize=fs)
        cb.ax.tick_params(labelsize=fs - 1)
        save(fig, "Figure_5.png", dpi=260, bbox_inches="tight")


# =============================================================================
# Figs. S1 and S2: all configurations
# =============================================================================
def figS1_all_configurations():
    fits = pd.read_csv(S.MODELS_DIR / "06_meanfit_fitted.csv.gz")
    per = read(S.MODELS_DIR, "02_cv_per_config.csv")
    occ = read(S.MODELS_DIR, "00_occasions.csv")
    peak_day = occ.time_days[occ.T_mean.idxmax()]
    with plt.rc_context(SMALL):
        fig, axes = plt.subplots(len(S.MEASURES), len(BANDS), figsize=(7.4, 9.6), sharex=True, sharey=True)
        for i, m in enumerate(S.MEASURES):
            for j, f in enumerate(BANDS):
                ax = axes[i, j]
                sel = fits[(fits.measure == m) & (fits.fmin_hz == f)]
                L = sel[sel.model == "L"].sort_values("occasion")
                t = sel[sel.model == "time-only"].sort_values("occasion")
                ax.plot(L.time_days, L.observed_mean, "o", ms=2.2, color="k")
                ax.plot(t.time_days, t.fitted, "--", color=GREY, lw=0.9)
                ax.plot(L.time_days, L.fitted, "-", color=GROUP_COL[S.FAMILY[m]], lw=1.0)
                ax.axvline(peak_day, color=GREY, lw=0.5, ls=":")
                ho = per[(per.measure == m) & (per.fmin_hz == f)].HO_L.iloc[0]
                ax.text(0.97, 0.97, f"{ho:.2f}", transform=ax.transAxes, ha="right", va="top", fontsize=7.5)
                ax.tick_params(labelsize=7.5, length=2, pad=1.5)
                ax.set_yticks([-2, 0, 2])
                ax.set_xticks([0, 5, 10])
                if i == 0:
                    ax.set_title(f"{f}{DASH}{2 * f}", fontsize=8)
                if j == 0:
                    ax.set_ylabel("\n".join(NAME[m].rsplit(" ", 1)), fontsize=8.5)
        fig.supxlabel("time (days)", fontsize=8.5, y=0.002)
        fig.text(0.5, 0.997, "octave band (Hz)", ha="center", va="top", fontsize=8.5)
        fig.tight_layout(h_pad=0.25, w_pad=0.15, rect=(0, 0, 1, 0.985))
        save(fig, "Figure_S1.png", dpi=220)


def figS2_skill_heatmap():
    per = read(S.MODELS_DIR, "02_cv_per_config.csv")
    with plt.rc_context(SMALL):
        fig, axes = plt.subplots(3, 1, figsize=(6.6, 11.2))
        for ax, (title, col) in zip(axes, [("(a) held-out skill of L", "HO_L"),
                                           ("(b) L minus time-only", "gain_L_vs_time-only"),
                                           ("(c) L minus L$_{\\mathrm{time}}$", "gain_L_vs_L_time")]):
            T = per.pivot(index="measure", columns="fmin_hz", values=col).loc[S.MEASURES]
            lim = np.nanmax(np.abs(T.values))
            im = ax.imshow(T.values, cmap="PRGn", vmin=-lim, vmax=lim, aspect="auto")
            for a in range(T.shape[0]):
                for b in range(T.shape[1]):
                    ax.text(b, a, f"{T.values[a, b]:.2f}", ha="center", va="center", fontsize=6.5)
            ax.axhline(3.5, color="k", lw=1.2)
            ax.set_xticks(range(len(BANDS)), [str(b) for b in BANDS], fontsize=7)
            ax.set_yticks(range(len(S.MEASURES)), [NAME[m] for m in S.MEASURES], fontsize=7)
            ax.set_xlabel("band lower edge (Hz)")
            ax.set_title(title, loc="left", fontsize=9)
            fig.colorbar(im, ax=ax, shrink=0.8)
        fig.tight_layout()
        save(fig, "Figure_S2.png", dpi=300)


# =============================================================================
# Fig. S3: held-out skill by band and measure (circular)
# =============================================================================
def figS3_skill_radial():
    models = S.MODELS + ["NP"]
    per = read(S.MODELS_DIR, "02_cv_per_config.csv")
    folds = read(S.MODELS_DIR, "05_heldout_skill_per_bucket.csv")
    point = np.column_stack([per[f"HO_{m}"] for m in models])                                  # K x models
    per_fold = np.stack([folds[folds.model == m].pivot(index="config", columns="heldout_bucket", values="skill")
                         .sort_index().to_numpy().T for m in models], -1)                     # buckets x K x models
    fam, measure = per.family.to_numpy(), per.measure.to_numpy()

    def categories(dimension):
        if dimension == "band":
            keys = [(g, b) for g in ("nonlinear", "conventional") for b in BANDS]
            idx = [np.where((fam == g) & (per.fmin_hz.to_numpy() == b))[0] for g, b in keys]
            labels = [f"{short_band(b)}{DASH}{short_band(2 * b)}" for _, b in keys]
        else:
            keys = [(S.FAMILY[m], m) for m in S.MEASURES]
            idx = [np.where(measure == m)[0] for m in S.MEASURES]
            labels = ["\n".join(NAME[m].rsplit(" ", 1)) for m in S.MEASURES]
        out = []
        for (g, _), ii, lab in zip(keys, idx, labels):
            srt = np.sort(np.median(per_fold[:, ii, :], 1), 0)
            out.append(dict(group=g, label=lab, point=np.median(point[ii], 0),
                            outer=(srt[0], srt[-1]), inner=(srt[1], srt[-2])))
        return out

    cats = {d: categories(d) for d in ("band", "measure")}
    vals = np.concatenate([np.concatenate([c["point"], *c["outer"], *c["inner"]])
                           for c in cats["band"] + cats["measure"]])
    lo, hi = np.percentile(vals, [0.5, 99.5])
    lim = (np.floor(lo / 0.1) * 0.1, np.ceil(hi / 0.1) * 0.1)
    radius = lambda v: 0.08 + 0.84 * np.clip((np.asarray(v, float) - lim[0]) / (lim[1] - lim[0]), 0, 1)
    value = lambda r: lim[0] + (np.asarray(r, float) - 0.08) / 0.84 * (lim[1] - lim[0])
    fs = FS_LABEL + 2
    li = models.index("L")

    def draw(ax, cs, panel, dimension):
        cs = sorted(cs, key=lambda c: c["point"][li])
        n = len(cs)
        ang = np.linspace(0, 2 * np.pi, n, endpoint=False)
        w = 2 * np.pi / n
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        for a, c in zip(ang, cs):
            ax.bar(a, 0.065, width=w * 0.98, bottom=0.965, color=GROUP_COL[c["group"]], alpha=0.96, edgecolor="white",
                   linewidth=0.5, zorder=7)
        off = np.linspace(-0.33, 0.33, len(models)) * w
        r_point = {}
        for j, m in enumerate(models):
            for key, alpha in (("outer", 0.20), ("inner", 0.50)):
                r1 = radius([c[key][0][j] for c in cs])
                r2 = radius([c[key][1][j] for c in cs])
                ax.bar(ang + off[j], np.abs(r2 - r1), bottom=np.minimum(r1, r2), width=w * 0.15, color=MODEL_COL[m],
                       alpha=alpha, edgecolor="none", zorder=2)
            r_point[m] = radius([c["point"][j] for c in cs])
            ax.scatter(ang + off[j], r_point[m], s=fs * 2.3, color=MODEL_COL[m], edgecolor="white", linewidth=1.1,
                       zorder=6)
        for i, a in enumerate(ang):
            ax.plot(a + off, [r_point[m][i] for m in models], color="#292929", lw=2.0, alpha=0.88, zorder=5)
        ax.set_xticks([])
        for a, c in zip(ang, cs):
            deg = np.degrees(a)
            if dimension == "band":
                rot, ha = (90 - deg, "left") if deg <= 180 else (270 - deg, "right")
                ax.text(a, 1.075, c["label"], rotation=rot, rotation_mode="anchor", ha=ha, va="center", fontsize=fs,
                        clip_on=False)
            else:
                flip = 180 if 90 < deg < 270 else 0
                ax.text(a, 1.16 if "\n" in c["label"] else 1.10, c["label"], rotation=-deg + flip,
                        rotation_mode="anchor", ha="center", va="center", fontsize=fs, clip_on=False)
        ax.set_ylim(0, 1.50)
        ticks = np.array([0.248, 0.416, 0.584, 0.752])
        ax.set_yticks(ticks)
        ax.set_yticklabels([""] * 4)
        for k, (r, v) in enumerate(zip(ticks, value(ticks))):
            ax.text(np.deg2rad(np.mod(11.0 * 1.03 ** k, 12) * 30), r, f"{v:.1f}", fontsize=fs * 0.9,
                    fontweight="semibold", ha="center", va="center", zorder=8,
                    bbox=dict(facecolor="white", edgecolor="none", alpha=0.72, pad=0.8))
        ax.grid(alpha=0.45, linewidth=1.2, color="#707070")
        ax.spines["polar"].set_visible(False)
        ax.text(-0.02, 1.02, panel, transform=ax.transAxes, fontsize=fs * 1.5, fontweight="bold", ha="left", va="top")

    with plt.rc_context(BASE):
        fig, axs = plt.subplots(1, 2, figsize=(17, 9.2), subplot_kw={"projection": "polar"})
        draw(axs[0], cats["band"], "A", "band")
        draw(axs[1], cats["measure"], "B", "measure")
        handles = [Line2D([], [], marker="o", ls="", color=MODEL_COL[m], mec="white", ms=17, label=MODEL_LABEL[m])
                   for m in models]
        handles += [Patch(color=GROUP_COL[g], label=g) for g in GROUP_COL]
        fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=len(models) + 2,
                   frameon=False, fontsize=FS_LEGEND + 6, columnspacing=1.4, handletextpad=0.4)
        fig.subplots_adjust(top=0.98, bottom=0.11, left=0.01, right=0.99, wspace=0.0)
        save(fig, "Figure_S3.png", dpi=170)


# =============================================================================
# Figs. S5 and S6: sensitivity of the held-out skill of L
# =============================================================================
SENS_PAIRS = [("Window length and step", "window_low", "window_high", ("7.5 s", "30 s")),
              ("Lambda multiplier", "lambda_low", "lambda_high", ("0.5", "2")),
              ("Lambda exponent", "lambda_power_low", "lambda_power_high", (f"{MINUS}0.25", f"{MINUS}1.0")),
              ("RMS width factor \N{GREEK SMALL LETTER KAPPA}", "rms_cycles_low", "rms_cycles_high", ("2.5", "10")),
              ("LZC alphabet", "lzc_alphabet_low", "lzc_alphabet_high", ("50", "200")),
              ("PE order", "pe_order_low", "pe_order_high", ("3", "5")),
              ("RSE block length", "rse_block_low", "rse_block_high", ("3", "5")),
              ("RSE alphabet", "rse_alphabet_low", "rse_alphabet_high", ("8", "12")),
              ("Combined stress", "combined_low_stress", "combined_high_stress", ("all low", "all high"))]
SENS_MAX = 0.10
SENS_CMAP = LinearSegmentedColormap.from_list("sensitivity", ["#2166ac", "#9ebbd9", "#f4f4f4", "#9fd6a8", "#1a9850"])
FS_CELL, FS_MEAN, FS_GLOBAL, FS_SMALL, FS_HEAD = 6.0, 6.0, 6.3, 7.5, 8.0
ROW_H, TITLE_H, TOP_H, BOTTOM_H, SENS_W = 0.19, 0.30, 0.25, 1.35, 7.2


def sensitivity_panel(ax, D, title, measures, xmode, mean_side):
    """Change in skill per configuration, with a mean column on the inner side and a mean row below."""
    mean_row = len(measures) > 1
    nm = D.shape[0]
    nr = nm + mean_row
    full = np.full((nr, D.shape[1] + 1), np.nan)
    nc = full.shape[1]
    mc = 0 if mean_side == "left" else nc - 1
    bc = slice(1, None) if mean_side == "left" else slice(0, nc - 1)
    full[:nm, bc] = D
    full[:nm, mc] = np.nanmean(D, 1)
    if mean_row:
        full[-1, bc] = np.nanmean(D, 0)
        full[-1, mc] = np.nanmean(D)
    ax.imshow(np.ma.masked_invalid(full), cmap=SENS_CMAP, vmin=-SENS_MAX, vmax=SENS_MAX, aspect="auto")
    for i in range(nr):
        for j in range(nc):
            v = full[i, j]
            if np.isfinite(v):
                is_global = mean_row and i == nr - 1 and j == mc
                is_mean = j == mc or (mean_row and i == nr - 1)
                ax.text(j, i, f"{v:+.3f}" if abs(v) >= 5e-4 else "0", ha="center", va="center",
                        fontsize=FS_GLOBAL if is_global else FS_MEAN if is_mean else FS_CELL,
                        fontweight="bold" if is_mean else "normal",
                        color="white" if abs(v) > 0.7 * SENS_MAX else "black")
    ax.axvline(0.5 if mean_side == "left" else nc - 1.5, color="black", lw=1.0)
    if mean_row:
        ax.axhline(nm - 0.5, color="black", lw=1.0)
    n_nonlinear = sum(m in S.NONLINEAR for m in measures)
    if 0 < n_nonlinear < len(measures):
        ax.axhline(n_nonlinear - 0.5, color="0.4", lw=0.8, ls=(0, (4, 2)))
    bl = [str(b) for b in BANDS]
    ax.set_xticks(range(nc), ["mean"] + bl if mean_side == "left" else bl + ["mean"], rotation=90, fontsize=FS_SMALL)
    ax.tick_params(axis="x", bottom=xmode == "bottom", labelbottom=xmode == "bottom", top=False, labeltop=False,
                   length=2, pad=1.5)
    ax.set_yticks(range(nr), [NAME[m] for m in measures] + (["mean"] if mean_row else []), fontsize=FS_SMALL)
    ax.tick_params(axis="y", length=2, pad=1.5)
    if title:
        ax.set_title(title, fontsize=FS_HEAD, fontweight="bold", loc="left", pad=3)
    for sp in ax.spines.values():
        sp.set_visible(False)


def sensitivity_colourbar(fig, rect):
    sm = plt.cm.ScalarMappable(cmap=SENS_CMAP, norm=plt.Normalize(-SENS_MAX, SENS_MAX))
    cb = fig.colorbar(sm, cax=fig.add_axes(rect), orientation="horizontal")
    cb.set_label("change in held-out skill of L (scenario minus primary setting)", fontsize=FS_SMALL)
    cb.set_ticks([-0.1, -0.05, 0, 0.05, 0.1])
    cb.ax.tick_params(labelsize=FS_SMALL, length=2)


def sensitivity_figures():
    d = read(S.MODELS_DIR, "14_sensitivity_per_config.csv")
    primary = d[d.scenario == "primary"].set_index(["measure", "fmin_hz"]).HO_L

    def delta(name):
        s = d[d.scenario == name].set_index(["measure", "fmin_hz"]).HO_L
        return np.array([[s[(m, b)] - primary[(m, b)] for b in BANDS] for m in S.MEASURES])

    def changed(*names):
        return [i for i in range(len(S.MEASURES)) if any(np.abs(delta(n)[i]).max() > 1e-12 for n in names)]

    def grid_figure(pairs, name):
        blocks = [(t, lo, hi, labs, changed(lo, hi)) for t, lo, hi, labs in pairs]
        rows = [len(keep) + (len(keep) > 1) for *_, keep in blocks]
        height = TOP_H + BOTTOM_H + ROW_H * sum(rows) + TITLE_H * len(blocks)
        fig = plt.figure(figsize=(SENS_W, height))
        gs = fig.add_gridspec(len(blocks), 2, height_ratios=rows, hspace=TITLE_H * len(blocks) / (ROW_H * sum(rows)),
                              wspace=0.06, left=1.05 / SENS_W, right=1 - 0.12 / SENS_W, top=1 - TOP_H / height,
                              bottom=BOTTOM_H / height)
        for bi, (title, lo, hi, labs, keep) in enumerate(blocks):
            for ci, (sc, lab) in enumerate(zip((lo, hi), labs)):
                ax = fig.add_subplot(gs[bi, ci])
                sensitivity_panel(ax, delta(sc)[keep], f"{title}: {lab}", [S.MEASURES[i] for i in keep],
                                  "bottom" if bi == len(blocks) - 1 else None, "right" if ci == 0 else "left")
                if ci == 1:
                    ax.set_yticklabels([])
                    ax.tick_params(axis="y", left=False)
        fig.text(0.5, 0.80 / height, "octave band, lower edge (Hz)", ha="center", va="top", fontsize=FS_SMALL)
        sensitivity_colourbar(fig, [0.30, 0.40 / height, 0.40, 0.08 / height])
        save(fig, name, dpi=300)

    with plt.rc_context(BASE):
        grid_figure(SENS_PAIRS[:3], "Figure_S5a.png")
        grid_figure(SENS_PAIRS[3:], "Figure_S5b.png")
        keep = changed(S.CENTRING_SCENARIO)
        nr = len(keep) + 1
        w, height = 5.0, 0.12 + ROW_H * 1.15 * nr + 1.35
        fig = plt.figure(figsize=(w, height))
        ax = fig.add_axes([1.05 / w, 1.35 / height, 1 - 1.2 / w, ROW_H * 1.15 * nr / height])
        sensitivity_panel(ax, delta(S.CENTRING_SCENARIO)[keep], "", [S.MEASURES[i] for i in keep], "bottom", "right")
        ax.set_xlabel("octave band, lower edge (Hz)", fontsize=FS_SMALL, labelpad=2)
        sensitivity_colourbar(fig, [0.30, 0.40 / height, 0.45, 0.08 / height])
        save(fig, "Figure_S6.png", dpi=300)


# =============================================================================
# Fig. S9: level-invariant control
# =============================================================================
def figS9_loudness_control():
    """RSR in the level-invariant control (quantisers fitted per recording; loudness_control.py): (a) mean trajectory with L and NP in four
    bands, (b) Bayesian-bootstrap distributions of the L coefficients by band."""
    folder = S.LOUDNESS_DIR
    per = read(folder, "per_config.csv")
    per = per[per.quantiser == "recording"].reset_index(drop=True)
    k_of = {(m, f): k for k, (m, f) in enumerate(zip(per.measure, per.fmin_hz))}
    draws = np.load(folder / "L_coefficient_draws.npy")
    col = MODEL_COL["L"]
    with plt.rc_context(LARGE):
        fig = plt.figure(figsize=(18, 10.5))
        outer = fig.add_gridspec(1, 2, width_ratios=[2.15, 1], wspace=0.17, left=0.055, right=0.985, top=0.93,
                                 bottom=0.2)
        ga, gb = outer[0].subgridspec(2, 2, hspace=0.34, wspace=0.22), outer[1].subgridspec(3, 1, hspace=0.18)
        for i, (m, f) in enumerate(LOUDNESS_EXAMPLES):
            ax = fig.add_subplot(ga[i // 2, i % 2])
            bootstrap_row([ax, None], m, f, with_np=True, phases=True, folder=folder)
            x = per.loc[k_of[(m, f)]]
            ax.legend([Line2D([], [], color=col, lw=2.6), Line2D([], [], color=MODEL_COL["NP"], lw=2.6)],
                      [f"L: R{SQ} = {x.R2_L:.2f}", f"NP: R{SQ} = {x.R2_NP:.2f}"], loc="upper right", frameon=True,
                      framealpha=0.85, edgecolor="none", fontsize=FS_LEGEND - 5, handlelength=1.2)
            ax.set_xlabel("time (days)" if i >= 2 else "")
            ax.set_ylabel("z-score" if i % 2 == 0 else "")
            ax.title.set_fontsize(FS_TITLE - 3)
        axes = []
        for r, (c, _, _, title) in enumerate(COEFS):
            ax = fig.add_subplot(gb[r, 0], sharex=axes[0] if axes else None)
            axes.append(ax)
            vals = [draws[k_of[("RSR", f)], :, c].astype(float) for f in BANDS]
            violins(ax, vals, GROUP_COL["nonlinear"], 1.0, 3.2, 16, 0.6)
            ax.axhline(0, color="black", lw=1.4, zorder=4)
            h = max(abs(np.percentile(np.concatenate(vals), [0.5, 99.5]))) * 1.1
            ax.set_ylim(-h, h)
            ax.set_xlim(-0.5, len(BANDS) - 0.5)
            ax.set_ylabel(title, fontsize=FS_LABEL - 4)
            ax.tick_params(labelsize=FS_TICK - 2)
            ax.set_xticks(range(len(BANDS)))
            ax.set_xticklabels([short_band(f) for f in BANDS] if r == len(COEFS) - 1 else [])
        axes[-1].set_xlabel("octave band, lower edge [Hz]", fontsize=FS_LABEL - 3)
        axes[0].set_title("RSR, L coefficients", loc="left", fontsize=FS_TITLE - 3, fontweight="bold")
        fig.text(0.012, 0.955, "a", fontsize=26, fontweight="bold")
        fig.text(0.675, 0.955, "b", fontsize=26, fontweight="bold")
        handles = [Line2D([], [], marker="o", ls="", color="black", ms=7, label="mean of 5 buckets, heating"),
                   Line2D([], [], marker="o", ls="", color="black", mfc="white", mew=1.3, ms=7,
                          label="mean of 5 buckets, cooling"),
                   Line2D([], [], color=col, lw=2.6, label="L fit"),
                   Line2D([], [], color=MODEL_COL["NP"], lw=2.6, label="NP fit")]
        handles += [Line2D([], [], color=DATA_BAR, lw=BAR_WIDTH[lv], alpha=0.55, label=f"data {lv} %") for lv in LEVELS]
        handles += [Patch(color=col, alpha=BAND_ALPHA[lv] + 0.1, label=f"fit {lv} %") for lv in LEVELS]
        fig.legend(handles=handles, loc="lower center", ncol=5, frameon=False, fontsize=FS_LEGEND - 2)
        save(fig, "Figure_S9.png", dpi=200)


# =============================================================================
# Fig. S8: signal-level surrogate effects
# =============================================================================
def figS8_signal_surrogates():
    e = read(S.SURROGATES_DIR, "signal_surrogate_effects.csv").reset_index(drop=True)
    effect = e.effect_sd.to_numpy(float)
    order = np.argsort(-np.abs(effect))
    n = len(e)
    cls = np.zeros(n, dtype=int)
    cls[order[:int(round(0.50 * n))]] = 1
    cls[order[:int(round(0.30 * n))]] = 2
    cls[order[:max(1, int(round(0.10 * n)))]] = 3
    e["rank_class"] = cls * np.sign(effect).astype(int)
    lookup = e.set_index(["measure", "fmin_hz"])
    classes = np.full((len(BANDS), len(S.MEASURES)), np.nan)

    def number(v, decimals):
        v = round(float(v), decimals)
        return f"{0.0:.{decimals}f}" if v == 0 else f"{v:+.{decimals}f}"

    with plt.rc_context(BASE):
        fig, ax = plt.subplots(figsize=(12.5, 6.3))
        for row, f in enumerate(BANDS):
            for col, m in enumerate(S.MEASURES):
                c = lookup.loc[(m, f)]
                level = int(c.rank_class)
                classes[row, col] = level
                point, low, high = float(c.effect_sd), float(c.ci_low), float(c.ci_high)
                weak = low <= 0.0 <= high
                decimals = 2
                while decimals < 4 and any(v != 0 and round(v, decimals) == 0 for v in (point, low, high)):
                    decimals += 1
                dark = abs(level) >= 2
                colour = ("#D9D9D9" if dark else "#8C8C8C") if weak else ("white" if dark else "#171717")
                style = "italic" if weak else "normal"
                ax.text(col, row - 0.13, number(point, decimals), ha="center", va="center", fontsize=11.5,
                        fontweight="normal" if weak else "bold", fontstyle=style, color=colour)
                ax.text(col, row + 0.22, f"[{number(low, decimals)}, {number(high, decimals)}]", ha="center",
                        va="center", fontsize=8.8, fontstyle=style, color=colour)
        cmap = colors.ListedColormap(CORR_COLOURS)
        cmap.set_bad("#E6E6E6")
        ax.imshow(np.ma.masked_invalid(classes), cmap=cmap, norm=colors.BoundaryNorm(np.arange(-3.5, 4.0, 1.0), cmap.N),
                  aspect="auto")
        ax.set_xticks(np.arange(len(S.MEASURES)))
        ax.set_xticklabels(["\n".join(NAME[m].rsplit(" ", 1)) for m in S.MEASURES], fontsize=12)
        ax.set_yticks(np.arange(len(BANDS)))
        ax.set_yticklabels(BAND_LABEL, fontsize=12)
        ax.tick_params(axis="x", top=True, labeltop=True, bottom=False, labelbottom=False, length=0, pad=8)
        ax.tick_params(axis="y", length=0)
        ax.set_xticks(np.arange(-0.5, len(S.MEASURES), 1), minor=True)
        ax.set_yticks(np.arange(-0.5, len(BANDS), 1), minor=True)
        ax.grid(which="minor", color="#B8B8B8", linewidth=0.75)
        ax.tick_params(which="minor", bottom=False, left=False)
        ax.axvline(len(S.NONLINEAR) - 0.5, color="#303030", linewidth=2.4)
        split = len(S.NONLINEAR) / len(S.MEASURES)
        ax.text(split / 2, 1.13, "Nonlinear measures", transform=ax.transAxes, ha="center", fontsize=13,
                fontweight="bold")
        ax.text((1 + split) / 2, 1.13, "Conventional measures", transform=ax.transAxes, ha="center", fontsize=13,
                fontweight="bold")
        legend = [(-3, "< 0, top 10% |effect|"), (-2, "< 0, next 20%"), (-1, "< 0, next 20%"), (0, "smallest 50%"),
                  (1, "> 0, next 20%"), (2, "> 0, next 20%"), (3, "> 0, top 10%")]
        handles = [Patch(facecolor=CORR_COLOURS[lv + 3], edgecolor="#999999", label=lab) for lv, lab in legend]
        fig.legend(handles=handles,
                   loc="lower center", ncol=7, frameon=False, fontsize=11.5, bbox_to_anchor=(0.5, 0.0),
                   handlelength=1.6, columnspacing=1.2)
        fig.subplots_adjust(top=0.84, bottom=0.10, left=0.12, right=0.995)
        save(fig, "Figure_S8.png", dpi=200)


def run():
    S.FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    for draw in (fig1_L_fit_examples, fig2_temperature_surrogates, fig3_heldout_by_band, coefficient_figures,
                 fig5_correlation_matrix, fig6_np_vs_L, figS1_all_configurations, figS2_skill_heatmap,
                 figS3_skill_radial, figS4_r2_by_band, sensitivity_figures, figS8_signal_surrogates,
                 figS9_loudness_control):
        draw()
    print(f"figures in {S.FIGURES_DIR}")


if __name__ == "__main__":
    run()
