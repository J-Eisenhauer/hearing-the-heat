"""Graphical abstract of "Hearing the heat" (three panels: setup - signal - result).

All numbers are read from the outputs of the `analysis` step (results/models); only layout constants are typed in
here. Run after run_all.py:

    python graphical_abstract.py            # writes results/figures/Graphical_abstract.png / .pdf
    python graphical_abstract.py --check    # additionally writes a 50 % PNG to check legibility
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.gridspec import GridSpec
from matplotlib.colors import to_rgb
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.ticker import FixedLocator, MaxNLocator

import settings as S
from figures import GREY, GROUP_COL, MODEL_COL                   # colours of the article figures

for _font in sorted((S.ROOT / "fonts").glob("*.ttf")):          # Inter (SIL OFL, see fonts/OFL.txt)
    matplotlib.font_manager.fontManager.addfont(str(_font))

# ---------------------------------------------------------------- paths
DATA_DIR = S.MODELS_DIR
OUT_DIR = S.FIGURES_DIR

# ---------------------------------------------------------------- selection
MODEL = "L"
WAV_OCCASION = 5  # recording shown as waveform: bucket A at the temperature peak
GROUPS = ("nonlinear", "conventional")
# example series in the centre panel: (measure, fmin, label, marker, marker size, fit line style)
EXAMPLES = (
    ("RSR", 128, "RSR, 128–256 Hz", "o", 6.0, "-"),
    ("RSR", 4096, "RSR, 4–8 kHz", "s", 5.6, (0, (0.01, 1.8))),
)

TITLES = ("Listening to\ndecomposition",
          "Sound tracks phases of rapid\norganic decomposition",
          "Nonlinear measures\npredict unseen buckets")

# ---------------------------------------------------------------- style
TEMP_COL = "#D55E00"  # GA only: temperature
SEC = "0.35"  # secondary text
PHASE_LINE = dict(color=SEC, lw=1.3, ls=":")
# buckets as in the paper: black tapered PP buckets, vermicompost + hay, covered with PE film
BUCKET_FACE = "#262626"
BUCKET_EDGE = "#0D0D0D"
RIM = "#6A6A6A"
INNER = "#151515"
SUBSTRATE = "#7A5C36"
SUBSTRATE_SHADES = ("#5A4026", "#6E5232", "#8B6A3E", "#A9844F", "#B8935A", "#C8A86B")
HAY_SHADES = ("#E6D08A", "#F0E2B0", "#D4B865", "#EAD9A0")
FILM = "#DCE8EF"
LOGO = "#5B3EA6"  # recorder logo (device purple, not a data colour)
NONLIN_SHARE = 4 / 10  # 4 nonlinear of 10 measures (cloud colours)

FIGSIZE = (13, 5.2)
DPI = 300
FS = 14  # axis labels, group labels and key numbers (bold)
FS_TITLE = 14.5
FS_TICK = 12.5  # = FS_SMALL
FS_SMALL = 12.5  # ticks, legends, captions, annotations
INK = "0.2"  # axes, ticks, data marks


def _tint(col, a: float):
    """colour `col` at opacity a over white (as a flat colour)"""
    return tuple(1 - a * (1 - np.array(to_rgb(col))))


ARROW_COL = "0.30"  # arrows between / inside panels
ARROW_L = 0.028  # length of the arrow inside panel 1


def _setup_rc() -> None:
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Inter", "DejaVu Sans"],
        "font.size": FS,
        "axes.titlesize": FS_TITLE,
        "axes.titleweight": "bold",
        "axes.titlepad": 19,
        "axes.labelsize": FS,
        "axes.edgecolor": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "axes.labelcolor": "black",
        "legend.fontsize": FS_SMALL,
        "xtick.labelsize": FS_TICK,
        "ytick.labelsize": FS_TICK,
        "axes.linewidth": 1.0,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
    })


# ======================================================================
# data
# ======================================================================
def load_group_summary(data_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(data_dir / "01_group_summary.csv").rename(columns={
        "R2_median": "meanfit_R2_median", "surrogate_median": "iaaft_null_median",
        "surrogate_q95": "iaaft_null_q95", "surrogate_p": "iaaft_p"})
    df = df[(df["model"] == MODEL) & (df["group"].isin(GROUPS))]
    df = df.set_index("group").loc[list(GROUPS)]
    cols = ["HO_median", "meanfit_R2_median", "iaaft_null_median",
            "iaaft_null_q95", "iaaft_p"]
    return df[cols].astype(float)


def load_null(data_dir: Path) -> pd.DataFrame:
    cols = {f"median_skill_{MODEL}_{g}": f"{MODEL}|{g}" for g in GROUPS}
    return pd.read_csv(data_dir / "07_temperature_surrogates.csv.gz",
                       usecols=list(cols)).rename(columns=cols)


def load_example_series(data_dir: Path) -> list[pd.DataFrame]:
    df = pd.read_csv(data_dir / "06_meanfit_fitted.csv.gz").rename(columns={"occasion": "measurement"})
    out = []
    for measure, fmin, *_ in EXAMPLES:
        d = df[(df["measure"] == measure) & (df["fmin_hz"] == fmin)
               & (df["model"] == MODEL)].sort_values("time_days")
        assert len(d), f"no rows for {measure} / {fmin} Hz / {MODEL}"
        assert not d["measurement"].duplicated().any(), "expected one row per measurement"
        out.append(d.reset_index(drop=True))
    return out


def fmt2(x: float) -> str:
    """Two decimals from the raw value (no double rounding)."""
    return f"{x:.2f}"


def fmt_p(p: float) -> str:
    if p < 0.001:
        return "p < 0.001"
    if p >= 0.05:
        return "n.s."
    return f"p = {p:.3f}"


def report_numbers(summary: pd.DataFrame, null: pd.DataFrame | None) -> None:
    print(f"\n[GA] numbers used (model {MODEL}) - check against the manuscript")
    print(f"{'group':<14}{'HO':>7}{'R2':>7}{'null med':>10}{'null q95':>10}{'p':>12}")
    for g, r in summary.iterrows():
        print(f"{g:<14}{fmt2(r.HO_median):>7}{fmt2(r.meanfit_R2_median):>7}"
              f"{fmt2(r.iaaft_null_median):>10}{fmt2(r.iaaft_null_q95):>10}"
              f"{fmt_p(r.iaaft_p):>12}   (raw p = {r.iaaft_p:.2g})")
    if null is not None:
        print("[GA] cross-check null from file 07:")
        for g in GROUPS:
            v = null[f"{MODEL}|{g}"].dropna().to_numpy()
            med, q95 = np.median(v), np.quantile(v, 0.95)
            ok = (abs(med - summary.loc[g, "iaaft_null_median"]) < 5e-3
                  and abs(q95 - summary.loc[g, "iaaft_null_q95"]) < 5e-3)
            print(f"  {g:<14} n = {len(v):,}  median {med:.3f}  q95 {q95:.3f}"
                  f"  {'OK' if ok else 'MISMATCH vs 01_group_summary'}")
    print()


# ======================================================================
# panel 1 - setup: buckets (drawn), one real recording, cloud of all window values
# ======================================================================
def _texture(ax, clip, rng, xlim, ylim, z, n_crumbs=90, n_hay=8, hay_len=(0.5, 1.1),
             hay_flat=0.35, keep=None) -> None:
    """Vermicompost crumbs and cut hay, clipped to a patch; keep(x, y) -> bool mask
    additionally restricts crumbs and hay ends to a region (e.g. the substrate surface)."""
    keep = keep or (lambda x, y: np.ones_like(x, dtype=bool))
    px = rng.uniform(*xlim, 4 * n_crumbs)
    py = rng.uniform(*ylim, 4 * n_crumbs)
    m = keep(px, py)
    px, py = px[m][:n_crumbs], py[m][:n_crumbs]
    sc = ax.scatter(px, py, s=rng.uniform(2, 14, len(px)),
                    c=[SUBSTRATE_SHADES[i] for i in rng.integers(0, len(SUBSTRATE_SHADES), len(px))],
                    linewidths=0, zorder=z)
    sc.set_clip_path(clip)
    drawn = 0
    for _ in range(20 * n_hay):
        if drawn == n_hay:
            break
        xa, ya = rng.uniform(*xlim), rng.uniform(*ylim)
        ang = rng.uniform(-np.pi, np.pi)
        ln = rng.uniform(*hay_len)
        xb, yb = xa + ln * np.cos(ang), ya + hay_flat * ln * np.sin(ang)
        if not keep(np.array([xa, xb]), np.array([ya, yb])).all():
            continue
        (line,) = ax.plot([xa, xb], [ya, yb], color=HAY_SHADES[rng.integers(0, len(HAY_SHADES))],
                          lw=1.3, solid_capstyle="round", zorder=z + 0.01)
        line.set_clip_path(clip)
        drawn += 1


def _draw_bucket(ax: plt.Axes, x0: float, y0: float, rng: np.random.Generator, z: float,
                 cutaway: bool = False, wt: float = 3.7, wb: float = 3.05, h: float = 2.85,
                 rim_h: float = 0.95, gap: float = 0.45, v: dict | None = None) -> None:
    """Wide black bucket seen slightly from above, filled to below the rim, PE film on top.
    gap = distance from rim to substrate surface; cutaway=True opens a window into the content."""
    v = v or {}
    cx = x0 + wt / 2
    for sc, al in ((1.14, 0.06), (1.05, 0.09)):  # soft cast shadow on the ground
        ax.add_patch(mpatches.Ellipse((cx + 0.14, y0 - 0.06), wb * sc, rim_h * wb / wt * sc,
                                      facecolor="black", alpha=al, edgecolor="none", zorder=z - 0.5))
    ax.add_patch(mpatches.Ellipse((cx, y0), wb, rim_h * wb / wt, facecolor=BUCKET_FACE,
                                  edgecolor=BUCKET_EDGE, lw=1.0, zorder=z))
    body = mpatches.Polygon(
        [(cx - wb / 2, y0), (cx + wb / 2, y0), (x0 + wt, y0 + h), (x0, y0 + h)],
        closed=True, facecolor=BUCKET_FACE, edgecolor="none", zorder=z + 0.1)
    ax.add_patch(body)
    g = np.linspace(0, 1, 96)[None, :]  # cylinder shading: dark edges, soft lit band right of centre
    prof = v.get("bright", 0.0) + 0.095 + 0.19 * np.exp(-((g - 0.58) / 0.24) ** 2) - 0.05 * np.exp(-((g - 0.02) / 0.06) ** 2)
    shade = ax.imshow(prof, extent=(x0, x0 + wt, y0, y0 + h), origin="lower", cmap="gray", vmin=0,
                      vmax=1, aspect="auto", interpolation="bilinear", zorder=z + 0.12)
    shade.set_clip_path(body)
    for xa, xb in ((cx - wb / 2, x0), (cx + wb / 2, x0 + wt)):
        ax.plot([xa, xb], [y0, y0 + h], color=BUCKET_EDGE, lw=1.0, zorder=z + 0.2)
    ax.add_patch(mpatches.Polygon(  # soft highlight
        [(cx + wb * 0.22, y0 + 0.15), (cx + wb * 0.30, y0 + 0.15),
         (x0 + wt * 0.82, y0 + h - 0.2), (x0 + wt * 0.74, y0 + h - 0.2)],
        closed=True, facecolor="white", alpha=v.get("hl", 0.12), edgecolor="none", zorder=z + 0.15))
    for sx_, sy_, sl_, sa_ in v.get("scuffs", []):  # faint scratches: each bucket gets its own
        xs = cx - wb / 2 + sx_ * wt * 0.85 + (sy_ * (x0 - (cx - wb / 2)))
        ax.plot([xs, xs + sl_ * 0.9], [y0 + 0.2 + sy_ * (h - 0.6), y0 + 0.2 + sy_ * (h - 0.6) + sl_ * 0.25],
                color="white", alpha=sa_, lw=0.8, solid_capstyle="round", zorder=z + 0.16)

    if cutaway:
        yb, yt, m = y0 + 0.42, y0 + h - gap, 0.35
        def wall(y):
            f = (y - y0) / h
            return (cx - wb / 2 + f * (x0 - (cx - wb / 2)) + m,
                    cx + wb / 2 + f * (x0 + wt - (cx + wb / 2)) - m)
        (l0, r0), (l1, r1) = wall(yb), wall(yt)
        split = 0.55
        xs0, xs1 = l0 + split * (r0 - l0), l1 + split * (r1 - l1)
        a_, b_ = (r0 - l0) / 2, (r0 - l0) / 2 * rim_h / wt  # bottom edge follows the base ellipse
        xb = np.linspace(l0, xs0, 40)
        bot = [(x, yb - b_ * np.sqrt(max(0.0, 1 - ((x - (l0 + r0) / 2) / a_) ** 2))) for x in xb]
        win = mpatches.Polygon(bot + [(xs1, yt + gap * 0.6), (l1, yt + gap * 0.6)], closed=True,
                               facecolor=INNER, edgecolor=RIM, lw=1.0, zorder=z + 0.16)
        ax.add_patch(win)
        fill = mpatches.Polygon(bot + [(xs1, yt), (l1, yt)], closed=True, facecolor=SUBSTRATE,
                                edgecolor="none", zorder=z + 0.17)
        ax.add_patch(fill)
        fill.set_clip_path(win)
        _texture(ax, fill, rng, (l1, r1), (yb, yt), z + 0.18, n_crumbs=200, n_hay=14,
                 hay_len=(0.35, 0.8), hay_flat=0.8)

    top = (cx, y0 + h)
    ax.add_patch(mpatches.Ellipse(top, wt + 0.16, rim_h + 0.16, facecolor=BUCKET_FACE,  # rolled lip
                                  edgecolor="#5A5A5A", lw=0.9, zorder=z + 0.25))
    rim = mpatches.Ellipse(top, wt, rim_h, facecolor=INNER, edgecolor="none", zorder=z + 0.3)
    ax.add_patch(rim)
    shrink = 1 - gap / h * (wt - wb) / wt  # bucket narrows downwards
    surf = mpatches.Ellipse((cx, y0 + h - gap), wt * 0.92 * shrink, rim_h * 0.85 * shrink,
                            facecolor=tuple(np.clip(np.array(to_rgb(SUBSTRATE)) * v.get("tint", 1.0), 0, 1)), edgecolor="none", zorder=z + 0.4)
    ax.add_patch(surf)
    surf.set_clip_path(rim)
    sw, sh, sy = wt * 0.92 * shrink / 2, rim_h * 0.85 * shrink / 2, y0 + h - gap
    rw, rh, ry = wt / 2, rim_h / 2, y0 + h
    def on_surface(x, y):  # inside the substrate surface and inside the rim opening
        return (((x - cx) / sw) ** 2 + ((y - sy) / sh) ** 2 < 0.92) & \
               (((x - cx) / rw) ** 2 + ((y - ry) / rh) ** 2 < 0.95)
    _texture(ax, rim, rng, (cx - sw, cx + sw), (sy - sh, sy + sh), z + 0.5, keep=on_surface,
             n_hay=v.get("hay_n", 8), hay_len=v.get("hay_len", (0.5, 1.1)))
    if not cutaway:  # PE film on the other buckets; the measured (front) bucket is open
        ax.add_patch(mpatches.Ellipse(top, wt, rim_h, facecolor=FILM, alpha=v.get("film", 0.22),
                                      edgecolor="none", zorder=z + 0.6))
        a0, aw, ax_ = v.get("arc", (110, 55, -0.08))
        ax.add_patch(mpatches.Arc((cx + wt * ax_, y0 + h + 0.02), wt * 0.62, rim_h * 0.5,
                                  theta1=a0, theta2=a0 + aw, color="white", lw=1.4, alpha=0.8,
                                  zorder=z + 0.65))
        for wx, wy, wl_, wa in v.get("wrinkles", []):  # faint creases in the film
            ax.plot([cx + wx * wt, cx + wx * wt + wl_], [y0 + h + wy * rim_h, y0 + h + wy * rim_h + 0.05 * wl_],
                    color="white", alpha=wa, lw=0.8, solid_capstyle="round", zorder=z + 0.62)
    ax.add_patch(mpatches.Ellipse(top, wt, rim_h, facecolor="none", edgecolor=RIM,
                                  lw=1.3, zorder=z + 0.7))


def _draw_recorder(ax: plt.Axes, x0: float, y0: float, w: float, h: float, z: float) -> None:
    """Generic white field recorder with an own logo (purple rounded square with a white sine wave)."""
    ax.add_patch(mpatches.FancyBboxPatch((x0, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.22",
                                         facecolor="#EEF2F3", edgecolor="0.55", lw=1.2, zorder=z))
    ax.add_patch(mpatches.FancyBboxPatch((x0 + 0.08, y0 + h * 0.62), w - 0.16, h * 0.32,
                                         boxstyle="round,pad=0.01,rounding_size=0.12",
                                         facecolor="#F8FAFB", edgecolor="0.7", lw=0.8, zorder=z + 0.1))
    cx, cy, r = x0 + w / 2, y0 + h * 0.36, 0.27
    ax.add_patch(mpatches.FancyBboxPatch((cx - r, cy - r), 2 * r, 2 * r,
                                         boxstyle=f"round,pad=0,rounding_size={0.25 * r}",
                                         facecolor=LOGO, edgecolor="none", zorder=z + 0.2))
    xx = np.linspace(-0.7 * r, 0.7 * r, 60)
    ax.plot(cx + xx, cy + 0.45 * r * np.sin(xx / r * 2.6 * np.pi) * np.hanning(60),
            color="white", lw=1.6, zorder=z + 0.3)
    ax.add_patch(mpatches.Rectangle((x0 + 0.47, y0 + h - 0.02), 0.12, 0.14, facecolor="#C9A227",
                                    edgecolor="0.3", lw=0.6, zorder=z + 0.05))  # connector


def draw_buckets(ax: plt.Axes) -> plt.Text:
    ax.axis("off")
    rng = np.random.default_rng(7)
    n, dx = 5, 2.5
    dy = 0.75  # rows staggered by about 17 degrees
    for i in reversed(range(n)):  # back (upper left) to front (lower right)
        vr = np.random.default_rng(300 + i)  # a hint of individuality per bucket
        v = dict(bright=vr.uniform(-0.014, 0.014), hl=vr.uniform(0.09, 0.15), tint=vr.uniform(0.9, 1.1),
                 hay_n=int(vr.integers(6, 12)), hay_len=(0.4 + vr.uniform(0, 0.15), 0.95 + vr.uniform(0, 0.3)),
                 film=vr.uniform(0.15, 0.28),
                 arc=(vr.uniform(98, 135), vr.uniform(40, 62), vr.uniform(-0.14, 0.0)),
                 scuffs=[(vr.uniform(0.1, 0.8), vr.uniform(0.1, 0.8), vr.uniform(0.15, 0.4), vr.uniform(0.07, 0.16))
                         for _ in range(vr.integers(2, 5))],
                 wrinkles=[(vr.uniform(-0.3, 0.3), vr.uniform(-0.25, 0.1), vr.uniform(0.25, 0.6), vr.uniform(0.25, 0.5))
                           for _ in range(vr.integers(2, 4))])
        _draw_bucket(ax, 0.2 + (n - 1 - i) * dx, 0.6 + i * dy, rng, z=10 * (n - i), cutaway=(i == 0), v=v)
    _draw_recorder(ax, x0=7.4, y0=-0.05, w=2.0, h=1.25, z=80)
    # contact sensor in the middle of the front bucket, cable over the rim to the recorder
    cx, ys = 0.2 + (n - 1) * dx + 3.7 / 2, 0.6 + 2.85 - 0.45
    ax.add_patch(mpatches.Ellipse((cx, ys), 0.32, 0.12, facecolor="0.55", edgecolor="0.2",
                                  lw=0.8, zorder=75))
    cable = MplPath([(cx, ys), (cx - 0.1, ys + 1.6), (8.2, 4.1), (7.93, 1.2)],
                    [MplPath.MOVETO, MplPath.CURVE4, MplPath.CURVE4, MplPath.CURVE4])
    ax.add_patch(mpatches.PathPatch(cable, facecolor="none", edgecolor="#808080", lw=2.0,
                                    capstyle="round", zorder=76))
    ax.set_xlim(-0.1, 0.2 + (n - 1) * dx + 3.95)
    ax.set_ylim(-0.1, 0.6 + (n - 1) * dy + 2.85 + 0.75)
    ax.set_aspect("equal", adjustable="datalim")
    cap = ax.text(0.0, 0.9, "5 buckets,\nvermicompost & hay", transform=ax.transData,
                  fontsize=FS_SMALL - 1.0, color=SEC, ha="left", va="center", linespacing=1.25)
    ax.set_title(TITLES[0])
    return cap


def read_envelope(wav: Path, n_bins: int = 800) -> tuple[np.ndarray, np.ndarray, float]:
    """Min/max amplitude envelope of a WAV file (stdlib only), normalised to the peak."""
    with wave.open(str(wav), "rb") as w:
        nch, sw, sr, nfr = w.getnchannels(), w.getsampwidth(), w.getframerate(), w.getnframes()
        per = max(1, nfr // n_bins)
        lo, hi = np.empty(n_bins), np.empty(n_bins)
        for k in range(n_bins):
            raw = w.readframes(per)
            if sw == 3:
                b = np.frombuffer(raw, np.uint8).reshape(-1, 3)
                x = (b[:, 0].astype(np.int32) | (b[:, 1].astype(np.int32) << 8)
                     | (b[:, 2].astype(np.int8).astype(np.int32) << 16))
            else:
                x = np.frombuffer(raw, {1: np.uint8, 2: np.int16, 4: np.int32}[sw]).astype(np.int64)
            x = x[::nch]
            lo[k], hi[k] = (x.min(), x.max()) if len(x) else (0, 0)
    m = max(abs(lo).max(), abs(hi).max()) or 1
    return lo / m, hi / m, nfr / sr


def draw_recording(ax: plt.Axes, n_windows: int, n_rec: int, dur: float = 600.0) -> plt.Text:
    """Schematic recording (synthetic, for illustration only): quiet background with
    sporadic decaying clicks and short bursts, as heard in the substrate. Returns the caption."""
    rng = np.random.default_rng(5)
    fs = 40  # samples per second of the drawing, not of the audio
    n = int(dur * fs)
    x = 0.06 * rng.normal(size=n)
    for _ in range(70):  # clicks
        i0, a, tau = rng.integers(0, n - 60), rng.uniform(0.25, 1.0) ** 1.5, rng.uniform(2, 8)
        k = np.arange(60)
        x[i0:i0 + 60] += a * np.exp(-k / tau) * np.sign(rng.normal(size=60)) * rng.uniform(0.4, 1, 60)
    for _ in range(6):  # short rustling bursts
        i0, ln = rng.integers(0, n - 400), rng.integers(150, 400)
        x[i0:i0 + ln] += 0.22 * np.hanning(ln) * rng.normal(size=ln)
    x /= np.abs(x).max()
    t = np.linspace(0, dur, n)
    ax.plot(t, x, color=INK, lw=0.35)
    for k in range(n_windows + 1):  # window ticks below the trace
        ax.plot([k * dur / n_windows] * 2, [1.15, 1.50], color=SEC, lw=0.7)
    ax.set_xlim(0, dur)
    ax.set_ylim(-1.05, 1.52)
    ax.axis("off")
    return ax.text(0.5, -0.06, f"{n_rec} measurements\nof {n_windows} \u00d7 {S.WINDOW_S:.0f} s windows",
                   transform=ax.transAxes, fontsize=FS_SMALL - 2.0, color=SEC, ha="center", va="top",
                   linespacing=1.1)


def draw_cloud(ax: plt.Axes, n_rec: int, n_windows: int, n_measures: int, n_bands: int):
    """Two Gaussian clouds of one point per 15 s window (recordings x 40), multiplied by the number of
    nonlinear (green) and conventional (purple) measure-band configurations. The labels below
    are figure-level texts (placed by _layout) so that nothing can touch the clouds."""
    n_win = n_rec * n_windows
    k_nl = len(S.NONLINEAR) * n_bands
    k_cv = len(S.CONVENTIONAL) * n_bands
    assert k_nl + k_cv == n_measures * n_bands
    rng = np.random.default_rng(11)
    cx0, sx, sy = 1.2, 0.40, 0.34
    kk = []
    for cx, g, k in ((-cx0, "nonlinear", k_nl), (cx0, "conventional", k_cv)):
        xy = rng.normal(0, 1, (n_win, 2)) * [sx, sy]
        ax.scatter(cx + xy[:, 0], xy[:, 1], s=0.55, c=GROUP_COL[g], alpha=0.55, linewidths=0,
                   rasterized=True)
        kk.append((ax.figure.text(0, 0, f"\u00d7 {k}", ha="center", va="bottom", fontsize=FS,
                                  fontweight="bold", color=GROUP_COL[g]), cx))
    fsb = FS_SMALL - 2.5
    mk = lambda txt, col, bold=False: ax.figure.text(
        0, 0, txt, ha="left", va="bottom", fontsize=fsb, color=col,
        fontweight="bold" if bold else "normal")
    band = [mk(str(len(S.NONLINEAR)), GROUP_COL["nonlinear"], True), mk("&", SEC),
            mk(str(len(S.CONVENTIONAL)), GROUP_COL["conventional"], True),
            mk(f"measures in {n_bands} frequency bands", SEC)]
    n = n_win * (k_nl + k_cv)
    total = ax.figure.text(0, 0, "= " + f"{n:,}".replace(",", " ") + " values", ha="center",
                           va="bottom", fontsize=FS - 1.0, fontweight="bold", color="black")
    ax.set_xlim(-2.0, 2.0)
    ax.set_ylim(-1.0, 1.0)
    ax.set_aspect(1.5, adjustable="datalim")  # slightly taller than wide: uses the free height
    ax.axis("off")
    return n, dict(kk=kk, band=band, total=total)


# ======================================================================
# panel 2 - signal (real data): temperature on top, sound measures below
# ======================================================================
def draw_signal(axT: plt.Axes, axM: plt.Axes, series: list[pd.DataFrame]) -> None:
    ref = series[0]
    t = ref["time_days"].to_numpy()
    T = ref["T_mean"].to_numpy()
    i_peak = int(np.nanargmax(T))  # cooling starts after the measurement with max T
    t_peak = t[i_peak]
    heating = np.arange(len(t)) <= i_peak
    x0, x1 = t.min() - 1.9, t.max() + 0.4

    # --- temperature
    axT.plot(t, T, color=TEMP_COL, lw=2.2)
    axT.axvline(t_peak, ymax=0.74, **PHASE_LINE)  # ends below the labels
    Tlo, Thi = np.nanmin(T), np.nanmax(T)
    axT.set_ylim(Tlo - 0.1 * (Thi - Tlo), Thi + 0.45 * (Thi - Tlo))
    axT.yaxis.set_major_locator(FixedLocator([v for v in (25, 30, 35, 40) if Tlo - 1 <= v <= Thi + 1]))
    axT.set_ylabel("T (°C)")
    axT.tick_params(axis="x", labelbottom=False)
    lab = dict(fontsize=FS_SMALL, color=SEC, style="italic", va="top", transform=axT.get_xaxis_transform())
    axT.text(t_peak + 0.05, 0.97, "heating", ha="right", **lab)
    axT.text(t_peak + 0.5, 0.97, "cooling", ha="left", **lab)

    # --- sound measures
    allv = []
    for s, (_, _, _, marker, ms, ls) in zip(series, EXAMPLES):
        assert np.allclose(s["time_days"].to_numpy(), t), "example series on different time grids"
        obs, fit = s["observed_mean"].to_numpy(), s["fitted"].to_numpy()
        allv += [obs, fit]
        axM.plot(t, fit, color=MODEL_COL[MODEL], lw=2.4, ls=ls, zorder=3, solid_capstyle="round",
                 dash_capstyle="round")
        axM.plot(t[heating], obs[heating], marker, ls="none", ms=ms, mfc="black", mec="black", zorder=4)
        axM.plot(t[~heating], obs[~heating], marker, ls="none", ms=ms, mfc="white", mec="black",
                 mew=1.2, zorder=4)
    axM.axvline(t_peak, ymin=0.30, **PHASE_LINE, zorder=0)  # starts above the legend
    allv = np.concatenate(allv)
    lo, hi = np.nanmin(allv), np.nanmax(allv)
    pad = 0.08 * (hi - lo)
    axM.set_ylim(lo - pad - 0.42 * (hi - lo), hi + pad)
    ticks = [v for v in (-2, 0, 2) if lo - pad <= v <= hi + pad]
    axM.yaxis.set_major_locator(FixedLocator(ticks))
    axM.xaxis.set_major_locator(MaxNLocator(4, integer=True))
    axM.set_xlabel("Days", labelpad=2)
    axM.set_ylabel("Sound metric (z)")
    fig_align = (axT, axM)  # y-labels aligned below

    for a in (axT, axM):
        a.set_xlim(x0, x1)
        for sp in ("top", "right"):
            a.spines[sp].set_visible(False)

    # marker = observed mean, line = model fit (solid / dashed pairs the fit with its series)
    handles = [Line2D([], [], color=MODEL_COL[MODEL], lw=2.4, ls=ls, marker=mk, ms=ms,
                      mfc="black", mec="black", dash_capstyle="round")
               for _, _, _, mk, ms, ls in EXAMPLES]
    labels = [lbl for _, _, lbl, *_ in EXAMPLES]
    axM.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.03, 0.03), ncol=1,
               labelspacing=0.25, frameon=False, fontsize=FS_SMALL, handlelength=2.0,
               handletextpad=0.5, borderpad=0.2, borderaxespad=0.1, labelcolor=SEC)
    axT.figure.align_ylabels(fig_align)
    axT.set_title(TITLES[1])


# ======================================================================
# panel 3 - result (real data)
# ======================================================================
BAR_W = 0.36
GROUP_STEP = 1.45  # spacing between the two groups (labels must not touch)


def draw_result(ax: plt.Axes, summary: pd.DataFrame, xpad: float = 0.55) -> None:
    for i, g in enumerate(GROUPS):
        x = i * GROUP_STEP
        r, col = summary.loc[g], GROUP_COL[g]
        ax.bar(x, r.meanfit_R2_median, BAR_W, facecolor="none", edgecolor=col,
               lw=2, ls="--", zorder=2)
        ax.bar(x, r.HO_median, BAR_W, color=col, edgecolor="none", zorder=3)
        ax.text(x + BAR_W / 2 + 0.11, r.HO_median, fmt2(r.HO_median), ha="left",
                va="center", fontsize=FS, fontweight="bold", color=col, zorder=4)
        ax.text(x, r.meanfit_R2_median + 0.015, fmt2(r.meanfit_R2_median),
                ha="center", va="bottom", fontsize=FS, color=col)
        ax.text(x, r.meanfit_R2_median + 0.11, fmt_p(r.iaaft_p), ha="center",
                va="bottom", fontsize=FS, color="black")

    ax.set_xlim(-xpad, (len(GROUPS) - 1) * GROUP_STEP + 0.6)
    ax.set_ylim(0, 1.22)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    ax.set_ylabel("Skill / R²")
    ax.set_xticks([i * GROUP_STEP for i in range(len(GROUPS))])
    ax.set_xticklabels(GROUPS, fontsize=FS, fontweight="bold")
    for lbl, g in zip(ax.get_xticklabels(), GROUPS):
        lbl.set_color(GROUP_COL[g])
    ax.tick_params(axis="x", length=0, pad=6)
    for s in ("top", "right", "bottom"):
        ax.spines[s].set_visible(False)
    ax.axhline(0, color="black", lw=1.0, zorder=5)

    cols = [GROUP_COL[g] for g in GROUPS]
    h_r2 = tuple(mpatches.Patch(facecolor="none", edgecolor=c, lw=1.6, ls="--") for c in cols)
    h_ho = tuple(mpatches.Patch(facecolor=c, edgecolor="none") for c in cols)
    ax.legend([h_r2, h_ho], ["in-sample R²", "held-out skill"],
              handler_map={tuple: HandlerTuple(ndivide=None, pad=0.2)},
              loc="upper right", bbox_to_anchor=(1.04, 1.04), frameon=False,
              fontsize=FS_SMALL, handlelength=1.8, handletextpad=0.5, labelspacing=0.3)
    ax.set_title(TITLES[2])


# ======================================================================
# assembly
# ======================================================================
def _layout(fig, axB, axR, axC, axM, ax3, cap, rec_txt, ct, gap: float = 4.0) -> float:
    """(1) Everything that hangs below the panels (x label 'Days', group labels of panel 3, the total of
    panel 1) ends on ONE line: the lowest of the axis descriptors. (2) Panel 1, bottom row: the label
    stack under the clouds is built upwards from that line and the clouds are fitted between the bucket
    caption and the stack. The waveform (caption above it) is NOT tied to the line: it sits level with
    the clouds, so the arrow between them is horizontal. Returns the line (figure fraction)."""
    r = fig.canvas.get_renderer()
    W, H = fig.bbox.width, fig.bbox.height
    pt = 72.0 / fig.dpi
    y0 = {"days": axM.xaxis.label.get_window_extent(r).y0,
          "groups": min(l.get_window_extent(r).y0 for l in ax3.get_xticklabels())}
    base = min(y0.values())
    print("[GA] bottom line (px):", {k: round(v, 1) for k, v in y0.items()}, "->", round(base, 1))
    axM.xaxis.labelpad = 2 + (y0["days"] - base) * pt
    ax3.tick_params(axis="x", pad=6 + (y0["groups"] - base) * pt)
    h = lambda t: t.get_window_extent(r).height

    # right block, built upwards: total, coloured band line, x32 / x48, clouds
    pieces = ct["band"]
    gsp = 0.30 * pieces[0].get_fontsize() * fig.dpi / 72  # width of a space (px)
    widths = [p_.get_window_extent(r).width for p_ in pieces]
    wl = sum(widths) + gsp * (len(pieces) - 1)
    bc = axC.get_window_extent(r)
    cxm = 0.5 * (bc.x0 + bc.x1)
    y = base
    ct["total"].set_position((cxm / W, y / H))
    y += h(ct["total"]) + gap
    x = cxm - wl / 2
    for p_, w_ in zip(pieces, widths):
        p_.set_position((x / W, y / H))
        x += w_ + gsp
    y += max(h(p_) for p_ in pieces) + gap
    y_kk = y
    y += max(h(t) for t, _ in ct["kk"]) + 1.5 * gap
    cap_bot = axB.get_window_extent(r).y0 - 6  # caption now sits inside panel 1
    top = cap_bot - 1.5 * gap
    pos = axC.get_position()
    hbox = min(top - y, pos.width * W * 0.75)  # 0.75 = box height / width the cloud needs
    axB.apply_aspect()
    x_end = axB.transData.transform((0.2 + 4 * 2.5 + 3.7, 0))[0]  # right rim of the front bucket
    axC.set_position([(x_end - pos.width * W) / W, y / H, pos.width, hbox / H])  # blobs end there
    axC.apply_aspect()
    for t, cx in ct["kk"]:
        t.set_position((axC.transData.transform((cx, 0))[0] / W, y_kk / H))

    # left block: waveform midline = cloud centre, caption above the waveform
    yc = y + hbox / 2
    rec_txt.set_transform(fig.transFigure)
    rec_txt.set_ha("left")
    rec_txt.set_va("bottom")
    rec_txt.set_multialignment("left")
    hc = h(rec_txt)
    hR_px = 0.16 * H  # 0.408 / 0.592: position of the waveform midline in its axes
    y_m = min(yc, cap_bot - 8 - 6 - hc - 0.592 * hR_px)  # as high as the caption allows, never above the cloud centre
    y_m = max(y_m, yc - 0.3 * hbox)  # ... and not much below it
    y_m -= 0.016 * H  # measurements text and waveform slightly lower
    hR_px = min(hR_px, (cap_bot - 8 - 6 - hc - y_m) / 0.592)
    pr = axR.get_position()
    y_axR = (y_m - 0.408 * hR_px) / H
    axR.set_position([pr.x0, y_axR, pr.width, hR_px / H])
    rec_txt.set_position((pr.x0, y_axR + hR_px / H + 6 / H))
    # bucket caption: same left indent as the measurements text
    cy = cap.get_transform().transform(cap.get_position())[1] / H
    cap.set_transform(fig.transFigure)
    cap.set_position((pr.x0, cy))
    return base / H


def _arrow(fig, xc: float, y: float, length: float = ARROW_L, ms: float = 24) -> None:
    fig.patches.append(mpatches.FancyArrowPatch(
        (xc - length / 2, y), (xc + length / 2, y), transform=fig.transFigure,
        arrowstyle="-|>", mutation_scale=ms, lw=2.6, color=ARROW_COL))


def _centre_titles(axes, top_edge: float) -> None:
    """Vertically centre each heading between the top edge of its panel and the top of its plot."""
    for a in axes:
        bb = a.get_position()
        yc = (top_edge + bb.y1) / 2
        a.set_title(a.get_title(), y=(yc - bb.y0) / bb.height, va="center", pad=0)


def _make_axes(fig):
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.25, 1.1, 0.88], wspace=0.42,
                  left=0.032, right=0.985, top=0.80, bottom=0.13)
    p1 = gs[0, 0].subgridspec(2, 1, height_ratios=[1.35, 1], hspace=0.12)
    p1b = p1[1].subgridspec(1, 2, width_ratios=[0.72, 1.45], wspace=0.18)
    axB, axR, axC = (fig.add_subplot(p1[0]), fig.add_subplot(p1b[0]), fig.add_subplot(p1b[1]))
    sub = gs[0, 1].subgridspec(2, 1, height_ratios=[1, 2.1], hspace=0.12)
    axT = fig.add_subplot(sub[0])
    axM = fig.add_subplot(sub[1], sharex=axT)
    return axB, axR, axC, axT, axM, fig.add_subplot(gs[0, 2])


def graphical_abstract(data_dir: Path = DATA_DIR, out_dir: Path = OUT_DIR, check: bool = False) -> plt.Figure:
    _setup_rc()
    summary = load_group_summary(data_dir)
    null = load_null(data_dir)
    series = load_example_series(data_dir)
    report_numbers(summary, null)
    rec = pd.read_csv(S.RECORDINGS_FILE)
    n_windows = int(round(S.EXCERPT_S / S.WINDOW_S))

    fig = plt.figure(figsize=FIGSIZE, facecolor="white")
    axB, axR, axC, axT, axM, ax3 = _make_axes(fig)
    cap = draw_buckets(axB)
    rec_txt = draw_recording(axR, n_windows, len(rec))
    n_vals, ct = draw_cloud(axC, len(rec), n_windows, len(S.NONLINEAR) + len(S.CONVENTIONAL), len(S.BANDS))
    draw_signal(axT, axM, series)
    draw_result(ax3, summary)
    fig.canvas.draw()
    _layout(fig, axB, axR, axC, axM, ax3, cap, rec_txt, ct)
    fig.canvas.draw()
    print(f"[GA] cloud: {n_vals:,} window values ({len(rec)} recordings)")

    inv = fig.transFigure.inverted()
    r = fig.canvas.get_renderer()
    y_mid = (axM.get_position().y0 + axT.get_position().y1) / 2
    _centre_titles((axB, axT, ax3), 1.0)
    pR, pC = axR.get_position(), axC.get_position()
    _arrow(fig, (pR.x1 + pC.x0 + 0.06 * pC.width) / 2, pR.y0 + 0.408 * pR.height)

    def ext(artists):
        return [a.get_window_extent(r).transformed(inv) for a in artists]
    b1 = max(max(a.get_tightbbox(r).transformed(inv).x1 for a in (axB, axC)),
             max(e.x1 for e in ext([rec_txt, ct["total"]] + ct["band"] + [t for t, _ in ct["kk"]])))
    b2 = [a.get_tightbbox(r).transformed(inv) for a in (axT, axM)]
    b3 = ax3.get_tightbbox(r).transformed(inv)
    l2, r2 = min(b.x0 for b in b2), axM.get_position().x1 + 0.005  # legend sits far below the arrows
    gaps = (l2 - b1, b3.x0 - r2)
    mid = ((b1 + l2) / 2, (r2 + b3.x0) / 2)
    length = min(gaps) - 0.010  # same length for both, clear of everything on either side
    print(f"[GA] panel gaps {gaps[0]:.3f} / {gaps[1]:.3f} (fig frac), arrow length {length:.3f}")
    for m in mid:
        _arrow(fig, m, y_mid)

    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "Graphical_abstract"
    png, pdf = out_dir / f"{tag}.png", out_dir / f"{tag}.pdf"
    fig.savefig(png, dpi=DPI, facecolor=fig.get_facecolor())
    fig.savefig(pdf, dpi=DPI, facecolor=fig.get_facecolor())
    print(f"[GA] wrote {png}  ({int(FIGSIZE[0] * DPI)} x {int(FIGSIZE[1] * DPI)} px)")
    print(f"[GA] wrote {pdf}")
    if check:
        chk = out_dir / f"{tag}_check50.png"
        fig.savefig(chk, dpi=DPI / 2, facecolor=fig.get_facecolor())
        print(f"[GA] wrote {chk}  (50 % scale - every label must stay readable)")
    return fig


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--data-dir", type=Path, default=DATA_DIR)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--check", action="store_true", help="also write a 50 %% scale PNG")
    a = ap.parse_args()
    graphical_abstract(a.data_dir, a.out_dir, check=a.check)
