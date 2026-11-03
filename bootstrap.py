"""Two-level Bayesian bootstrap of the fits to the mean trajectory.

Every draw gives Dirichlet(1) weights to the 15 s windows of each recording (weighted median) and to the five
buckets (weighted mean trajectory), standardises every bucket as in the main analysis and refits the model with
the observed mean temperature and cooling indicator. Equal weights reproduce the plain median and mean.

Outputs: coefficient intervals of L for all configurations, the difference between the cooling and heating branch
of L, and the uncertainty bands of the example fits shown in Figs. 1 and 6.
"""
import numpy as np
import pandas as pd

import models as M
import settings as S

N_DRAWS = 4000
CHUNK = 250
COEFFICIENT_SEED = 20261016
FIT_SEED = 20261015
FIT_EXAMPLES = [("RSR", 128), ("RSR", 1024), ("RSR", 4096)]      # Fig. 1
NP_EXAMPLE = ("PE", 4096)                           # Fig. 6
LEVELS = (50, 80, 95)
GRID_POINTS = 80
GRID_MARGIN = 0.04            # branch lines extend this share of the temperature range beyond their end points
DELTA_GRID_POINTS = 41
OUT = S.BOOTSTRAP_DIR


class Bootstrap:
    """Window values, observed mean trajectory and the design of L for the fits to the mean trajectory.
    folder: recording and window values (default results/measures; the loudness control passes its own)."""

    def __init__(self, folder=None):
        folder = folder or S.MEASURES_DIR
        self.values = pd.read_csv(folder / "measures_by_recording.csv.gz")
        self.windows = pd.read_csv(folder / "measures_by_window.csv.gz")
        self.Y, self.cfg, self.valid = M.load_values(self.values)
        self.ybar = M.mean_trajectory(self.Y, self.valid)
        self.ok = ~np.isnan(self.ybar).any(0)
        self.cool = M.cooling(M.T_MEAN)
        self.X = M.design("L", M.T_MEAN, self.cool)[self.ok]
        self.P = np.linalg.pinv(self.X)
        margin = GRID_MARGIN * (M.T_MEAN.max() - M.T_MEAN.min())
        self.grid = {}
        for phase, sel in (("heating", self.ok & (self.cool == 0)), ("cooling", self.ok & (self.cool == 1))):
            self.grid[phase] = np.linspace(M.T_MEAN[sel].min() - margin, M.T_MEAN[sel].max() + margin, GRID_POINTS)

    def window_array(self, measure, fmin):
        """Window values as buckets x occasions x windows."""
        w = self.windows[(self.windows.measure == measure) & (self.windows.fmin_hz == fmin)]
        V = np.full((len(S.BUCKETS), S.N_OCCASIONS, 40), np.nan)
        for (b, o), g in w.groupby(["bucket", "occasion"]):
            if o <= S.N_OCCASIONS:
                V[S.BUCKETS.index(b), o - 1, :len(g)] = g.sort_values("window").value.to_numpy()
        return V

    def fit_L(self, traj):
        """traj (draws x occasions) -> coefficients, fitted values, R2."""
        y = traj[:, self.ok]
        B = (self.P @ y.T).T
        f = B @ self.X.T
        r2 = 1 - ((y - f) ** 2).sum(1) / ((y - y.mean(1, keepdims=True)) ** 2).sum(1)
        return B, f, r2

    def branch_lines(self, B):
        return {"heating": B[:, [0]] + B[:, [2]] * (self.grid["heating"] - S.T_REF_C),
                "cooling": B[:, [0]] + B[:, [1]] + (B[:, [2]] + B[:, [3]]) * (self.grid["cooling"] - S.T_REF_C)}

    def draws(self, measure, fmin, n, rng):
        """Weighted mean trajectories (draws x occasions) of one configuration."""
        V = self.window_array(measure, fmin)
        recorded = ~np.isnan(V).all(-1)
        mask = ~np.isnan(V)
        with np.errstate(all="ignore"):
            row_max = np.where(mask.any(-1, keepdims=True), np.nanmax(np.where(mask, V, -np.inf), -1, keepdims=True),
                               0.0)
        order = np.argsort(np.where(mask, V, row_max), axis=-1)
        V_sorted = np.take_along_axis(np.where(mask, V, row_max), order, -1)
        n_win = V.shape[-1]

        def weighted_median(w):
            """Weighted median over the windows (weights sum to one; equal weights give the median)."""
            ws = np.take_along_axis(w, order[None], -1)
            c = np.cumsum(ws, -1) - ws / 2
            i = (c < 0.5).sum(-1)
            lo, hi = np.clip(i - 1, 0, n_win - 1), np.clip(i, 0, n_win - 1)
            take = lambda a, j: np.take_along_axis(a, j[..., None], -1)[..., 0]
            c_lo, c_hi = take(c, lo), take(c, hi)
            v = np.broadcast_to(V_sorted, c.shape)
            v_lo, v_hi = take(v, lo), take(v, hi)
            frac = np.where(c_hi > c_lo, (0.5 - c_lo) / np.where(c_hi > c_lo, c_hi - c_lo, 1), 0.0)
            return np.where(recorded[None], v_lo + np.clip(frac, 0, 1) * (v_hi - v_lo), np.nan)

        def standardise(x):
            return (x - np.nanmean(x, -1, keepdims=True)) / np.nanstd(x, -1, ddof=1, keepdims=True)

        vm = self.valid[None].astype(float)
        out = []
        for start in range(0, n, CHUNK):
            d = min(CHUNK, n - start)
            g = rng.gamma(1.0, size=(d, len(S.BUCKETS), S.N_OCCASIONS, n_win)) * mask[None]
            window_weights = g / np.maximum(g.sum(-1, keepdims=True), 1e-300)
            bucket_weights = rng.dirichlet(np.ones(len(S.BUCKETS)), size=d)
            Z = standardise(weighted_median(window_weights))
            out.append((bucket_weights[..., None] * np.where(self.valid[None], Z, 0.0) * vm).sum(1)
                       / (bucket_weights[..., None] * vm).sum(1))
        return np.concatenate(out)


def band(a, level):
    h = (100 - level) / 2
    return np.percentile(a, [h, 100 - h], axis=0)


def fit_bands(bs, measure, fmin, rng, with_np=False, np_bandwidth=None, out=None):
    """Quantile bands of the data and of the refitted models over the draws, in time and temperature space.
    out: output folder (default results/bootstrap)."""
    out = out or OUT
    k = int(bs.cfg.index[(bs.cfg.measure == measure) & (bs.cfg.fmin_hz == fmin)][0])
    traj = bs.draws(measure, fmin, N_DRAWS, rng)
    B, fits, _ = bs.fit_L(traj)
    B0, f0, _ = bs.fit_L(bs.ybar[[k]])
    by_time = pd.DataFrame(dict(occasion=np.arange(1, S.N_OCCASIONS + 1)[bs.ok], time_days=M.TIME_DAYS[bs.ok],
                                T_mean=M.T_MEAN[bs.ok], observed_mean=bs.ybar[k, bs.ok], L=f0[0]))
    curves = {"data": traj[:, bs.ok], "L": fits}
    if with_np:
        curves["NP"] = M.smooth(np.where(bs.ok, traj, np.nan), np_bandwidth)[:, bs.ok]
        by_time["NP"] = M.smooth(np.where(bs.ok, bs.ybar[k], np.nan), np_bandwidth)[bs.ok]
    for name, arr in curves.items():
        for lv in LEVELS:
            by_time[f"{name}_lo{lv}"], by_time[f"{name}_hi{lv}"] = band(arr, lv)
    rows = []
    lines, lines0 = bs.branch_lines(B), bs.branch_lines(B0)
    for phase in ("heating", "cooling"):
        part = pd.DataFrame(dict(phase=phase, T=bs.grid[phase], L=lines0[phase][0]))
        for lv in LEVELS:
            part[f"L_lo{lv}"], part[f"L_hi{lv}"] = band(lines[phase], lv)
        rows.append(part)
    tag = f"{measure}_{fmin}"
    by_time.to_csv(out / f"fit_bands_{tag}.csv", index=False)
    pd.concat(rows).to_csv(out / f"branch_bands_{tag}.csv", index=False)


def coefficient_draws(bs):
    """Bayesian-bootstrap draws of the L coefficients for every configuration (configurations x draws x 4)."""
    rng = np.random.default_rng(COEFFICIENT_SEED)
    out = np.empty((len(bs.cfg), N_DRAWS, 4), np.float32)
    for k, (m, f) in enumerate(zip(bs.cfg.measure, bs.cfg.fmin_hz)):
        out[k] = bs.fit_L(bs.draws(m, f, N_DRAWS, rng))[0]
        print(f"  coefficients {k + 1}/{len(bs.cfg)}", flush=True)
    return out


def branch_difference(bs, draws):
    """Difference between the cooling and heating branch of L, b + e (T - 30), over the temperature range covered
    by both phases of the mean course, from the fit, the heating-occasion jackknife and the bootstrap draws."""
    T = M.T_MEAN
    heat, cool = bs.ok & (bs.cool == 0), bs.ok & (bs.cool == 1)
    lo, hi = max(T[heat].min(), T[cool].min()), min(T[heat].max(), T[cool].max())
    mid = (lo + hi) / 2
    grid = np.linspace(lo, hi, DELTA_GRID_POINTS)
    X = M.design("L", T, bs.cool)
    full = M.fit(X[bs.ok], bs.ybar[:, bs.ok].T)
    b, e = full[1], full[3]
    jk = []
    for j in np.where(heat)[0]:
        keep = bs.ok.copy()
        keep[j] = False
        jk.append(M.fit(X[keep], bs.ybar[:, keep].T))
    jk = np.array(jk)
    excludes = lambda q: (q[0] > 0) | (q[1] < 0)
    q_mid = np.quantile(draws[:, :, 1] + draws[:, :, 3] * (mid - S.T_REF_C), [0.025, 0.975], axis=1)
    q_grid = np.quantile(draws[:, :, 1][..., None] + draws[:, :, 3][..., None] * (grid - S.T_REF_C),
                         [0.025, 0.975], axis=1)
    t_cross = np.where(e != 0, S.T_REF_C - b / np.where(e != 0, e, 1), np.nan)
    per = bs.cfg.copy()
    per["difference_mid"] = b + e * (mid - S.T_REF_C)
    per["ci_low_mid"], per["ci_high_mid"] = q_mid
    per["ci_excludes_zero_mid"] = excludes(q_mid)
    per["share_range_ci_excludes_zero"] = excludes(q_grid).mean(1)
    per["branches_cross_in_range"] = (t_cross > lo) & (t_cross < hi)
    per["jackknife_sign_stable_mid"] = np.all(np.sign(jk[:, 1] + jk[:, 3] * (mid - S.T_REF_C))
                                              == np.sign(per.difference_mid.to_numpy()), 0)
    rows = []
    for g in ("nonlinear", "conventional"):
        s = per[per.family == g]
        rows.append(dict(group=g, n=len(s), T_low=lo, T_high=hi, T_mid=mid,
                         median_abs_difference_mid=s.difference_mid.abs().median(),
                         n_ci_excludes_zero_mid=int(s.ci_excludes_zero_mid.sum()),
                         n_ci_excludes_zero_somewhere=int((s.share_range_ci_excludes_zero > 0).sum()),
                         n_ci_excludes_zero_everywhere=int((s.share_range_ci_excludes_zero == 1).sum()),
                         median_share_range=s.share_range_ci_excludes_zero.median(),
                         n_branches_cross=int(s.branches_cross_in_range.sum()),
                         n_jackknife_sign_stable_mid=int(s.jackknife_sign_stable_mid.sum())))
    return per, pd.DataFrame(rows)


def run():
    OUT.mkdir(parents=True, exist_ok=True)
    bs = Bootstrap()

    rng = np.random.default_rng(FIT_SEED)
    for measure, fmin in FIT_EXAMPLES:
        fit_bands(bs, measure, fmin, rng)
    np_bandwidth = M.choose_bandwidth(bs.Y, bs.valid, list(range(len(S.BUCKETS))))
    fit_bands(bs, *NP_EXAMPLE, np.random.default_rng(FIT_SEED), with_np=True, np_bandwidth=np_bandwidth)

    draws = coefficient_draws(bs)
    np.save(OUT / "L_coefficient_draws.npy", draws)
    estimates = bs.fit_L(bs.ybar)[0]
    rows = []
    for k in range(len(bs.cfg)):
        for i, name in enumerate(["alpha", "b", "d", "e"]):
            v = draws[k, :, i].astype(float)
            lo, med, hi = np.percentile(v, [2.5, 50, 97.5])
            rows.append(dict(family=bs.cfg.family[k], measure=bs.cfg.measure[k], fmin_hz=int(bs.cfg.fmin_hz[k]),
                             coefficient=name, estimate=estimates[k, i], draw_median=med, lo95=lo, hi95=hi,
                             share_draws_positive=np.mean(v > 0), interval_excludes_zero=bool(lo > 0 or hi < 0)))
    coef = pd.DataFrame(rows)
    coef.to_csv(OUT / "L_coefficients_bayes_boot.csv", index=False)

    per, summary = branch_difference(bs, draws)
    per.to_csv(OUT / "branch_difference_per_config.csv", index=False)
    summary.to_csv(OUT / "branch_difference_summary.csv", index=False)

    lines = [f"BAYESIAN BOOTSTRAP ({N_DRAWS} draws): configurations whose 95% interval of the L coefficient excludes 0"]
    for name in ("b", "d", "e"):
        s = coef[coef.coefficient == name]
        lines.append(f"  {name}: " + ", ".join(
            f"{g} {int(s[s.family == g].interval_excludes_zero.sum())} of {int((s.family == g).sum())}"
            for g in ("nonlinear", "conventional")))
    r = summary.iloc[0]
    lines.append(f"difference between the branches over {r.T_low:.1f}-{r.T_high:.1f} C (midpoint {r.T_mid:.1f} C):")
    for r in summary.itertuples():
        lines.append(f"  {r.group}: median |difference| at the midpoint {r.median_abs_difference_mid:.2f} SD; interval "
                     f"excludes zero at the midpoint in {r.n_ci_excludes_zero_mid}/{r.n}, somewhere in the range in "
                     f"{r.n_ci_excludes_zero_somewhere}, everywhere in {r.n_ci_excludes_zero_everywhere} (median share "
                     f"{r.median_share_range:.2f}); branches cross in {r.n_branches_cross}; jackknife sign stable at "
                     f"the midpoint in {r.n_jackknife_sign_stable_mid}")
    for measure, fmin in FIT_EXAMPLES:
        x = per[(per.measure == measure) & (per.fmin_hz == fmin)].iloc[0]
        lines.append(f"  {measure} {fmin} Hz: {x.difference_mid:+.2f} SD (95% interval {x.ci_low_mid:+.2f} to "
                     f"{x.ci_high_mid:+.2f})")
    (OUT / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
