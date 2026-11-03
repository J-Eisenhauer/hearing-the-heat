"""Model analysis: fits to the mean trajectory, leave-one-bucket-out skill, the temperature-surrogate test,
NP benchmark, residual diagnostics, bucket agreement, redundancy, sensitivity scenarios, broadband conventional
indices, the further checks of the Supplementary Analyses and the band-energy check of the Limitations. Writes tables and report.txt to results/models."""
import time

import numpy as np
import pandas as pd
from scipy.stats import chi2

import models as M
import settings as S

PRIMARY_TESTS = [("L", "nonlinear"), ("L", "conventional")]     # plus the group difference of L
REDUNDANCY_RHO = 0.95
LJUNG_BOX_LAG = 5
NEAR_COPY_R = 0.9
GROUPS = ("nonlinear", "conventional", "all")
OUT = S.MODELS_DIR


def read_values(name):
    return pd.read_csv(S.MEASURES_DIR / name)


def scenario_values(main, scenarios, name):
    """Recording values of one sensitivity scenario: the main values with the affected cells replaced."""
    s = scenarios[scenarios.scenario == name].drop(columns="scenario")
    key = ["measure", "fmin_hz"]
    affected = s[key].drop_duplicates()
    rest = main.merge(affected, on=key, how="left", indicator=True)
    return pd.concat([rest[rest._merge == "left_only"].drop(columns="_merge"), s], ignore_index=True)


def with_config(cfg, rows):
    """Long table with the configuration columns in front."""
    return cfg.reset_index(names="config").merge(pd.DataFrame(rows), on="config")


def heldout(Y, valid, models):
    """Held-out skill per bucket (buckets x configurations), its mean and the pooled RMSE for every model."""
    out = {}
    for m in models:
        folds = M.ceiling_folds(Y, valid) if m == "training-mean" else M.cv_folds(Y, valid, m)
        per_bucket = M.skill(folds)
        out[m] = dict(folds=folds, per_bucket=per_bucket, skill=per_bucket.mean(0), rmse=M.rmse(folds))
    return out


# =============================================================================
# Temperature surrogates
# =============================================================================
def temperature_surrogates(sets):
    """Re-evaluate the temperature models with IAAFT surrogates of the mean temperature course.
    Each surrogate replaces the mean course; every bucket keeps its real deviation from the mean.
    `sets` maps a name to (Y, valid, models). Returns float32 skill arrays and per-surrogate properties."""
    rng = np.random.default_rng(S.TEMPERATURE_SURROGATE_SEED)
    n = S.N_TEMPERATURE_SURROGATES
    cv = {k: M.SurrogateCV(Y, valid, models) for k, (Y, valid, models) in sets.items()}
    null = {(k, m): np.empty((n, Y.shape[0]), np.float32) for k, (Y, _, models) in sets.items() for m in models}
    offsets = M.T_IN - M.T_MEAN
    folds = [[b for b in range(len(S.BUCKETS)) if b != h] for h in range(len(S.BUCKETS))]
    real_breaks = [M.cooling(M.T_IN[f].mean(0)) for f in folds]
    corr, peak, same_break = np.empty(n), np.empty(n, int), np.empty(n, bool)
    t0 = time.time()
    for i in range(n):
        s = M.iaaft(M.T_MEAN, rng)
        t_in = s + offsets
        for k, f in cv.items():
            for m, v in f(t_in).items():
                null[(k, m)][i] = v
        corr[i] = np.corrcoef(s, M.T_MEAN)[0, 1]
        peak[i] = int(np.argmax(s)) + 1
        same_break[i] = all(np.array_equal(M.cooling(t_in[f].mean(0)), r) for f, r in zip(folds, real_breaks))
        if (i + 1) % 10000 == 0:
            print(f"  temperature surrogates {i + 1}/{n} ({time.time() - t0:.0f} s)", flush=True)
    return null, pd.DataFrame(dict(draw=np.arange(n), corr_with_real=corr, peak_occasion=peak,
                                   same_break_all_folds=same_break))


# =============================================================================
# Diagnostics, agreement, redundancy
# =============================================================================
def residual_diagnostics(fits, cfg):
    rows = []
    for m, f in fits.items():
        r = f["observed"] - f["fitted"]
        n = len(r)
        lags = np.arange(1, LJUNG_BOX_LAG + 1)
        for k in range(len(cfg)):
            e = r[:, k] - r[:, k].mean()
            acf = np.array([(e[l:] * e[:-l]).sum() for l in lags]) / (e ** 2).sum()
            q = n * (n + 2) * np.sum(acf ** 2 / (n - lags))
            rows.append(dict(config=k, model=m, lag1_autocorrelation=acf[0], ljung_box_Q=q,
                             ljung_box_p=chi2.sf(q, LJUNG_BOX_LAG)))
    return with_config(cfg, rows)


def bucket_agreement(Y, valid, cfg):
    rows = []
    for k in range(len(cfg)):
        rho = pd.DataFrame(np.where(valid, Y[k], np.nan).T, columns=S.BUCKETS).corr(method="spearman").to_numpy()
        pairs = rho[np.triu_indices(len(S.BUCKETS), 1)]
        rows.append(dict(median_pairwise_spearman=np.median(pairs), min_pairwise_spearman=pairs.min()))
    return cfg.join(pd.DataFrame(rows))


def correlation_matrix(Y, valid, cfg):
    """Spearman correlations between all configurations over the standardised bucket-occasion values."""
    stacked = np.where(valid[None], Y, np.nan).reshape(len(cfg), -1)
    labels = (cfg.measure + " | " + cfg.band).tolist()
    return pd.DataFrame(stacked.T, columns=labels).corr(method="spearman")


def redundancy(rho, cfg):
    rows = []
    R = rho.to_numpy()
    for a in range(len(cfg)):
        for b in range(a + 1, len(cfg)):
            rows.append(dict(measure_a=cfg.measure[a], band_a=cfg.band[a], measure_b=cfg.measure[b],
                             band_b=cfg.band[b], same_band=cfg.fmin_hz[a] == cfg.fmin_hz[b], spearman=R[a, b],
                             redundant=abs(R[a, b]) >= REDUNDANCY_RHO))
    pairs = pd.DataFrame(rows)
    within = pairs[pairs.same_band]
    summary = (within.groupby(["measure_a", "measure_b"], sort=False)
               .agg(min_spearman=("spearman", "min"), median_spearman=("spearman", "median"),
                    max_spearman=("spearman", "max"), bands_redundant=("redundant", "sum")).reset_index())
    return pairs, summary


def flat_spectrum_check(values):
    """Spectral entropy H and flatness F of nearly flat spectra: H ~ 1 - v / (2 ln N), F ~ exp(-v / 2)."""
    rows = []
    for fmin, fmax in S.BANDS:
        fs = min(S.SAMPLE_RATE_HZ, round(10 * fmax))
        f = np.fft.rfftfreq(512, 1 / fs)
        n = int(((f >= fmin) & (f <= fmax)).sum())
        h = values[(values.measure == "SpectralEntropy") & (values.fmin_hz == fmin)].value.median()
        fl = values[(values.measure == "SpectralFlatness") & (values.fmin_hz == fmin)].value.median()
        rows.append(dict(fmin_hz=int(fmin), n_bins=n, median_entropy=h, median_flatness=fl,
                         v_from_entropy=2 * np.log(n) * (1 - h), v_from_flatness=-2 * np.log(fl)))
    return pd.DataFrame(rows)


# =============================================================================
# Further checks (Supplementary Analyses)
# =============================================================================
def l_time_comparison(ho, fits, groups):
    """L against L_time (elapsed time instead of temperature within the two phases), and L plus a linear time term."""
    rows = []
    for g in GROUPS:
        gi = groups[g]
        per_bucket = np.median(ho["L"]["per_bucket"][:, gi] - ho["L_time"]["per_bucket"][:, gi], 1)
        d = ho["L"]["skill"][gi] - ho["L_time"]["skill"][gi]
        rows.append(dict(group=g, n=len(gi), skill_L_time=np.median(ho["L_time"]["skill"][gi]),
                         skill_L=np.median(ho["L"]["skill"][gi]), median_difference=np.median(d),
                         n_L_better=int((d > 0).sum()), R2_L_time=np.median(fits["L_time"]["r2"][gi]),
                         R2_L=np.median(fits["L"]["r2"][gi]),
                         **{f"median_difference_heldout_{b}": v for b, v in zip(S.BUCKETS, per_bucket)},
                         skill_L_plus_time=np.median(ho["L+time"]["skill"][gi])))
    return pd.DataFrame(rows)


def conditional_surrogate_test(ho, null, surr, groups):
    """Surrogate test of L_time, and of L restricted to surrogates that keep the real cooling break in every fold."""
    rows = []
    same = surr.same_break_all_folds.to_numpy()
    for g in ("nonlinear", "conventional"):
        gi = groups[g]
        real_lt, real_l = np.median(ho["L_time"]["skill"][gi]), np.median(ho["L"]["skill"][gi])
        lt, l = np.median(null["L_time"][:, gi], 1), np.median(null["L"][:, gi], 1)
        rows.append(dict(group=g, real_L_time=real_lt, null_median_L_time=np.median(lt),
                         null_q95_L_time=np.quantile(lt, 0.95), p_L_time=M.rank_p(lt, real_lt),
                         n_same_break=int(same.sum()), real_L=real_l,
                         conditional_null_median_L=np.median(l[same]),
                         conditional_null_q95_L=np.quantile(l[same], 0.95),
                         conditional_p_L=M.rank_p(l[same], real_l),
                         median_corr_with_real=np.median(surr.corr_with_real[same])))
    return pd.DataFrame(rows)


def mean_temperature_check(Y, valid, ho, groups, per):
    """Held-out skill with each bucket's own temperature against the training-mean temperature."""
    rows = []
    for m in ("T-only", "L"):
        mt = M.skill(M.cv_folds(Y, valid, m, mean_temperature=True)).mean(0)
        per[f"skill_{m}_mean_temperature"] = mt
        for g in GROUPS:
            gi = groups[g]
            d = ho[m]["skill"][gi] - mt[gi]
            rows.append(dict(model=m, group=g, skill_own=np.median(ho[m]["skill"][gi]),
                             skill_mean_temperature=np.median(mt[gi]), median_difference=np.median(d),
                             max_abs_difference=np.abs(d).max()))
    return pd.DataFrame(rows)


def skill_by_phase(ho, groups, per):
    """Held-out skill within the heating and cooling phase, and the share of SSE0 and of the explained part
    that falls on heating occasions. Returns the table and the number of values per phase."""
    phases = ("heating", "cooling")
    sse0 = {p: 0.0 for p in phases}
    explained = {p: 0.0 for p in phases}
    n_points = {p: 0 for p in phases}
    for f in ho["L"]["folds"]:
        for p in phases:
            keep = f["heating"] if p == "heating" else ~f["heating"]
            y, q = f["observed"][keep], f["predicted"][keep]
            a, e = (y ** 2).sum(0), ((y - q) ** 2).sum(0)
            sse0[p] = sse0[p] + a
            explained[p] = explained[p] + (a - e)
            n_points[p] += int(keep.sum())
    share_sse0 = sse0["heating"] / (sse0["heating"] + sse0["cooling"])
    share_explained = explained["heating"] / (explained["heating"] + explained["cooling"])
    for m in S.MODELS:
        for p in phases:
            per[f"skill_{p}_{m}"] = M.skill(ho[m]["folds"], p).mean(0)
    rows = []
    for g in GROUPS:
        gi = groups[g]
        row = dict(group=g, n_heating=n_points["heating"], n_cooling=n_points["cooling"],
                   share_SSE0_heating=np.median(share_sse0[gi]),
                   share_explained_heating=np.median(share_explained[gi]))
        for m in S.MODELS:
            row.update({f"all_{m}": np.median(ho[m]["skill"][gi]),
                        f"heating_{m}": np.median(per[f"skill_heating_{m}"][gi]),
                        f"cooling_{m}": np.median(per[f"skill_cooling_{m}"][gi])})
        rows.append(row)
    return pd.DataFrame(rows), n_points


def heating_jackknife(Y, valid, cfg, fits):
    """Refit L to the mean trajectory without one heating occasion at a time; is the sign of b, d, e stable?"""
    full = fits["L"]["coef"]
    draws = []
    for j in np.where(M.cooling(M.T_MEAN) == 0)[0]:
        keep = np.ones(S.N_OCCASIONS, bool)
        keep[j] = False
        draws.append(M.mean_fit(Y, valid, "L", keep=keep)["coef"])
    draws = np.array(draws)
    jk = cfg.copy()
    for i, name in ((1, "b"), (2, "d"), (3, "e")):
        jk[f"{name}_sign_stable"] = np.all(np.sign(draws[:, i]) == np.sign(full[i]), 0)
    return jk


def mains_hum():
    """Share of hum windows per recording as a bucket x occasion array."""
    table = read_values("mains_hum_by_recording.csv")
    hum = np.full((len(S.BUCKETS), S.N_OCCASIONS), np.nan)
    for b, o, v in zip(table.bucket, table.occasion, table.hum):
        hum[S.BUCKETS.index(b), o - 1] = v
    return hum


def covariate_check(Y, valid, cfg, ho, fits, groups, per, hum):
    """Afternoon/evening session and mains hum as an additional term of L."""
    label = (cfg.measure + " " + cfg.band).to_numpy()
    covariates = {"afternoon/evening": (np.tile(M.DAY, (len(S.BUCKETS), 1)), M.DAY),
                  "mains hum": (hum, np.nanmean(np.where(valid, hum, np.nan), 0))}
    rows = []
    for name, (per_bucket, mean_course) in covariates.items():
        skill = M.skill(M.cv_folds(Y, valid, "L", covariate=per_bucket)).mean(0)
        f = M.mean_fit(Y, valid, "L", covariate=mean_course)
        d = skill - ho["L"]["skill"]
        per[f"skill_L_plus_{name}"] = skill
        for g in GROUPS:
            gi = groups[g]
            rows.append(dict(covariate=name, group=g,
                             diff_median_skill=np.median(skill[gi]) - np.median(ho["L"]["skill"][gi]),
                             diff_median_R2=np.median(f["r2"][gi]) - np.median(fits["L"]["r2"][gi]),
                             n_gain=int((d[gi] > 0).sum()), n_loss=int((d[gi] < 0).sum()),
                             max_gain=d[gi].max(), max_loss=d[gi].min(),
                             config_max_gain=label[gi][np.argmax(d[gi])],
                             config_max_loss=label[gi][np.argmin(d[gi])]))
    return pd.DataFrame(rows)


def recording_order():
    """Order of the buckets within each session and the spread of the recorder start times."""
    rec = pd.read_csv(S.RECORDINGS_FILE, parse_dates=["recorder_start"]).dropna(subset=["recorder_start"])
    sessions = []
    for o, g in rec.sort_values("recorder_start").groupby("occasion"):
        span = (g.recorder_start.max() - g.recorder_start.min()).total_seconds() / 60
        sessions.append(dict(occasion=o, order="".join(g.bucket), first_to_last_min=span,
                             median_gap_min=g.recorder_start.diff().dt.total_seconds().median() / 60))
    return pd.DataFrame(sessions)


def between_bucket_check(Y, valid, cfg, fits, groups):
    """Temperature differences between buckets within occasions against the prediction of L.

    Within an occasion, elapsed time is the same for all buckets. Temperature and standardised values are centred on
    their occasion means; L predicts a centred value of (d + e * C_i) times the centred temperature, with d and e from
    the fit to the mean trajectory. gamma = sum(x * y) / sum(x^2) over all points and configurations of a group
    (x predicted, y observed). Null: the temperature deviations permuted between the buckets of each occasion, jointly
    for all configurations."""
    coef = fits["L"]["coef"]                                            # (alpha, b, d, e) x configurations
    cool = M.cooling(M.T_MEAN)
    t_in = np.where(valid, M.T_IN, np.nan)
    dev_t = (t_in - np.nanmean(t_in, 0))[valid]                          # valid points, bucket-major
    occ = np.where(valid)[1]
    ym = np.where(valid[None], Y, np.nan)
    dev_y = (ym - np.nanmean(ym, 1, keepdims=True))[:, valid]           # configurations x points
    slope = coef[2][:, None] + coef[3][:, None] * cool[occ][None]        # configurations x points
    rng = np.random.default_rng(S.BETWEEN_BUCKET_SEED)
    members = [np.where(occ == o)[0] for o in np.unique(occ)]
    rows = []
    stats = {g: dict(u=(slope[groups[g]] * dev_y[groups[g]]).sum(0), w=(slope[groups[g]] ** 2).sum(0)) for g in GROUPS}
    null = {g: [] for g in GROUPS}
    done, chunk = 0, 5000
    while done < S.N_BETWEEN_BUCKET_PERMUTATIONS:
        n = min(chunk, S.N_BETWEEN_BUCKET_PERMUTATIONS - done)
        perm = np.empty((n, len(dev_t)))
        for m in members:
            order = np.argsort(rng.random((n, len(m))), 1)
            perm[:, m] = dev_t[m][order]
        for g in GROUPS:
            null[g].append((perm @ stats[g]["u"]) / ((perm ** 2) @ stats[g]["w"]))
        done += n
    for g in GROUPS:
        gi = groups[g]
        gamma = (dev_t @ stats[g]["u"]) / ((dev_t ** 2) @ stats[g]["w"])
        nl = np.concatenate(null[g])
        match = ((slope[gi] * dev_t[None] * dev_y[gi]).sum(1) > 0).mean()
        rows.append(dict(group=g, n_configs=len(gi), gamma=gamma, null_median=np.median(nl),
                         null_q95=np.quantile(nl, 0.95), p=(1 + np.sum(nl >= gamma)) / (1 + len(nl)),
                         share_configs_matching_sign=match, n_points=len(dev_t),
                         median_abs_deviation_C=np.median(np.abs(dev_t)), sd_deviation_C=dev_t.std(),
                         max_abs_deviation_C=np.abs(dev_t).max()))
    return pd.DataFrame(rows)


CROSS_LAMBDAS = 10 ** np.linspace(-2, 4, 31)     # ridge penalties (times the number of training rows)


def _ridge_path(X, Y_):
    """Ridge coefficients for all penalties; predictors standardised and targets centred with training statistics."""
    m, s = X.mean(0), X.std(0)
    s[s < 1e-12] = 1.0
    ym = Y_.mean(0)
    U, sv, Vt = np.linalg.svd((X - m) / s, full_matrices=False)
    UY = U.T @ (Y_ - ym)
    B = np.array([Vt.T @ ((sv / (sv ** 2 + lam * len(X)))[:, None] * UY) for lam in CROSS_LAMBDAS])
    return lambda Xn, k: ym + ((Xn - m) / s) @ B[k]


def _ridge_choice(D, buckets, xi, yi):
    """Penalty per target by leave-one-bucket-out within `buckets` (smallest summed squared error)."""
    sse = np.zeros((len(CROSS_LAMBDAS), len(yi)))
    for g in buckets:
        tr = [b for b in buckets if b != g]
        f = _ridge_path(np.vstack([D[b][:, xi] for b in tr]), np.vstack([D[b][:, yi] for b in tr]))
        for k in range(len(CROSS_LAMBDAS)):
            sse[k] += ((D[g][:, yi] - f(D[g][:, xi], k)) ** 2).sum(0)
    return sse.argmin(0)


def cross_prediction(Y, valid, cfg, groups):
    """Redundancy between the groups: ridge regression predicts every configuration of one group from all
    configurations of the other group in the held-out bucket (leave-one-bucket-out). The penalty is chosen per target
    by leave-one-bucket-out within the four training buckets only. Variant: predictors from the target's own band
    removed. Returns the skill per target and the report lines."""
    nb = len(S.BUCKETS)
    D = {b: Y[:, b, valid[b]].T for b in range(nb)}
    rows = []
    for src, dst in (("conventional", "nonlinear"), ("nonlinear", "conventional")):
        for own in (True, False):
            for f in sorted(cfg.fmin_hz.unique()):
                yi = groups[dst][cfg.fmin_hz.values[groups[dst]] == f]
                xi = groups[src] if own else groups[src][cfg.fmin_hz.values[groups[src]] != f]
                skills = []
                for h in range(nb):
                    tr = [b for b in range(nb) if b != h]
                    ks = _ridge_choice(D, tr, xi, yi)
                    fit_ = _ridge_path(np.vstack([D[b][:, xi] for b in tr]), np.vstack([D[b][:, yi] for b in tr]))
                    pred = np.column_stack([fit_(D[h][:, xi], k)[:, j] for j, k in enumerate(ks)])
                    skills.append(1 - ((D[h][:, yi] - pred) ** 2).sum(0) / (D[h][:, yi] ** 2).sum(0))
                for j, c in enumerate(yi):
                    rows.append(dict(predictors=src, target_group=dst, own_band="included" if own else "excluded",
                                     n_predictors=len(xi), measure=cfg.measure[c], fmin_hz=cfg.fmin_hz[c],
                                     skill=np.mean(skills, 0)[j]))
    t = pd.DataFrame(rows)
    lines = ["", "CROSS-PREDICTION BETWEEN THE GROUPS (Redundancy; ridge, held-out buckets)"]
    for (src, own), g in t.groupby(["predictors", "own_band"], sort=False):
        lines.append(f"{src} -> {g.target_group.iloc[0]} ({g.n_predictors.iloc[0]} predictors, own band {own}): "
                     f"median skill {g.skill.median():.3f} (range {g.skill.min():.3f} to {g.skill.max():.3f}); "
                     "per target measure: " + ", ".join(f"{m} {v:.2f}" for m, v in g.groupby("measure", sort=False).skill.median().items()))
    return t, lines


TOP_BAND_HZ = 8192          # 8192-16384 Hz: band energy varied by only ~1 dB (Limitations); results recomputed without it


def top_band_check(cfg, ho, null, groups):
    """Main comparisons with and without the highest octave band: median held-out skill of every model, counts of
    configurations in which L beats the other models, and the temperature-surrogate p of L from the same surrogates."""
    rows = []
    for g in ("nonlinear", "conventional"):
        for label, idx in (("all bands", groups[g]), ("without 8192-16384 Hz", groups[g][cfg.fmin_hz.values[groups[g]] != TOP_BAND_HZ])):
            r = dict(group=g, bands=label, n_configs=len(idx))
            for m in ("time-only", "T-only", "L_time", "L", "NP"):
                r[f"skill_{m}"] = np.median(ho[m]["skill"][idx])
            for m in ("time-only", "T-only", "L_time"):
                r[f"n_L_better_than_{m}"] = int((ho["L"]["skill"][idx] > ho[m]["skill"][idx]).sum())
            nl = np.median(null["L"][:, idx], axis=1)
            r["surrogate_median_L"] = np.median(nl)
            r["surrogate_p_L"] = (1 + np.sum(nl >= r["skill_L"])) / (1 + len(nl))
            rows.append(r)
    t = pd.DataFrame(rows)
    lines = ["", "WITHOUT THE HIGHEST BAND (8192-16384 Hz)"]
    for _, x in t.iterrows():
        lines.append(f"{x['group']}, {x['bands']} ({x['n_configs']}): skill time-only {x['skill_time-only']:.3f}, "
                     f"T-only {x['skill_T-only']:.3f}, L_time {x['skill_L_time']:.3f}, L {x['skill_L']:.3f}, NP {x['skill_NP']:.3f}; "
                     f"L better than time-only/T-only/L_time in {x['n_L_better_than_time-only']}/{x['n_L_better_than_T-only']}/"
                     f"{x['n_L_better_than_L_time']}; surrogate median {x['surrogate_median_L']:.3f}, p = {x['surrogate_p_L']:.5f}")
    return t, lines


def further_checks(Y, cfg, valid, ho, fits, null, surr):
    """All further checks; returns the tables and the report lines."""
    groups = M.groups_of(cfg)
    per = cfg.copy()
    hum = mains_hum()
    t21 = l_time_comparison(ho, fits, groups)
    t22 = conditional_surrogate_test(ho, null, surr, groups)
    t23 = mean_temperature_check(Y, valid, ho, groups, per)
    t24, n_points = skill_by_phase(ho, groups, per)
    t25 = heating_jackknife(Y, valid, cfg, fits)
    t26 = covariate_check(Y, valid, cfg, ho, fits, groups, per, hum)
    t27 = recording_order()
    t29 = between_bucket_check(Y, valid, cfg, fits, groups)
    out = {"20_checks_per_config": per, "21_L_vs_L_time": t21, "22_conditional_surrogate_test": t22,
           "23_own_vs_mean_temperature": t23, "24_skill_by_phase": t24, "25_heating_jackknife": t25,
           "26_covariates": t26, "27_recording_order": t27,
           "29_between_bucket_temperature": t29}

    lines = ["", "FURTHER CHECKS"]
    r, c = t21.set_index("group"), t22.set_index("group")
    for g in ("nonlinear", "conventional"):
        gains = ", ".join(f"{r[f'median_difference_heldout_{b}'][g]:+.3f}" for b in S.BUCKETS)
        lines.append(f"{g}: L_time {r.skill_L_time[g]:.3f} (surrogate median {c.null_median_L_time[g]:.3f}, "
                     f"p = {c.p_L_time[g]:.3f}); L better than L_time in {r.n_L_better[g]}/{r.n[g]}, "
                     f"median gain per held-out bucket {gains}")
        lines.append(f"  conditional test ({c.n_same_break[g]} surrogates with the real break, median r "
                     f"{c.median_corr_with_real[g]:.2f}): surrogate median {c.conditional_null_median_L[g]:.3f} "
                     f"(95th percentile {c.conditional_null_q95_L[g]:.3f}), observed {c.real_L[g]:.3f}, "
                     f"p = {c.conditional_p_L[g]:.3f}; L + linear time {r.skill_L_plus_time[g]:.3f}")
    for g in ("nonlinear", "conventional"):
        x = t23[(t23.model == "L") & (t23.group == g)].iloc[0]
        lines.append(f"{g}: L with the training-mean temperature {x.skill_mean_temperature:.3f} against "
                     f"{x.skill_own:.3f} (median paired difference {x.median_difference:.4f}, largest "
                     f"{x.max_abs_difference:.3f})")
    deviation = np.abs(M.T_IN - M.T_MEAN)
    lines.append(f"bucket deviation from the mean temperature: median {np.median(deviation):.2f} C, largest "
                 f"{deviation.max():.2f} C; mean course {M.T_MEAN.min():.1f}-{M.T_MEAN.max():.1f} C")
    p = t24.set_index("group")
    for g in ("nonlinear", "conventional"):
        lines.append(f"{g} skill of L by phase: heating {p.heating_L[g]:.2f}, cooling {p.cooling_L[g]:.2f}; "
                     f"heating occasions carry {p.share_SSE0_heating[g]:.0%} of SSE0 and "
                     f"{p.share_explained_heating[g]:.0%} of the explained part "
                     f"({n_points['heating']} heating and {n_points['cooling']} cooling values)")
    for g in ("nonlinear", "conventional"):
        s = t25[t25.family == g]
        lines.append(f"{g} heating jackknife, sign unchanged: b {s.b_sign_stable.mean():.0%}, "
                     f"d {s.d_sign_stable.mean():.0%}, e {s.e_sign_stable.mean():.0%}")
    for x in t26.itertuples():
        if x.group != "all":
            lines.append(f"L + {x.covariate}, {x.group}: skill {x.diff_median_skill:+.3f}, "
                         f"R2 {x.diff_median_R2:+.3f}, gain/loss {x.n_gain}/{x.n_loss}, "
                         f"max/min {x.max_gain:+.3f}/{x.max_loss:+.3f}")
    lines.append(f"mains hum present in {np.nanmean(np.where(valid, hum, np.nan)):.0%} of the analysed recordings")
    for x in t29.itertuples():
        lines.append(f"between buckets within occasions, {x.group}: gamma {x.gamma:.2f} (permutation median "
                     f"{x.null_median:.2f}, 95th percentile {x.null_q95:.2f}, p = {x.p:.4f}); sign as predicted in "
                     f"{x.share_configs_matching_sign:.0%} of {x.n_configs} configurations")
    x = t29.iloc[0]
    lines.append(f"temperature deviations from the occasion mean: median |dev| {x.median_abs_deviation_C:.2f} C, "
                 f"sd {x.sd_deviation_C:.2f} C, largest {x.max_abs_deviation_C:.2f} C ({x.n_points} values)")
    in_order = sum(o == "".join(sorted(o)) for o in t27.order)
    lines.append(f"recording order A-E in {in_order}/{len(t27)} sessions; "
                 f"first to last start {t27.first_to_last_min.median():.0f} min (median)")
    return out, lines


ENERGY_RHO_LOW = 0.85      # descriptive threshold below the redundancy threshold (Limitations)
N_END = 3                  # occasions averaged at the start and the end of the cycle


def band_energy_check(values, pairs):
    """Log band energy as a possible carrier of changes in sound transmission (Limitations).

    Per band: the bucket-mean log band energy (dB), its range over the cycle and the difference between the mean
    of the last and the first N_END occasions, plus the Spearman correlation of each nonlinear measure with log
    band energy in the same band (taken from the redundancy pairs)."""
    e = values[(values.measure == "LogBandEnergy") & (values.occasion <= S.N_OCCASIONS)]
    mean = e.groupby(["fmin_hz", "occasion"]).value.mean().unstack("occasion")
    first = mean.loc[:, sorted(mean.columns)[:N_END]].mean(1)
    last = mean.loc[:, sorted(mean.columns)[-N_END:]].mean(1)
    per_bucket = e.groupby(["fmin_hz", "bucket", "occasion"]).value.mean().unstack("occasion")
    cols = sorted(per_bucket.columns)
    change_b = (per_bucket[cols[-N_END:]].mean(1) - per_bucket[cols[:N_END]].mean(1)).unstack("bucket")
    table = pd.DataFrame({"fmin_hz": mean.index.astype(int), "range_db": (mean.max(1) - mean.min(1)).values,
                          "last_minus_first_db": (last - first).values,
                          "min_bucket_change_db": change_b.min(1).values,
                          "max_bucket_change_db": change_b.max(1).values})
    nl = [m for m in S.MEASURES if S.FAMILY[m] == "nonlinear"]
    rho = pairs[pairs.same_band & ((pairs.measure_a == "LogBandEnergy") | (pairs.measure_b == "LogBandEnergy"))]
    rho = rho.assign(measure=np.where(rho.measure_a == "LogBandEnergy", rho.measure_b, rho.measure_a))
    rho = rho[rho.measure.isin(nl)]
    for m in nl:
        r = rho[rho.measure == m].set_index("band_a").spearman
        table[f"rho_{m}"] = [r.get(f"{f}-{2 * f} Hz", np.nan) for f in table.fmin_hz]

    a = rho.spearman.abs()
    hi = rho[a >= REDUNDANCY_RHO]
    lines = ["", "BAND ENERGY (Limitations)",
             f"nonlinear measure-band pairs with |rho| to log band energy >= {REDUNDANCY_RHO}: "
             f"{int((a >= REDUNDANCY_RHO).sum())}/{len(rho)} ("
             + ", ".join(f"{x.measure} {x.band_a}" for x in hi.itertuples()) + f"); >= {ENERGY_RHO_LOW}: "
             f"{int((a >= ENERGY_RHO_LOW).sum())}/{len(rho)}",
             f"log band energy, bucket mean, last {N_END} minus first {N_END} occasions (dB): "
             + ", ".join(f"{x.fmin_hz} {x.last_minus_first_db:+.1f}" for x in table.itertuples()),
             "range over the cycle (dB): " + ", ".join(f"{x.fmin_hz} {x.range_db:.1f}" for x in table.itertuples()),
             "per bucket, last minus first (dB, min-max): "
             + ", ".join(f"{x.fmin_hz} {x.min_bucket_change_db:+.1f}/{x.max_bucket_change_db:+.1f}"
                         for x in table.itertuples())]
    return table, lines


# =============================================================================
# Summaries
# =============================================================================
def np_mean_fit(Y, valid, fits):
    """NP smoother fitted to the mean trajectory, with the bandwidth chosen on all buckets."""
    ok = fits["L"]["ok"]
    bandwidth = M.choose_bandwidth(Y, valid, list(range(len(S.BUCKETS))))
    W = M.np_smoother(ok, bandwidth)
    y = fits["L"]["observed"]
    fitted = (y.T @ W.T).T
    sse = ((y - fitted) ** 2).sum(0)
    fit = dict(fitted=fitted, observed=y, ok=ok, r2=1 - sse / ((y - y.mean(0)) ** 2).sum(0),
               rmse=np.sqrt(sse / ok.sum()), coef=None)
    return fit, bandwidth, float(np.trace(W))


def write_fit_tables(cfg, ho, fits, edf):
    K = len(cfg)
    rows = [dict(config=k, model=m, R2=fits[m]["r2"][k], RMSE=fits[m]["rmse"][k],
                 parameters=edf if m == "NP" else len(M.COEFFICIENTS[m]))
            for m in S.MODELS + ["NP"] for k in range(K)]
    with_config(cfg, rows).to_csv(OUT / "03_meanfit_metrics.csv", index=False)
    rows = [dict(config=k, model=m, coefficient=c, estimate=fits[m]["coef"][i, k])
            for m in S.MODELS for k in range(K) for i, c in enumerate(M.COEFFICIENTS[m])]
    with_config(cfg, rows).to_csv(OUT / "04_meanfit_coefficients.csv", index=False)
    rows = [dict(config=k, model=m, heldout_bucket=b, skill=ho[m]["per_bucket"][i, k])
            for m in S.MODELS + ["NP"] for k in range(K) for i, b in enumerate(S.BUCKETS)]
    with_config(cfg, rows).to_csv(OUT / "05_heldout_skill_per_bucket.csv", index=False)
    ok = fits["L"]["ok"]
    occasions = np.arange(1, S.N_OCCASIONS + 1)[ok]
    parts = [pd.DataFrame(dict(config=k, model=m, occasion=occasions, time_days=M.TIME_DAYS[ok],
                               T_mean=M.T_MEAN[ok], observed_mean=fits[m]["observed"][:, k],
                               fitted=fits[m]["fitted"][:, k]))
             for m in S.MODELS + ["NP"] for k in range(K)]
    with_config(cfg, pd.concat(parts)).to_csv(OUT / "06_meanfit_fitted.csv.gz", index=False)


def group_summary(ho, fits, null_med, groups):
    """Median skill and fit per group and model, with the surrogate test of the group median."""
    rows = []
    for g, gi in groups.items():
        for m in S.MODELS + ["L+time", "training-mean", "NP"]:
            skill = ho[m]["skill"][gi]
            row = dict(group=g, n_configs=len(gi), model=m, HO_median=np.median(skill),
                       RMSE_HO_median=np.median(ho[m]["rmse"][gi]))
            if m in fits:
                row.update(R2_median=np.median(fits[m]["r2"][gi]), RMSE_median=np.median(fits[m]["rmse"][gi]))
            if m in ("L", "L+time"):
                for other in ("time-only", "T-only", "L_time"):
                    row[f"share_beats_{other}"] = np.mean(skill > ho[other]["skill"][gi])
                both = (skill > ho["time-only"]["skill"][gi]) & (skill > ho["T-only"]["skill"][gi])
                row["n_beats_time-only_and_T-only"] = int(both.sum())
            if m in S.SURROGATE_MODELS and g in GROUPS:
                d = null_med[(m, g)]
                row.update(surrogate_median=np.median(d), surrogate_q95=np.quantile(d, 0.95),
                           surrogate_p=M.rank_p(d, row["HO_median"]),
                           test="primary" if (m, g) in PRIMARY_TESTS else "descriptive")
            rows.append(row)
    return pd.DataFrame(rows)


def group_difference(ho, null_med, groups):
    """Nonlinear minus conventional median skill against the same difference under the surrogates."""
    rows = []
    for m in S.SURROGATE_MODELS:
        real = np.median(ho[m]["skill"][groups["nonlinear"]]) - np.median(ho[m]["skill"][groups["conventional"]])
        d = null_med[(m, "nonlinear")] - null_med[(m, "conventional")]
        rows.append(dict(model=m, observed_difference=real, surrogate_median=np.median(d),
                         surrogate_q95=np.quantile(d, 0.95), surrogate_p=M.rank_p(d, real),
                         n_surrogates_at_least_observed=int(np.sum(d >= real)),
                         test="primary" if m == "L" else "descriptive"))
    return pd.DataFrame(rows)


def sensitivity(values):
    """Held-out skill under every sensitivity scenario; returns the group table and the per-configuration table."""
    scenarios = read_values("sensitivity_by_recording.csv.gz")
    runs = [("primary", values, True), (S.CENTRING_SCENARIO, values, False)]
    runs += [(name, scenario_values(values, scenarios, name), True) for name in S.SCENARIOS]
    rows, per_config = [], []
    for name, table, scale in runs:
        Ys, cs, vs = M.load_values(table, scale=scale)
        h = heldout(Ys, vs, S.MODELS)
        gs = M.groups_of(cs)
        for g in GROUPS:
            skill = {m: h[m]["skill"][gs[g]] for m in S.MODELS}
            row = dict(scenario=name, group=g, **{f"HO_{m}": np.median(v) for m, v in skill.items()})
            row["share_L_beats_time-only"] = np.mean(skill["L"] > skill["time-only"])
            row["share_L_beats_L_time"] = np.mean(skill["L"] > skill["L_time"])
            rows.append(row)
        per_config.append(cs.assign(scenario=name, HO_L=h["L"]["skill"]))
    return pd.DataFrame(rows), pd.concat(per_config)


def broadband_summary(cfg_b, ho_b, fits_b, null_b, per, fits, ho, null_med, groups):
    """Broadband conventional indices per index and in summary, compared with the octave-band versions."""
    bb = cfg_b.copy()
    for m in S.MODELS + ["training-mean"]:
        bb[f"HO_{m}"] = ho_b[m]["skill"]
    for m in S.MODELS:
        bb[f"R2_{m}"] = fits_b[m]["r2"]
    for m in ("T-only", "L"):
        bb[f"surrogate_p_{m}"] = [M.rank_p(null_b[m][:, k], ho_b[m]["skill"][k]) for k in range(len(cfg_b))]
    octave = per.assign(R2_L=fits["L"]["r2"])
    oc = octave[octave.family == "conventional"]
    bb["octave_R2_L_median"] = bb.measure.map(oc.groupby("measure").R2_L.median())
    bb["octave_HO_L_median"] = bb.measure.map(oc.groupby("measure").HO_L.median())
    bb["octave_HO_L_best"] = bb.measure.map(oc.groupby("measure").HO_L.max())

    d_null = null_med[("L", "nonlinear")] - np.median(null_b["L"], 1)
    d_real = np.median(ho["L"]["skill"][groups["nonlinear"]]) - np.median(bb.HO_L)
    summary = dict(**{f"HO_{m}": np.median(bb[f"HO_{m}"]) for m in S.MODELS},
                   **{f"R2_{m}": np.median(bb[f"R2_{m}"]) for m in S.MODELS})
    for m in ("T-only", "L"):
        d = np.median(null_b[m], 1)
        summary.update({f"surrogate_median_{m}": np.median(d), f"surrogate_q95_{m}": np.quantile(d, 0.95),
                        f"surrogate_p_{m}": M.rank_p(d, summary[f"HO_{m}"])})
    summary.update(octave_conventional_HO_L=np.median(oc.HO_L), octave_conventional_R2_L=np.median(oc.R2_L),
                   diff_nonlinear_minus_broadband=d_real, diff_surrogate_median=np.median(d_null),
                   diff_surrogate_q95=np.quantile(d_null, 0.95), diff_surrogate_p=M.rank_p(d_null, d_real))
    return bb, pd.DataFrame([summary])


def main_report(summary, gdiff, surr, ho, groups, np_info, resid, agree, pairs, flat, sens, bsum):
    np_bandwidths, bandwidth, edf = np_info
    L = [f"MODEL ANALYSIS ({S.N_TEMPERATURE_SURROGATES} temperature surrogates, "
         f"seed {S.TEMPERATURE_SURROGATE_SEED})"]
    for g in GROUPS:
        L += ["", f"{g} ({len(groups[g])} configurations)"]
        for r in summary[summary.group == g].itertuples():
            line = f"  {r.model:13s} skill {r.HO_median:6.3f}"
            if not np.isnan(getattr(r, "R2_median", np.nan)):
                line += f"   R2 {r.R2_median:5.3f}"
            if r.model in S.SURROGATE_MODELS:
                line += (f"   surrogates median {r.surrogate_median:.3f}, 95th percentile {r.surrogate_q95:.3f}, "
                         f"p = {r.surrogate_p:.5f} ({r.test})")
            L.append(line)
        s = summary[(summary.group == g) & (summary.model == "L")].iloc[0]
        L.append(f"  L better than time-only in {s['share_beats_time-only']:.0%}, than T-only in "
                 f"{s['share_beats_T-only']:.0%}, than both in {int(s['n_beats_time-only_and_T-only'])} "
                 f"configurations")
    per_measure = summary[summary.model == "L"].set_index("group").HO_median
    L += ["", "median skill of L per measure: " + ", ".join(f"{m} {per_measure[m]:.2f}" for m in S.MEASURES)]
    for r in gdiff.itertuples():
        L.append(f"nonlinear minus conventional, {r.model}: {r.observed_difference:.3f} (surrogates median "
                 f"{r.surrogate_median:.3f}, 95th percentile {r.surrogate_q95:.3f}, "
                 f"{r.n_surrogates_at_least_observed} at least as large, p = {r.surrogate_p:.5f}; {r.test})")

    near = surr.corr_with_real
    peak = int(np.argmax(M.T_MEAN)) + 1
    L += ["", f"temperature surrogates: correlation with the real course median {near.median():+.2f} (IQR "
              f"{near.quantile(.25):+.2f} to {near.quantile(.75):+.2f}), {np.mean(near >= NEAR_COPY_R):.1%} with "
              f"r >= {NEAR_COPY_R}; peak at occasion {peak} in {np.mean(surr.peak_occasion == peak):.0%}; "
              f"real course ends {M.T_MEAN[-1] - M.T_MEAN[0]:+.1f} C from its start",
          f"NP: bandwidth per fold {np_bandwidths} d, {bandwidth} d for the mean trajectory, {edf:.1f} effective "
          f"degrees of freedom"]
    d = ho["NP"]["skill"] - ho["L"]["skill"]
    by_measure = ", ".join(f"{m} {np.median(ho['NP']['skill'][groups[m]]):.2f}/"
                           f"{np.median(ho['L']['skill'][groups[m]]):.2f}" for m in S.MEASURES)
    L.append(f"NP minus L: above +0.10 in {int((d > 0.10).sum())} configurations "
             f"({int((d[groups['conventional']] > 0.10).sum())} conventional), below -0.05 in "
             f"{int((d < -0.05).sum())}; median skill NP/L by measure: {by_measure}")
    for g in ("nonlinear", "conventional"):
        r = resid[resid.family == g]
        L.append(f"residuals ({g}): " + "; ".join(
            f"{m} lag-1 {r[r.model == m].lag1_autocorrelation.median():.2f}, Ljung-Box p < 0.05 in "
            f"{np.mean(r[r.model == m].ljung_box_p < 0.05):.0%}" for m in ("time-only", "T-only", "L")))
    agreement = agree.groupby("family").median_pairwise_spearman.median()
    L.append(f"bucket agreement (median pairwise Spearman): nonlinear {agreement['nonlinear']:.2f}, "
             f"conventional {agreement['conventional']:.2f}")
    red = pairs[pairs.redundant]
    L.append(f"redundant pairs (|rho| >= {REDUNDANCY_RHO}): {int(red.same_band.sum())} within bands, "
             f"{int((~red.same_band & (red.measure_a == red.measure_b)).sum())} of one measure between bands")
    ratio = flat.v_from_entropy / flat.v_from_flatness
    L.append(f"flat spectra: median entropy {flat.median_entropy.min():.3f}-{flat.median_entropy.max():.3f}, "
             f"median flatness {flat.median_flatness.min():.3f}-{flat.median_flatness.max():.3f}, bins "
             f"{sorted(set(flat.n_bins))}, v from entropy / v from flatness {ratio.min():.2f}-{ratio.max():.2f}")
    s = sens[sens.group == "nonlinear"]
    L.append(f"sensitivity ({len(s) - 1} settings): nonlinear median skill of L {s.HO_L.min():.3f}-"
             f"{s.HO_L.max():.3f}; L above time-only in {int((s.HO_L > s['HO_time-only']).sum())} and above "
             f"L_time in {int((s.HO_L > s.HO_L_time).sum())} of {len(s)}")
    b = bsum.iloc[0]
    L.append(f"broadband conventional: skill of L {b.HO_L:.3f} (surrogates median {b.surrogate_median_L:.3f}, "
             f"95th percentile {b.surrogate_q95_L:.3f}, p = {b.surrogate_p_L:.3f}); nonlinear minus broadband "
             f"{b.diff_nonlinear_minus_broadband:.3f} (surrogates median {b.diff_surrogate_median:.3f}, "
             f"95th percentile {b.diff_surrogate_q95:.3f}, p = {b.diff_surrogate_p:.3f})")
    return L


# =============================================================================
# Main
# =============================================================================
def run():
    t0 = time.time()
    OUT.mkdir(parents=True, exist_ok=True)
    values = read_values("measures_by_recording.csv.gz")
    Y, cfg, valid = M.load_values(values)
    groups = M.groups_of(cfg)
    M.OCCASIONS.assign(T_mean=M.T_MEAN, cooling=M.cooling(M.T_MEAN)).to_csv(OUT / "00_occasions.csv", index=False)

    # held-out skill and fits to the mean trajectory
    ho = heldout(Y, valid, S.MODELS + ["L+time", "training-mean"])
    np_f, np_bandwidths = M.np_folds(Y, valid)
    ho["NP"] = dict(folds=np_f, per_bucket=M.skill(np_f), rmse=M.rmse(np_f))
    ho["NP"]["skill"] = ho["NP"]["per_bucket"].mean(0)
    fits = {m: M.mean_fit(Y, valid, m) for m in S.MODELS}
    fits["NP"], bandwidth, edf = np_mean_fit(Y, valid, fits)

    # broadband conventional indices
    Yb, cfg_b, valid_b = M.load_values(read_values("broadband_by_recording.csv"))
    ho_b = heldout(Yb, valid_b, S.MODELS + ["training-mean"])
    fits_b = {m: M.mean_fit(Yb, valid_b, m) for m in S.MODELS}

    # temperature surrogates
    null_all, surr = temperature_surrogates({"octave": (Y, valid, S.SURROGATE_MODELS),
                                             "broadband": (Yb, valid_b, ["T-only", "L"])})
    null = {m: null_all[("octave", m)] for m in S.SURROGATE_MODELS}
    null_b = {m: null_all[("broadband", m)] for m in ("T-only", "L")}
    null_med = {(m, g): np.median(null[m][:, groups[g]], axis=1) for m in S.SURROGATE_MODELS for g in GROUPS}
    for (m, g), v in null_med.items():
        surr[f"median_skill_{m}_{g}"] = v
    surr["median_skill_L_broadband"] = np.median(null_b["L"], axis=1)
    surr.to_csv(OUT / "07_temperature_surrogates.csv.gz", index=False)

    # per configuration
    per = cfg.copy()
    for m in S.MODELS + ["L+time", "training-mean", "NP"]:
        per[f"HO_{m}"] = ho[m]["skill"]
        per[f"RMSE_HO_{m}"] = ho[m]["rmse"]
    for m in S.SURROGATE_MODELS:
        per[f"surrogate_p_{m}"] = [M.rank_p(null[m][:, k], ho[m]["skill"][k]) for k in range(len(cfg))]
    for m in ("time-only", "T-only", "L_time"):
        per[f"gain_L_vs_{m}"] = ho["L"]["skill"] - ho[m]["skill"]
    per.to_csv(OUT / "02_cv_per_config.csv", index=False)
    write_fit_tables(cfg, ho, fits, edf)

    # group summaries and the three primary tests
    summary = group_summary(ho, fits, null_med, groups)
    summary.to_csv(OUT / "01_group_summary.csv", index=False)
    gdiff = group_difference(ho, null_med, groups)
    gdiff.to_csv(OUT / "08_group_difference.csv", index=False)

    # diagnostics, agreement, redundancy
    resid = residual_diagnostics({m: fits[m] for m in ("time-only", "T-only", "L")}, cfg)
    resid.to_csv(OUT / "09_residual_diagnostics.csv", index=False)
    agree = bucket_agreement(Y, valid, cfg)
    agree.to_csv(OUT / "10_bucket_agreement.csv", index=False)
    rho = correlation_matrix(Y, valid, cfg)
    rho.to_csv(OUT / "11_correlation_matrix.csv")
    pairs, red_summary = redundancy(rho, cfg)
    pairs.to_csv(OUT / "11_redundancy_pairs.csv.gz", index=False)
    red_summary.to_csv(OUT / "12_redundancy_summary.csv", index=False)
    flat = flat_spectrum_check(values)

    # sensitivity scenarios and broadband indices
    sens, sens_per_config = sensitivity(values)
    sens.to_csv(OUT / "13_sensitivity_scenarios.csv", index=False)
    sens_per_config.to_csv(OUT / "14_sensitivity_per_config.csv", index=False)
    bb, bsum = broadband_summary(cfg_b, ho_b, fits_b, null_b, per, fits, ho, null_med, groups)
    bb.to_csv(OUT / "15_broadband_per_index.csv", index=False)
    bsum.to_csv(OUT / "16_broadband_summary.csv", index=False)

    L = main_report(summary, gdiff, surr, ho, groups, (np_bandwidths, bandwidth, edf),
                    resid, agree, pairs, flat, sens, bsum)
    checks, lines = further_checks(Y, cfg, valid, ho, fits, null, surr)
    for name, table in sorted(checks.items()):
        table.to_csv(OUT / f"{name}.csv", index=False)
    L += lines
    t28, energy_lines = band_energy_check(values, pairs)
    t28.to_csv(OUT / "28_band_energy.csv", index=False)
    L += energy_lines
    t31, top_lines = top_band_check(cfg, ho, null, groups)
    t31.to_csv(OUT / "31_without_top_band.csv", index=False)
    L += top_lines
    t30, cross_lines = cross_prediction(Y, valid, cfg, groups)
    t30.to_csv(OUT / "30_cross_prediction.csv", index=False)
    L += cross_lines
    L.append(f"runtime {(time.time() - t0) / 60:.1f} min")
    (OUT / "report.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))


if __name__ == "__main__":
    run()
