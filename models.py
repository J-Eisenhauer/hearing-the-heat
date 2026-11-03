"""Data preparation and the models: standardised trajectories, designs, least-squares fits, leave-one-bucket-out
cross-validation, the nonparametric benchmark NP and IAAFT surrogates of the temperature course."""
import numpy as np
import pandas as pd

import settings as S

COEFFICIENTS = {"time-only": ["alpha", "beta"], "T-only": ["alpha", "d"], "L": ["alpha", "b", "d", "e"],
                "L_time": ["alpha", "b", "beta", "e"], "L+time": ["alpha", "b", "d", "e", "beta"]}
NP_BANDWIDTHS_DAYS = [0.4, 0.75, 1.0, 1.5, 2.0, 3.0, 4.0]
IAAFT_MAX_ITER = 1000


# =============================================================================
# Occasions and covariates
# =============================================================================
def _hours(clock):
    h, m = str(clock).split(":")[:2]
    return int(h) + int(m) / 60.0


def load_occasions():
    """Time (days since the first occasion), bucket temperatures and afternoon/evening indicator of every occasion."""
    p = pd.read_csv(S.PROTOCOL_FILE).sort_values("occasion").iloc[:S.N_OCCASIONS].reset_index(drop=True)
    starts = list(zip(p.date, p.start_time))
    days = np.array([pd.Timestamp(d).value / 864e11 + (_hours(t) + S.OCCASION_TIME_OFFSET_H) / 24.0
                     for d, t in starts])
    days -= days[0]
    p["time_days"] = days
    p["day"] = np.where(p.session == "D", 1.0, np.where(p.session == "N", 0.0, 0.5))   # 1 = afternoon (13:00), 0 = evening (21:00); baseline (18:00): 0.5
    return p


OCCASIONS = load_occasions()
T_IN = OCCASIONS[[f"T_{b}" for b in S.BUCKETS]].to_numpy(float).T      # buckets x occasions
T_MEAN = T_IN.mean(0)
TIME_DAYS = OCCASIONS.time_days.to_numpy()
TIME = TIME_DAYS - TIME_DAYS.mean()                                     # centred time in the models
DAY = OCCASIONS.day.to_numpy()


def cooling(t_mean):
    """1 after the peak of the given mean temperature course."""
    return (np.arange(len(t_mean)) > int(np.argmax(t_mean))).astype(float)


# =============================================================================
# Standardised values
# =============================================================================
def load_values(table, scale=True):
    """Standardised values Y (configurations x buckets x occasions), configuration table and validity mask.

    Each bucket and configuration is centred on its mean over all occasions and, with scale=True, divided by
    its standard deviation. An occasion of a bucket is valid when it has a value for every configuration."""
    configs, Y = [], []
    for measure in S.MEASURES:
        sub = table[table.measure == measure]
        for fmin in sorted(sub.fmin_hz.unique()):
            s = sub[sub.fmin_hz == fmin]
            M = np.full((len(S.BUCKETS), S.N_OCCASIONS), np.nan)
            for b, o, v in zip(s.bucket, s.occasion, s.value):
                if o <= S.N_OCCASIONS:
                    M[S.BUCKETS.index(b), o - 1] = v
            M = M - np.nanmean(M, 1, keepdims=True)
            if scale:
                M = M / np.nanstd(M, 1, ddof=1, keepdims=True)
            configs.append(dict(family=S.FAMILY[measure], measure=measure, fmin_hz=int(fmin),
                                band=f"{int(fmin)}-{2 * int(fmin)} Hz"))
            Y.append(M)
    Y = np.array(Y)
    return Y, pd.DataFrame(configs), ~np.isnan(Y).any(0)


def groups_of(cfg):
    g = {"nonlinear": cfg.index[cfg.family == "nonlinear"].to_numpy(),
         "conventional": cfg.index[cfg.family == "conventional"].to_numpy(), "all": cfg.index.to_numpy()}
    for m in cfg.measure.unique():
        g[m] = cfg.index[cfg.measure == m].to_numpy()
    return g


def mean_trajectory(Y, valid):
    """Mean over the valid buckets of every configuration (configurations x occasions)."""
    return np.nanmean(np.where(valid[None], Y, np.nan), 1)


# =============================================================================
# Designs and least squares
# =============================================================================
def design(model, t_in, cool, extra=None):
    """Design matrix with intercept for one bucket (own temperature) or the mean trajectory (mean temperature).
    `extra` adds one covariate column to L (afternoon/evening session or mains hum)."""
    c = t_in - S.T_REF_C
    one = np.ones(len(c))
    X = {"time-only": lambda: [one, TIME],
         "T-only": lambda: [one, c],
         "L": lambda: [one, cool, c, cool * c],
         "L_time": lambda: [one, cool, TIME, cool * TIME],
         "L+time": lambda: [one, cool, c, cool * c, TIME]}[model]()
    if extra is not None:
        X.append(extra)
    return np.column_stack(X)


def fit(X, y):
    return np.linalg.lstsq(X, y, rcond=None)[0]


def cv_folds(Y, valid, model, t_in=None, covariate=None, mean_temperature=False):
    """Leave-one-bucket-out predictions. Every model is fitted to all values of four buckets, each with its own
    temperature, and predicts the fifth from its own temperature; the cooling indicator follows the mean
    temperature of the training buckets. mean_temperature=True gives every bucket the training-mean course.
    Returns one dict per held-out bucket with observed and predicted values (occasions x configurations)."""
    t_in = T_IN if t_in is None else t_in
    folds = []
    for h in range(len(S.BUCKETS)):
        train = [b for b in range(len(S.BUCKETS)) if b != h]
        t_train = t_in[train].mean(0)
        cool = cooling(t_train)
        temp = (lambda b: t_train) if mean_temperature else (lambda b: t_in[b])
        extra = (lambda b: covariate[b]) if covariate is not None else (lambda b: None)
        X = np.vstack([design(model, temp(b), cool, extra(b))[valid[b]] for b in train])
        y = np.vstack([Y[:, b, valid[b]].T for b in train])
        X_h = design(model, temp(h), cool, extra(h))[valid[h]]
        folds.append(dict(observed=Y[:, h, valid[h]].T, predicted=X_h @ fit(X, y), heating=cool[valid[h]] == 0))
    return folds


def skill(folds, part=None):
    """1 - SSE / SSE0 per held-out bucket (buckets x configurations); SSE0 = sum of squares of the standardised
    values. part='heating' or 'cooling' restricts the score to the occasions of one phase."""
    out = []
    for f in folds:
        keep = np.ones(len(f["heating"]), bool) if part is None else (f["heating"] if part == "heating"
                                                                       else ~f["heating"])
        y, p = f["observed"][keep], f["predicted"][keep]
        out.append(1 - ((y - p) ** 2).sum(0) / (y ** 2).sum(0))
    return np.array(out)


def ceiling_folds(Y, valid):
    """Mean of the four training buckets used directly as prediction of the held-out bucket."""
    folds = []
    for h in range(len(S.BUCKETS)):
        train = [b for b in range(len(S.BUCKETS)) if b != h]
        mean = np.nanmean(np.where(valid[train][None], Y[:, train, :], np.nan), 1)
        folds.append(dict(observed=Y[:, h, valid[h]].T, predicted=mean[:, valid[h]].T,
                          heating=np.zeros(valid[h].sum(), bool)))
    return folds


def rmse(folds):
    sse = sum(((f["observed"] - f["predicted"]) ** 2).sum(0) for f in folds)
    return np.sqrt(sse / sum(len(f["observed"]) for f in folds))


class SurrogateCV:
    """Held-out skill of the temperature models for many temperature courses, with the data stacks cached."""

    def __init__(self, Y, valid, models):
        self.models, self.valid, self.folds = models, valid, []
        for h in range(len(S.BUCKETS)):
            train = [b for b in range(len(S.BUCKETS)) if b != h]
            y_h = Y[:, h, valid[h]].T
            self.folds.append((h, train, np.vstack([Y[:, b, valid[b]].T for b in train]), y_h, (y_h ** 2).sum(0)))

    def __call__(self, t_in):
        out = {}
        for m in self.models:
            total = 0.0
            for h, train, y, y_h, sse0 in self.folds:
                cool = cooling(t_in[train].mean(0))
                X = np.vstack([design(m, t_in[b], cool)[self.valid[b]] for b in train])
                p = design(m, t_in[h], cool)[self.valid[h]] @ fit(X, y)
                total = total + 1 - ((y_h - p) ** 2).sum(0) / sse0
            out[m] = total / len(S.BUCKETS)
        return out


def mean_fit(Y, valid, model, covariate=None, keep=None):
    """Least-squares fit to the mean trajectory of the buckets with their mean temperature.
    `keep` restricts the fit to a subset of occasions (heating-occasion jackknife)."""
    ybar = mean_trajectory(Y, valid)
    ok = ~np.isnan(ybar).any(0)
    if keep is not None:
        ok = ok & keep
    y = ybar[:, ok].T
    n = int(ok.sum())
    X = design(model, T_MEAN, cooling(T_MEAN), covariate)[ok]
    B = fit(X, y)
    f = X @ B
    sse = ((y - f) ** 2).sum(0)
    r2 = 1 - sse / ((y - y.mean(0)) ** 2).sum(0)
    return dict(coef=B, fitted=f, observed=y, ok=ok, r2=r2,
                rmse=np.sqrt(sse / n))


# =============================================================================
# Nonparametric benchmark NP
# =============================================================================
KERNELS = {s: np.exp(-0.5 * ((TIME_DAYS[:, None] - TIME_DAYS[None, :]) / s) ** 2) for s in NP_BANDWIDTHS_DAYS}


def smooth(m, s):
    """Gaussian-kernel (Nadaraya-Watson) smooth over time; missing occasions are masked."""
    w = (~np.isnan(m)).astype(float)
    return (np.where(np.isnan(m), 0.0, m) * w) @ KERNELS[s].T / np.maximum(w @ KERNELS[s].T, 1e-12)


def bucket_mean(Y, valid, buckets):
    with np.errstate(all="ignore"):
        return np.nanmean(np.where(valid[None, buckets], Y[:, buckets], np.nan), 1)


def choose_bandwidth(Y, valid, buckets):
    """One bandwidth for all configurations: leave-one-bucket-out within `buckets`, smallest summed squared error."""
    sse = {}
    for s in NP_BANDWIDTHS_DAYS:
        sse[s] = 0.0
        for b in buckets:
            c = smooth(bucket_mean(Y, valid, [x for x in buckets if x != b]), s)
            sse[s] += ((Y[:, b, valid[b]] - c[:, valid[b]]) ** 2).sum()
    return min(sse, key=sse.get)


def np_folds(Y, valid):
    """Leave-one-bucket-out predictions of NP; the bandwidth is chosen within the training buckets of each fold."""
    folds, chosen = [], []
    for h in range(len(S.BUCKETS)):
        train = [b for b in range(len(S.BUCKETS)) if b != h]
        s = choose_bandwidth(Y, valid, train)
        chosen.append(s)
        curve = smooth(bucket_mean(Y, valid, train), s)
        folds.append(dict(observed=Y[:, h, valid[h]].T, predicted=curve[:, valid[h]].T,
                          heating=np.zeros(valid[h].sum(), bool)))
    return folds, chosen


def np_smoother(ok, s):
    """Row-normalised smoother matrix on the occasions used for the mean trajectory."""
    W = KERNELS[s][np.ix_(ok, ok)]
    return W / W.sum(1, keepdims=True)


# =============================================================================
# Temperature surrogates
# =============================================================================
def iaaft(x, rng):
    """IAAFT surrogate of a short series: same values and power spectrum, randomised timing."""
    ranked = np.sort(x)
    amplitude = np.abs(np.fft.rfft(x))
    y = rng.permutation(x)
    for _ in range(IAAFT_MAX_ITER):
        z = np.fft.irfft(amplitude * np.exp(1j * np.angle(np.fft.rfft(y))), n=len(x))
        z = ranked[np.argsort(np.argsort(z))]
        if np.array_equal(z, y):
            break
        y = z
    return y


def rank_p(null, observed):
    """(1 + number of surrogates at least as large as the observed value) / (1 + number of surrogates)"""
    return (1 + np.sum(null >= observed)) / (1 + len(null))
