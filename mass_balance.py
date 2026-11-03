"""Mass and moisture balance of the five buckets (Monte Carlo propagation of the weighing uncertainties).

Two estimates of dry-mass loss are reported separately: the direct change in oven-dry mass, and the retained
metabolic water converted to oxidised dry matter. Balance specifications (KERN data sheets): HCB 50K20 readability
0.02 kg, repeatability 0.02 kg, linearity 0.04 kg; EMB 5.2K1 readability 1 g, repeatability 1 g, linearity 3 g.
Metabolic-water yields: EFSA NDA Panel (2010), EFSA Journal 8(3):1459.
"""
import numpy as np
import pandas as pd

import settings as S

SEED = 20260909
N_DRAWS = 200_000

# balance specifications: (repeatability SD, linearity limit, readability)
HCB_KG = (0.020, 0.040, 0.020)
EMB_G = (1.0, 3.0, 1.0)

# readings: bucket plus content on the HCB (kg), empty buckets on the EMB (g)
GROSS_BEFORE_KG = np.array([6.30, 6.38, 5.74, 6.08, 5.58])
GROSS_AFTER_KG = np.array([6.12, 6.14, 5.50, 5.80, 5.34])
TARE_G = np.array([715.0, 704.0, 712.0, 695.0, 695.0])

# hay before and after soaking (HCB, kg); its moisture before soaking was not measured
HAY_BEFORE_SOAKING_KG = 3.82
HAY_AFTER_SOAKING_KG = 12.98
HAY_MOISTURE = (0.050, 0.075, 0.110)                  # triangular (min, mode, max), fraction of the as-is mass

# moisture samples dried at 60 C for 48 h (EMB, g, including the 100 g container)
CONTAINER_G = 100.0
VERMICOMPOST_WET_G, VERMICOMPOST_DRY_G = 715.0, 343.0
END_WET_G = {"A": (681.0,), "B": (680.0, 672.0), "C": (710.0,), "D": (576.0,), "E": (641.0,)}
END_DRY_G = {"A": (276.0,), "B": (264.0, 270.0), "C": (270.0,), "D": (226.0,), "E": (249.0,)}

MIXING_RATIO = (2.00, 0.02)                           # vermicompost : hay wet mass, mean and SD
WATER_YIELD = (0.48, 0.556, 0.62)                     # kg water per kg oxidised dry matter, triangular


def reading(rng, value, size, spec):
    """One balance reading: normal repeatability plus uniform linearity and display-rounding errors."""
    repeatability, linearity, readability = spec
    return (np.asarray(value, dtype=float) + rng.normal(0.0, repeatability, size)
            + rng.uniform(-linearity, linearity, size) + rng.uniform(-readability / 2, readability / 2, size))


def positive_normal(rng, mean, sd, size):
    x = rng.normal(mean, sd, size)
    while np.any(x <= 0):
        x[x <= 0] = rng.normal(mean, sd, int(np.sum(x <= 0)))
    return x


def end_water_fraction(bucket):
    wet = sum(END_WET_G[bucket]) - CONTAINER_G * len(END_WET_G[bucket])
    dry = sum(END_DRY_G[bucket]) - CONTAINER_G * len(END_DRY_G[bucket])
    return 1 - dry / wet


def run():
    S.MASS_DIR.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(SEED)
    n, nb = N_DRAWS, len(S.BUCKETS)

    # net masses; the sampled tare of a bucket is used before and after
    tare = reading(rng, TARE_G, (n, nb), EMB_G) / 1000.0
    before = reading(rng, GROSS_BEFORE_KG, (n, nb), HCB_KG) - tare
    after = reading(rng, GROSS_AFTER_KG, (n, nb), HCB_KG) - tare

    # initial water fraction of the substrate (no dry-matter loss during soaking)
    hay = reading(rng, HAY_BEFORE_SOAKING_KG, n, HCB_KG)
    hay_soaked = reading(rng, HAY_AFTER_SOAKING_KG, n, HCB_KG)
    water_hay = 1 - hay * (1 - rng.triangular(*HAY_MOISTURE, n)) / hay_soaked
    container = reading(rng, CONTAINER_G, n, EMB_G)
    vermicompost_wet = reading(rng, VERMICOMPOST_WET_G, n, EMB_G) - container
    vermicompost_dry = reading(rng, VERMICOMPOST_DRY_G, n, EMB_G) - container
    water_vermicompost = 1 - vermicompost_dry / vermicompost_wet
    ratio = positive_normal(rng, *MIXING_RATIO, n)
    water_start = np.clip((ratio * water_vermicompost + water_hay) / (ratio + 1), 0, 1)

    # final water fraction; bucket B combines a top and a bottom sample, half their difference is the
    # representation uncertainty of one sample, applied to every bucket
    container = reading(rng, CONTAINER_G, n, EMB_G)
    water_end = np.empty((n, nb))
    for j, b in enumerate(S.BUCKETS):
        wet = np.column_stack([reading(rng, v, n, EMB_G) for v in END_WET_G[b]])
        dry = np.column_stack([reading(rng, v, n, EMB_G) for v in END_DRY_G[b]])
        water_end[:, j] = 1 - np.sum(dry - container[:, None], 1) / np.sum(wet - container[:, None], 1)
    top, bottom = [1 - (d - CONTAINER_G) / (w - CONTAINER_G) for w, d in zip(END_WET_G["B"], END_DRY_G["B"])]
    water_end = np.clip(water_end + rng.normal(0.0, abs(top - bottom) / 2, water_end.shape), 0, 1)

    dry_start = before * (1 - water_start[:, None])
    direct_loss = dry_start - after * (1 - water_end)
    water_loss = (after * water_end - before * water_start[:, None]) / rng.triangular(*WATER_YIELD, n)[:, None]

    rows = []
    for j, b in enumerate(S.BUCKETS):
        d, w = 100 * direct_loss[:, j] / dry_start[:, j], 100 * water_loss[:, j] / dry_start[:, j]
        rows.append(dict(bucket=b, net_mass_start_kg=before[:, j].mean(), net_mass_end_kg=after[:, j].mean(),
                         water_fraction_end=end_water_fraction(b), direct_loss_pct=d.mean(),
                         direct_loss_sd=d.std(ddof=1), retained_water_loss_pct=w.mean(),
                         retained_water_loss_sd=w.std(ddof=1)))
    pd.DataFrame(rows).to_csv(S.MASS_DIR / "mass_balance_by_bucket.csv", index=False)
    pooled = []
    for name, loss in (("direct", direct_loss), ("retained water", water_loss)):
        x = 100 * loss.sum(1) / dry_start.sum(1)
        pooled.append(dict(estimate=name, mean_pct=x.mean(), sd_pct=x.std(ddof=1), q025=np.quantile(x, 0.025),
                           q975=np.quantile(x, 0.975)))
    pooled = pd.DataFrame(pooled)
    pooled.to_csv(S.MASS_DIR / "mass_balance_pooled.csv", index=False)
    vermicompost_water = 1 - (VERMICOMPOST_DRY_G - CONTAINER_G) / (VERMICOMPOST_WET_G - CONTAINER_G)
    lines = [f"water fraction at the start: vermicompost {vermicompost_water:.3f}"
             f", soaked hay at the mode of the hay moisture "
             f"{1 - HAY_BEFORE_SOAKING_KG * (1 - HAY_MOISTURE[1]) / HAY_AFTER_SOAKING_KG:.2f}, mixture "
             f"{np.median(water_start):.2f}; end-sample representation SD {abs(top - bottom) / 2:.4f}"]
    lines += [f"pooled dry-mass loss, {r.estimate}: {r.mean_pct:.1f}% (95% interval {r.q025:.1f}-{r.q975:.1f}%)"
              for r in pooled.itertuples()]
    (S.MASS_DIR / "report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    run()
