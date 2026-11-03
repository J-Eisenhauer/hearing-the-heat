"""LaTeX tables of the manuscript (Table 1) and the Supplementary Material (Tables S2-S10)."""
import pandas as pd

import settings as S

MODEL_TEX = {"time-only": "time-only", "T-only": "temperature-only", "L_time": "L$_{\\mathrm{time}}$", "L": "L"}
SCENARIO_TEX = {
    "primary": "primary analysis", "window_low": "window 7.5 s", "window_high": "window 30 s",
    "lambda_low": "lambda multiplier 0.5", "lambda_high": "lambda multiplier 2",
    "lambda_power_low": "lambda exponent $-0.25$", "lambda_power_high": "lambda exponent $-1.0$",
    "rms_cycles_low": "RMS width $2.5\\lambda/f_{\\min}$", "rms_cycles_high": "RMS width $10\\lambda/f_{\\min}$",
    "lzc_alphabet_low": "LZC alphabet 50", "lzc_alphabet_high": "LZC alphabet 200",
    "pe_order_low": "PE order 3", "pe_order_high": "PE order 5",
    "rse_block_low": "RSE block length 3", "rse_block_high": "RSE block length 5",
    "rse_alphabet_low": "RSE alphabet 8", "rse_alphabet_high": "RSE alphabet 12",
    "combined_low_stress": "combined low", "combined_high_stress": "combined high",
    S.CENTRING_SCENARIO: "centring only (no SD scaling)"}
LOUDNESS_TEX = {"bucket": "Quantiser per bucket (main analysis)", "recording": "Quantiser per recording (control)",
                "window": "Quantiser per 15\\,s window", "recording, median": "Per recording, two median windows only"}
GROUP_HEADER = (" & \\multicolumn{3}{c}{Nonlinear (32)} & \\multicolumn{3}{c}{Conventional (48)}\\\\\n"
                "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n")


def tabular(columns, header, lines):
    return (f"\\begin{{tabular}}{{{columns}}}\n\\toprule\n{header}\\midrule\n" + "\n".join(lines)
            + "\n\\bottomrule\n\\end{tabular}\n")


def p_value(p, primary=False):
    if pd.isna(p):
        return "--"
    text = f"{p:.5f}" if p < 0.001 else f"{p:.3f}"
    return f"\\textbf{{{text}}}" if primary else text


def model_comparison(summary):
    """Table 1: R2 of the fit to the mean trajectory, held-out skill and surrogate p per model and group."""
    lines = []
    for m in S.MODELS:
        cells = []
        for g in ("nonlinear", "conventional"):
            r = summary[(summary.group == g) & (summary.model == m)].iloc[0]
            cells += [f"{r.R2_median:.3f}", f"{r.HO_median:.3f}", p_value(r.surrogate_p, r.test == "primary")]
        lines.append(f"{MODEL_TEX[m]} & " + " & ".join(cells) + "\\\\")
    cells = []
    for g in ("nonlinear", "conventional"):
        r = summary[(summary.group == g) & (summary.model == "NP")].iloc[0]
        cells += [f"{r.R2_median:.3f}", f"{r.HO_median:.3f}", "--"]
    lines.append("\\midrule\nNP & " + " & ".join(cells) + "\\\\")
    return tabular("lcccccc", GROUP_HEADER + "Model & $R^2$ & skill & $p$ & $R^2$ & skill & $p$\\\\\n", lines)


def sensitivity(sens):
    """Table S2: median held-out skill for every preprocessing scenario; conventional cells are left blank where
    the scenario does not affect the conventional measures."""
    columns = ["HO_time-only", "HO_L_time", "HO_L"]
    order = ["primary"] + list(S.SCENARIOS) + [S.CENTRING_SCENARIO]
    lines = []
    for name in order:
        cells = []
        for g in ("nonlinear", "conventional"):
            r = sens[(sens.scenario == name) & (sens.group == g)].iloc[0]
            affected = name in ("primary", S.CENTRING_SCENARIO) or any(
                m in S.CONVENTIONAL for m in S.SCENARIOS[name][1])
            cells += [f"{r[c]:.3f}" if g == "nonlinear" or affected else "" for c in columns]
        lines.append(("\\midrule\n" if name == S.CENTRING_SCENARIO else "") + f"{SCENARIO_TEX[name]} & "
                     + " & ".join(cells) + "\\\\" + ("\\midrule" if name == "primary" else ""))
    header = (" & \\multicolumn{3}{c}{Nonlinear} & \\multicolumn{3}{c}{Conventional}\\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\nScenario & time-only & L$_{\\mathrm{time}}$ & L & time-only & "
              "L$_{\\mathrm{time}}$ & L\\\\\n")
    return tabular("lcccccc", header, lines)


def broadband(per_index, summary):
    """Table S3: conventional indices computed broadband against their octave-band versions."""
    names = {"LogBandEnergy": "log band energy", "SpectralEntropy": "spectral entropy",
             "SpectralFlatness": "spectral flatness", "TemporalEntropy": "temporal entropy"}
    lines = [f"{names.get(r.measure, r.measure)} & {r.R2_L:.2f} & {r.HO_L:.2f} & {r.surrogate_p_L:.3f} & "
             f"{r.octave_R2_L_median:.2f} & {r.octave_HO_L_median:.2f} & {r.octave_HO_L_best:.2f}\\\\"
             for r in per_index.itertuples()]
    b = summary.iloc[0]
    lines.append(f"\\midrule\nmedian & {b.R2_L:.2f} & {b.HO_L:.2f} & {b.surrogate_p_L:.3f} & "
                 f"{b.octave_conventional_R2_L:.2f} & {b.octave_conventional_HO_L:.2f} & --\\\\")
    header = (" & \\multicolumn{3}{c}{Broadband 64--16384\\,Hz} & \\multicolumn{3}{c}{Octave bands}\\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
              "Index & $R^2$ & skill & $p$ & median $R^2$ & median skill & best skill\\\\\n")
    return tabular("lcccccc", header, lines)


def mass_balance(buckets, pooled):
    """Table S4: masses, final water fractions and dry-mass losses per bucket."""
    lines = [f"{r.bucket} & {r.net_mass_start_kg:.3f} & {r.net_mass_end_kg:.3f} & {r.water_fraction_end:.3f} & "
             f"${r.direct_loss_pct:.1f}\\pm{r.direct_loss_sd:.1f}$ & "
             f"${r.retained_water_loss_pct:.1f}\\pm{r.retained_water_loss_sd:.1f}$\\\\" for r in buckets.itertuples()]
    d, w = pooled.iloc[0], pooled.iloc[1]
    lines.append(f"\\midrule\nPooled & & & & {d.mean_pct:.1f} ({d.q025:.1f}--{d.q975:.1f}) & "
                 f"{w.mean_pct:.1f} ({w.q025:.1f}--{w.q975:.1f})\\\\")
    header = (" & \\multicolumn{2}{c}{Net mass (kg)} & Water fraction & \\multicolumn{2}{c}{Dry-mass loss (\\%)}\\\\\n"
              "\\cmidrule(lr){2-3}\\cmidrule(lr){5-6}\nBucket & start & end & end & direct & retained water\\\\\n")
    return tabular("lcccccc", header, lines)


def skill_by_phase(phase):
    """Table S5: held-out skill over all, heating and cooling occasions."""
    rows = {g: phase[phase.group == g].iloc[0] for g in ("nonlinear", "conventional")}
    lines = [f"{MODEL_TEX[m]} & " + " & ".join(f"{rows[g][f'{part}_{m}']:.2f}" for g in rows
                                               for part in ("all", "heating", "cooling")) + "\\\\"
             for m in S.MODELS]
    share = lambda col: " & ".join(f"\\multicolumn{{3}}{{c}}{{{rows[g][col]:.0%}}}" for g in rows)
    lines.append(f"\\midrule\nshare of $\\mathrm{{SSE}}_0$ (heating) & {share('share_SSE0_heating')}\\\\")
    lines.append(f"share of explained SS (heating, L) & {share('share_explained_heating')}\\\\")
    header = GROUP_HEADER + "Model & all & heating & cooling & all & heating & cooling\\\\\n"
    return tabular("lcccccc", header, lines).replace("%", "\\%")


def covariates(cov):
    """Table S6: afternoon/evening session and mains hum as an additional term of L."""
    lines = [f"{r.covariate} & {r.group} & ${r.diff_median_skill:+.3f}$ & ${r.diff_median_R2:+.3f}$ & "
             f"{r.n_gain}/{r.n_loss} & ${r.max_gain:+.2f}$ / ${r.max_loss:+.2f}$\\\\"
             for r in cov[cov.group != "all"].itertuples()]
    header = "Covariate & Group & $\\Delta$ skill & $\\Delta R^2$ & gain/loss & max/min\\\\\n"
    return tabular("llcccc", header, lines)


def ljung_box_by_band(resid):
    """Table S7: configurations per band whose residuals had Ljung-Box p < 0.05 (fit to the mean trajectory)."""
    models = ("time-only", "T-only", "L")
    lines = []
    for fmin in sorted(resid.fmin_hz.unique()):
        cells = []
        for fam in ("nonlinear", "conventional"):
            for m in models:
                r = resid[(resid.fmin_hz == fmin) & (resid.family == fam) & (resid.model == m)]
                cells.append(f"{int((r.ljung_box_p < 0.05).sum())}/{len(r)}")
        lines.append(f"{int(fmin)}--{2 * int(fmin)} Hz & " + " & ".join(cells) + "\\\\")
    short = {"time-only": "time", "T-only": "T", "L": "L"}
    header = GROUP_HEADER + "Band & " + " & ".join(short[m] for m in models * 2) + "\\\\\n"
    return tabular("lcccccc", header, lines)


def without_top_band(top):
    """Table S8: main comparisons with and without the 8192-16384 Hz band."""
    lines = []
    for _, r in top.iterrows():
        bands = "all" if r["bands"] == "all bands" else "without 8--16\\,kHz"
        better = f"{r['n_L_better_than_time-only']}/{r['n_L_better_than_T-only']}/{r['n_L_better_than_L_time']}"
        p = p_value(r["surrogate_p_L"], primary=(r["group"] == "nonlinear" and r["bands"] == "all bands"))
        lines.append(f"{r['group']} & {bands} & {r['n_configs']} & {r['skill_time-only']:.2f} & {r['skill_T-only']:.2f} & "
                     f"{r['skill_L_time']:.2f} & {r['skill_L']:.2f} & {r['skill_NP']:.2f} & {better} & {p}\\\\")
    header = "Group & Bands & $n$ & time-only & T-only & L$_{\\mathrm{time}}$ & L & NP & L better & $p$ (L)\\\\\n"
    return tabular("llcccccccc", header, lines)


def loudness_variants(var):
    """Table S9: skill and fit with the quantiser fitted per bucket, per recording, per window and per recording in
    the median windows only."""
    lines = []
    for q, label in LOUDNESS_TEX.items():
        cells = []
        for g in ("nonlinear", "quantised"):
            r = var[(var.quantiser == q) & (var.group == g)].iloc[0]
            cells += [f"{r.skill_time_only:.2f}", f"{r.skill_L:.2f}", f"{r.R2_L:.2f}",
                      f"{r.n_L_better_than_time}/{r.n}"]
        lines.append(f"{label} & " + " & ".join(cells) + "\\\\")
    n = {g: var[var.group == g].n.iloc[0] for g in ("nonlinear", "quantised")}
    header = (f" & \\multicolumn{{4}}{{c}}{{Nonlinear ({n['nonlinear']})}} & "
              f"\\multicolumn{{4}}{{c}}{{LZC78, RSE$_4$, RSR ({n['quantised']})}}\\\\\n"
              "\\cmidrule(lr){2-5}\\cmidrule(lr){6-9}\n"
              "Quantiser & time & L & $R^2$ of L & L $>$ time & time & L & $R^2$ of L & L $>$ time\\\\\n")
    return tabular("lcccccccc", header, lines)


def loudness_residuals(corr):
    """Table S10: correlation of the residuals of L of LZC78, RSE4 and RSR with those of PE in the same band."""
    signed = lambda r: f"{r:+.2f}".replace("-", "$-$")
    lines = []
    for fmin, g in corr.groupby("fmin_hz"):
        g = g.set_index("measure")
        cells = [signed(g.loc[m, f"r_{q}"]) for q in ("bucket", "recording") for m in ("LZC78", "RSE4", "RSR")]
        lines.append(f"{int(fmin)}--{2 * int(fmin)} & " + " & ".join(cells) + "\\\\")
    header = (" & \\multicolumn{3}{c}{Quantiser per bucket} & "
              "\\multicolumn{3}{c}{Quantiser per recording}\\\\\n"
              "\\cmidrule(lr){2-4}\\cmidrule(lr){5-7}\n"
              "Band (Hz) & LZC78 & RSE$_4$ & RSR & LZC78 & RSE$_4$ & RSR\\\\\n")
    return tabular("lcccccc", header, lines)


def run():
    S.TABLES_DIR.mkdir(parents=True, exist_ok=True)
    m = lambda name: pd.read_csv(S.MODELS_DIR / name)
    out = {
        "table1_model_comparison.tex": model_comparison(m("01_group_summary.csv")),
        "tableS2_sensitivity.tex": sensitivity(m("13_sensitivity_scenarios.csv")),
        "tableS3_broadband.tex": broadband(m("15_broadband_per_index.csv"), m("16_broadband_summary.csv")),
        "tableS4_mass_balance.tex": mass_balance(pd.read_csv(S.MASS_DIR / "mass_balance_by_bucket.csv"),
                                                 pd.read_csv(S.MASS_DIR / "mass_balance_pooled.csv")),
        "tableS5_skill_by_phase.tex": skill_by_phase(m("24_skill_by_phase.csv")),
        "tableS6_covariates.tex": covariates(m("26_covariates.csv")),
        "tableS7_ljung_box_by_band.tex": ljung_box_by_band(m("09_residual_diagnostics.csv")),
        "tableS8_without_top_band.tex": without_top_band(m("31_without_top_band.csv")),
        "tableS9_loudness_variants.tex": loudness_variants(pd.read_csv(S.LOUDNESS_DIR / "variants.csv")),
        "tableS10_loudness_residuals.tex": loudness_residuals(
            pd.read_csv(S.LOUDNESS_DIR / "residual_correlation_with_PE.csv")),
    }
    for name, text in out.items():
        (S.TABLES_DIR / name).write_text(text, encoding="utf-8")
    print(f"{len(out)} tables in {S.TABLES_DIR}")


if __name__ == "__main__":
    run()
