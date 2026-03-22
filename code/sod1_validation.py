#!/usr/bin/env python3
"""
SOD1 Mutation Validation of the sigma Framework
=================================================
Validates the disorder propensity sigma against SOD1 mutations
causing familial ALS (amyotrophic lateral sclerosis).

sigma formula: sigma = exp(-DG_unfold / (N * R * T))
For mutations: sigma_mut = sigma_wt * exp(DDG / (N * R * T))
  where DDG > 0 is destabilising (standard convention).

SOD1: N=153 residues (monomer), homodimer, Cu/Zn metalloenzyme.
Disease mechanism is primarily stability-driven: mutations destabilise
the SOD1 dimer, leading to misfolding and aggregation.

IMPORTANT CAVEATS:
- SOD1/ALS is more complex than TTR amyloidosis
- Some mutations act via gain-of-toxic-function, not just loss of stability
- Metal-binding mutations (H46R, H48Q) may have different mechanisms
- Onset age in ALS has HIGH variability (same mutation, different patients)
- The DDG values used here are approximate literature values
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# -- Constants ----------------------------------------------------------------
N = 153                      # SOD1 monomer residues
R_KCAL = 0.001987            # kcal/(mol*K)
T = 310                      # K (37 C)
NRT = N * R_KCAL * T         # denominator in kcal/mol
DG_UNFOLD_WT = 9.0           # kcal/mol (monomer stability, literature)
DRIFT_RATE = 0.03            # sigma increase per decade of age
BASELINE_AGE = 30            # age at which drift begins accumulating

# -- Wild-type sigma ----------------------------------------------------------
sigma_wt = np.exp(-DG_UNFOLD_WT / NRT)

print("=" * 78)
print("SOD1 / FAMILIAL ALS VALIDATION OF THE sigma FRAMEWORK")
print("=" * 78)
print(f"\nParameters: N={N}, R={R_KCAL} kcal/(mol*K), T={T} K")
print(f"NRT = {NRT:.4f} kcal/mol")
print(f"DG_unfold(WT) = {DG_UNFOLD_WT} kcal/mol")
print(f"sigma_wt = exp(-{DG_UNFOLD_WT}/{NRT:.4f}) = {sigma_wt:.6f}")
print(f"Drift rate = {DRIFT_RATE}/decade from age {BASELINE_AGE}\n")

# -- Curated mutation dataset -------------------------------------------------
# DDG values: approximate from literature (Rodriguez et al. 2005, Bystrom et al.
# 2010, Lindberg et al. 2005 PNAS, and review compilations).
# Onset ages: median reported onset from clinical series.
#
# HONEST NOTE: These DDG values are approximate. Different studies report
# different values depending on conditions (apo vs holo, monomer vs dimer,
# pH, temperature). Onset ages are medians with large inter-patient variance.

sod1_mutations = [
    # mutation, DDG (kcal/mol), onset_age (yr), notes
    {"mutation": "A4V",   "ddg_kcal": 2.5, "onset_age": 47,
     "notes": "Most common NA. Aggressive. Stability-driven."},
    {"mutation": "G93A",  "ddg_kcal": 1.0, "onset_age": 45,
     "notes": "Most studied (mouse model). Moderate destabilisation."},
    {"mutation": "D90A",  "ddg_kcal": 0.5, "onset_age": 44,
     "notes": "Recessive in Scandinavia. Mildly destabilising. Slow progression."},
    {"mutation": "H46R",  "ddg_kcal": 3.0, "onset_age": 55,
     "notes": "Metal-binding site. Slow progression. May not be purely stability-driven."},
    {"mutation": "G85R",  "ddg_kcal": 3.5, "onset_age": 45,
     "notes": "Highly destabilising. Metal-deficient."},
    {"mutation": "I113T", "ddg_kcal": 1.5, "onset_age": 58,
     "notes": "UK common. Variable penetrance."},
    {"mutation": "L38V",  "ddg_kcal": 1.2, "onset_age": 45,
     "notes": "Moderately destabilising."},
    {"mutation": "E100G", "ddg_kcal": 2.0, "onset_age": 45,
     "notes": "Moderately destabilising."},
    {"mutation": "G37R",  "ddg_kcal": 1.8, "onset_age": 35,
     "notes": "Early onset. Upper motor neuron predominant."},
    {"mutation": "A89V",  "ddg_kcal": 1.5, "onset_age": 48,
     "notes": "Moderately destabilising."},
]

# -- Compute sigma and predictions -------------------------------------------
records = []
for m in sod1_mutations:
    ddg = m["ddg_kcal"]
    sigma_mut = sigma_wt * np.exp(ddg / NRT)

    # Classification
    if sigma_mut < sigma_wt:
        cat = "protective"
    elif sigma_mut < 1.0:
        cat = "pathogenic (sigma<1)"
    else:
        cat = "severely destabilised (sigma>=1)"

    # sigma-drift onset prediction
    if sigma_mut >= 1.0:
        predicted_onset = BASELINE_AGE  # already past threshold
    elif sigma_mut < sigma_wt:
        predicted_onset = None
    else:
        predicted_onset = BASELINE_AGE + (1.0 - sigma_mut) / DRIFT_RATE * 10

    records.append({
        "mutation": m["mutation"],
        "ddg_kcal": ddg,
        "sigma": round(sigma_mut, 6),
        "sigma_ratio": round(sigma_mut / sigma_wt, 4),
        "category": cat,
        "onset_actual": m["onset_age"],
        "onset_predicted": round(predicted_onset, 1) if predicted_onset is not None else None,
        "notes": m["notes"],
    })

df = pd.DataFrame(records)

# -- Display full table -------------------------------------------------------
print("-" * 78)
print("FULL RESULTS TABLE")
print("-" * 78)
# Print a clean table
print(f"{'Mutation':>8s}  {'DDG':>6s}  {'sigma':>10s}  {'sigma/wt':>8s}  "
      f"{'Category':>24s}  {'Onset':>5s}  {'Pred':>5s}")
print("-" * 78)
for _, row in df.iterrows():
    pred_str = f"{row['onset_predicted']:.0f}" if row['onset_predicted'] is not None else "  -"
    print(f"{row['mutation']:>8s}  {row['ddg_kcal']:>+5.1f}  {row['sigma']:>10.6f}  "
          f"{row['sigma_ratio']:>8.4f}  {row['category']:>24s}  "
          f"{row['onset_actual']:>5.0f}  {pred_str:>5s}")

# -- ClinVar-style check: all pathogenic mutations should have sigma > sigma_wt
print("\n" + "=" * 78)
print("CLINVAR CLASSIFICATION CHECK")
print("=" * 78)
all_above_wt = all(r["sigma"] > sigma_wt for r in records)
print(f"  All pathogenic mutations have sigma > sigma_wt? {all_above_wt}")
for r in records:
    status = "PASS" if r["sigma"] > sigma_wt else "FAIL"
    print(f"    {r['mutation']:>6s}: sigma={r['sigma']:.6f} vs sigma_wt={sigma_wt:.6f}  [{status}]")

# -- Correlation analysis -----------------------------------------------------
has_onset = df.dropna(subset=["onset_actual"]).copy()

r_sigma_onset, p_sigma   = stats.pearsonr(has_onset["sigma"], has_onset["onset_actual"])
r_ddg_onset, p_ddg       = stats.pearsonr(has_onset["ddg_kcal"], has_onset["onset_actual"])
rho_sigma, p_rho_sigma   = stats.spearmanr(has_onset["sigma"], has_onset["onset_actual"])
rho_ddg, p_rho_ddg       = stats.spearmanr(has_onset["ddg_kcal"], has_onset["onset_actual"])

print("\n" + "=" * 78)
print("CORRELATION: sigma / DDG  vs  CLINICAL ONSET AGE")
print("=" * 78)
print(f"  Pearson  r(sigma, onset) = {r_sigma_onset:+.4f}  (p = {p_sigma:.4f})")
print(f"  Pearson  r(DDG,   onset) = {r_ddg_onset:+.4f}  (p = {p_ddg:.4f})")
print(f"  Spearman rho(sigma, onset) = {rho_sigma:+.4f}  (p = {p_rho_sigma:.4f})")
print(f"  Spearman rho(DDG,   onset) = {rho_ddg:+.4f}  (p = {p_rho_ddg:.4f})")
print(f"\n  n = {len(has_onset)} mutations with known onset age")

# -- KEY INTERPRETATION -------------------------------------------------------
print("\n" + "=" * 78)
print("KEY RESULT INTERPRETATION")
print("=" * 78)
if abs(rho_sigma) > 0.7:
    print(f"  Spearman rho = {rho_sigma:+.4f} indicates STRONG correlation.")
    print("  SOD1 data supports the sigma framework for stability-driven mutations.")
elif abs(rho_sigma) > 0.4:
    print(f"  Spearman rho = {rho_sigma:+.4f} indicates MODERATE correlation.")
    print("  Partial support -- some mutations may not be purely stability-driven.")
else:
    print(f"  Spearman rho = {rho_sigma:+.4f} indicates WEAK or NO correlation.")
    print("  SOD1/ALS onset age is NOT well predicted by thermodynamic stability alone.")
    print("  This is consistent with gain-of-toxic-function and metal-binding mechanisms")
    print("  playing important roles alongside or instead of stability loss.")

# Honest assessment of which mutations fit vs don't
print("\n  MUTATION-BY-MUTATION ASSESSMENT:")
for _, row in df.iterrows():
    if row["onset_predicted"] is not None:
        residual = row["onset_actual"] - row["onset_predicted"]
        fit = "GOOD" if abs(residual) < 8 else ("POOR" if abs(residual) > 15 else "FAIR")
        print(f"    {row['mutation']:>6s}: actual={row['onset_actual']:.0f}, "
              f"predicted={row['onset_predicted']:.0f}, "
              f"residual={residual:+.0f} yr  [{fit}]")
    else:
        print(f"    {row['mutation']:>6s}: actual={row['onset_actual']:.0f}, "
              f"predicted=N/A")

# -- sigma-drift model accuracy -----------------------------------------------
has_both = has_onset.dropna(subset=["onset_predicted"]).copy()
if len(has_both) > 2:
    r_pred, p_pred = stats.pearsonr(has_both["onset_predicted"], has_both["onset_actual"])
    mae = np.mean(np.abs(has_both["onset_predicted"] - has_both["onset_actual"]))
    rmse = np.sqrt(np.mean((has_both["onset_predicted"] - has_both["onset_actual"])**2))

    print(f"\n" + "=" * 78)
    print("sigma-DRIFT MODEL: PREDICTED vs ACTUAL ONSET AGE")
    print("=" * 78)
    print(f"  Pearson r(predicted, actual) = {r_pred:+.4f}  (p = {p_pred:.4f})")
    print(f"  MAE  = {mae:.1f} years")
    print(f"  RMSE = {rmse:.1f} years")
    print(f"  n = {len(has_both)} mutations")

    # Bootstrap 95% CI for MAE
    n_boot = 10_000
    rng_boot = np.random.default_rng(2024)
    actual_arr = has_both["onset_actual"].values
    predicted_arr = has_both["onset_predicted"].values
    n_mut = len(actual_arr)

    boot_maes = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng_boot.integers(0, n_mut, size=n_mut)
        boot_maes[i] = np.mean(np.abs(predicted_arr[idx] - actual_arr[idx]))

    mae_ci_lo = np.percentile(boot_maes, 2.5)
    mae_ci_hi = np.percentile(boot_maes, 97.5)
    print(f"  Bootstrap 95% CI for MAE: [{mae_ci_lo:.1f} - {mae_ci_hi:.1f}] years")

    # Drift-rate sensitivity
    print(f"\n  Drift-rate sensitivity:")
    drift_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
    for dr in drift_rates:
        preds = []
        actuals = []
        for m in sod1_mutations:
            ddg = m["ddg_kcal"]
            sm = sigma_wt * np.exp(ddg / NRT)
            if sm >= 1.0:
                po = BASELINE_AGE
            elif sm < sigma_wt:
                continue
            else:
                po = BASELINE_AGE + (1.0 - sm) / dr * 10
            preds.append(po)
            actuals.append(m["onset_age"])
        if len(preds) > 2:
            rho_dr, _ = stats.spearmanr(preds, actuals)
            mae_dr = np.mean(np.abs(np.array(preds) - np.array(actuals)))
            print(f"    drift={dr:.2f}/decade  ->  Spearman rho = {rho_dr:+.4f},  MAE = {mae_dr:.1f} yr")

# -- Comparison with TTR ------------------------------------------------------
print("\n" + "=" * 78)
print("COMPARISON WITH TTR VALIDATION")
print("=" * 78)
print(f"  TTR:  Spearman rho ~ -0.977 (N=127, 24 mutations, pure amyloid disease)")
print(f"  SOD1: Spearman rho = {rho_sigma:+.4f} (N=153, {len(has_onset)} mutations)")
print()
if abs(rho_sigma) < 0.5:
    print("  HONEST ASSESSMENT: SOD1/ALS does NOT replicate the strong correlation")
    print("  seen in TTR amyloidosis. This is expected because:")
    print("    1. ALS involves gain-of-toxic-function mechanisms beyond misfolding")
    print("    2. Metal-binding mutations (H46R) alter function independently of stability")
    print("    3. Clinical onset in ALS has very high inter-patient variability")
    print("    4. Some mutations (e.g., G93A) retain near-normal stability but are pathogenic")
    print()
    print("  CONCLUSION: The sigma framework captures the stability component of")
    print("  SOD1 pathogenicity but cannot fully explain ALS onset because ALS is")
    print("  NOT a pure stability-loss disease like TTR amyloidosis.")
elif abs(rho_sigma) > 0.7:
    print("  RESULT: SOD1 shows strong correlation, supporting the sigma framework")
    print("  even in a more complex disease context.")
else:
    print("  RESULT: Moderate correlation. The stability component is real but")
    print("  additional mechanisms complicate the picture for SOD1/ALS.")

# -- Outlier analysis ---------------------------------------------------------
print("\n" + "=" * 78)
print("OUTLIER ANALYSIS: MUTATIONS THAT DON'T FIT THE STABILITY MODEL")
print("=" * 78)
if len(has_both) > 0:
    residuals = has_both["onset_actual"].values - has_both["onset_predicted"].values
    for idx, (_, row) in enumerate(has_both.iterrows()):
        res = residuals[idx]
        if abs(res) > 10:
            direction = "later than predicted" if res > 0 else "earlier than predicted"
            print(f"  {row['mutation']}: onset {direction} by {abs(res):.0f} years")
            # Find the matching notes
            for m in sod1_mutations:
                if m["mutation"] == row["mutation"]:
                    print(f"    Notes: {m['notes']}")
                    break

# -- Save CSV -----------------------------------------------------------------
csv_path = "/home/ffai/code/papers/sod1_results.csv"
df_out = df.drop(columns=["notes"])
df_out.to_csv(csv_path, index=False)
print(f"\nResults saved to {csv_path}")

# -- Figure -------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 5))
fig.suptitle("SOD1/ALS Validation of the sigma Framework",
             fontsize=13, fontweight="bold", y=1.02)

# Panel A: sigma vs onset age
ax = axes[0]
ax.scatter(has_onset["sigma"], has_onset["onset_actual"],
           c="#2166ac", s=60, edgecolors="k", linewidths=0.5, zorder=3)

# Label each point
for _, row in has_onset.iterrows():
    ax.annotate(row["mutation"], (row["sigma"], row["onset_actual"]),
                fontsize=7, ha="left", va="bottom",
                xytext=(4, 4), textcoords="offset points")

# Regression line
slope, intercept, _, _, _ = stats.linregress(has_onset["sigma"], has_onset["onset_actual"])
x_fit = np.linspace(has_onset["sigma"].min() - 0.002, has_onset["sigma"].max() + 0.002, 100)
ax.plot(x_fit, slope * x_fit + intercept, "k--", linewidth=1, alpha=0.6)

# Reference lines
ax.axvline(x=sigma_wt, color="green", linestyle=":", linewidth=1,
           alpha=0.7, label=f"sigma_wt = {sigma_wt:.4f}")

ax.set_xlabel("sigma (disorder propensity)", fontsize=10)
ax.set_ylabel("Clinical onset age (years)", fontsize=10)
ax.set_title(f"A. sigma vs onset age\n"
             f"Spearman rho = {rho_sigma:.3f} (p = {p_rho_sigma:.3f})",
             fontsize=11)
ax.legend(loc="best", frameon=False, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Panel B: predicted vs actual onset
ax = axes[1]
if len(has_both) > 2:
    ax.scatter(has_both["onset_actual"], has_both["onset_predicted"],
               c="#b2182b", s=60, edgecolors="k", linewidths=0.5, zorder=3)

    for _, row in has_both.iterrows():
        ax.annotate(row["mutation"], (row["onset_actual"], row["onset_predicted"]),
                    fontsize=7, ha="left", va="bottom",
                    xytext=(4, 4), textcoords="offset points")

    lims = [25, 70]
    ax.plot(lims, lims, "k-", linewidth=1, alpha=0.4, label="perfect prediction")
    ax.set_xlim(lims)
    ax.set_ylim([min(has_both["onset_predicted"].min() - 5, 25),
                 max(has_both["onset_predicted"].max() + 5, 70)])
    ax.set_xlabel("Actual onset age (years)", fontsize=10)
    ax.set_ylabel("Predicted onset age (years)", fontsize=10)
    ax.set_title(f"B. sigma-drift onset prediction\n"
                 f"r = {r_pred:.3f}, MAE = {mae:.1f} yr",
                 fontsize=11)
    ax.legend(loc="best", frameon=False, fontsize=8)
else:
    ax.text(0.5, 0.5, "Insufficient data\nfor prediction plot",
            transform=ax.transAxes, ha="center", va="center", fontsize=12)
    ax.set_title("B. sigma-drift onset prediction", fontsize=11)

ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout()
fig_path = "/home/ffai/code/papers/paper5_submission/fig_sod1.pdf"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to {fig_path}")

print("\n" + "=" * 78)
print("VALIDATION COMPLETE")
print("=" * 78)
