#!/usr/bin/env python3
"""
TTR Mutation Validation of the σ Framework
============================================
Validates the disorder propensity σ against Transthyretin (TTR) mutations
causing familial amyloid polyneuropathy (FAP) and cardiomyopathy (FAC).

σ formula: σ = exp(ΔG_unfold / (N·R·T))
  where N = 127 (TTR monomer), R = 8.314e-3 kJ/(mol·K), T = 310 K

For mutations: σ_mut = σ_wt × exp(ΔΔG_kJ / (N·R·T))
  where ΔΔG > 0 is destabilizing (standard convention).
"""

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── Constants ──────────────────────────────────────────────────────────────
N = 127                    # TTR monomer residues
R = 8.314e-3               # kJ/(mol·K)
T = 310                    # K (37 °C)
NRT = N * R * T            # denominator ~327.3 kJ/mol
DG_UNFOLD_WT = 25.0        # kJ/mol  (positive = stable fold)
                           # Source: Hammarström et al. 2002, 37°C, pH 7.0
DRIFT_RATE = 0.03          # σ increase per decade of age
BASELINE_AGE = 30          # age at which drift begins accumulating

# ── Wild-type σ ────────────────────────────────────────────────────────────
sigma_wt = np.exp(-DG_UNFOLD_WT / NRT)

print("=" * 72)
print("TTR VALIDATION OF THE σ FRAMEWORK")
print("=" * 72)
print(f"\nParameters: N={N}, R={R} kJ/(mol·K), T={T} K, NRT={NRT:.2f} kJ/mol")
print(f"ΔG_unfold(WT) = {DG_UNFOLD_WT} kJ/mol")
print(f"σ_wt = exp(-{DG_UNFOLD_WT}/{NRT:.2f}) = {sigma_wt:.6f}")
print(f"Drift rate = {DRIFT_RATE}/decade from age {BASELINE_AGE}\n")

# ── Curated mutation dataset ──────────────────────────────────────────────
ttr_mutations = [
    # Highly destabilizing – early onset
    {"mutation": "L55P",  "ddg_kcal": 3.6, "onset_age": 20,   "phenotype": "FAP"},
    {"mutation": "V30M",  "ddg_kcal": 2.1, "onset_age": 33,   "phenotype": "FAP"},
    {"mutation": "Y114C", "ddg_kcal": 2.8, "onset_age": 30,   "phenotype": "FAP"},
    {"mutation": "D18G",  "ddg_kcal": 4.0, "onset_age": 25,   "phenotype": "CNS"},
    {"mutation": "A25T",  "ddg_kcal": 3.2, "onset_age": 28,   "phenotype": "CNS"},
    # Moderately destabilizing – middle onset
    {"mutation": "V122I", "ddg_kcal": 1.5, "onset_age": 60,   "phenotype": "FAC"},
    {"mutation": "T60A",  "ddg_kcal": 1.8, "onset_age": 45,   "phenotype": "FAP+FAC"},
    {"mutation": "I84S",  "ddg_kcal": 2.0, "onset_age": 42,   "phenotype": "FAP+FAC"},
    {"mutation": "S77Y",  "ddg_kcal": 1.3, "onset_age": 55,   "phenotype": "FAP"},
    {"mutation": "E54K",  "ddg_kcal": 1.6, "onset_age": 50,   "phenotype": "FAP"},
    {"mutation": "L58H",  "ddg_kcal": 2.3, "onset_age": 40,   "phenotype": "FAP"},
    {"mutation": "F64L",  "ddg_kcal": 1.7, "onset_age": 48,   "phenotype": "FAP"},
    {"mutation": "I107V", "ddg_kcal": 1.4, "onset_age": 55,   "phenotype": "FAC"},
    {"mutation": "S50R",  "ddg_kcal": 1.9, "onset_age": 43,   "phenotype": "FAP"},
    {"mutation": "G47R",  "ddg_kcal": 1.5, "onset_age": 52,   "phenotype": "FAP"},
    # Mildly destabilizing / protective – late or no onset
    {"mutation": "T119M", "ddg_kcal": -0.8, "onset_age": None, "phenotype": "protective"},
    {"mutation": "R104H", "ddg_kcal": 0.8,  "onset_age": 65,  "phenotype": "FAC"},
    {"mutation": "A97S",  "ddg_kcal": 0.6,  "onset_age": 68,  "phenotype": "FAP"},
    {"mutation": "G6S",   "ddg_kcal": 0.3,  "onset_age": 72,  "phenotype": "FAC"},
    {"mutation": "V14A",  "ddg_kcal": 0.4,  "onset_age": 70,  "phenotype": "mixed"},
    # Additional well-characterised
    {"mutation": "Y78F",  "ddg_kcal": 1.1, "onset_age": 58,   "phenotype": "FAP"},
    {"mutation": "A36P",  "ddg_kcal": 2.5, "onset_age": 38,   "phenotype": "FAP"},
    {"mutation": "H88R",  "ddg_kcal": 0.9, "onset_age": 62,   "phenotype": "FAP"},
    {"mutation": "E89K",  "ddg_kcal": 1.0, "onset_age": 57,   "phenotype": "FAP"},
    {"mutation": "V71A",  "ddg_kcal": 1.2, "onset_age": 54,   "phenotype": "FAP"},
]

# ── Compute σ and predictions ─────────────────────────────────────────────
records = []
for m in ttr_mutations:
    ddg_kJ = m["ddg_kcal"] * 4.184
    sigma_mut = sigma_wt * np.exp(ddg_kJ / NRT)

    # Classification
    if sigma_mut < sigma_wt:
        cat = "protective"
    elif sigma_mut < 1.0:
        cat = "pathogenic (σ<1)"
    else:
        cat = "severely destabilised (σ≥1)"

    # σ-drift onset prediction
    if sigma_mut >= 1.0:
        predicted_onset = BASELINE_AGE  # already past threshold
    elif sigma_mut < sigma_wt:
        predicted_onset = None           # never reaches threshold
    else:
        predicted_onset = BASELINE_AGE + (1.0 - sigma_mut) / DRIFT_RATE * 10

    records.append({
        "mutation": m["mutation"],
        "ddg_kcal": m["ddg_kcal"],
        "ddg_kJ": round(ddg_kJ, 3),
        "sigma": round(sigma_mut, 6),
        "category": cat,
        "phenotype": m["phenotype"],
        "onset_actual": m["onset_age"],
        "onset_predicted": round(predicted_onset, 1) if predicted_onset is not None else None,
    })

df = pd.DataFrame(records)

# ── Display full table ─────────────────────────────────────────────────────
print("-" * 72)
print("FULL RESULTS TABLE")
print("-" * 72)
print(df.to_string(index=False))

# ── Correlation analysis (mutations with known onset) ─────────────────────
has_onset = df.dropna(subset=["onset_actual"]).copy()

r_sigma_onset, p_sigma = stats.pearsonr(has_onset["sigma"], has_onset["onset_actual"])
r_ddg_onset, p_ddg     = stats.pearsonr(has_onset["ddg_kcal"], has_onset["onset_actual"])
rho_sigma, p_rho_sigma  = stats.spearmanr(has_onset["sigma"], has_onset["onset_actual"])
rho_ddg, p_rho_ddg      = stats.spearmanr(has_onset["ddg_kcal"], has_onset["onset_actual"])

print("\n" + "=" * 72)
print("CORRELATION: σ / ΔΔG  vs  CLINICAL ONSET AGE")
print("=" * 72)
print(f"  Pearson  r(σ,    onset) = {r_sigma_onset:+.4f}  (p = {p_sigma:.2e})")
print(f"  Pearson  r(ΔΔG,  onset) = {r_ddg_onset:+.4f}  (p = {p_ddg:.2e})")
print(f"  Spearman ρ(σ,    onset) = {rho_sigma:+.4f}  (p = {p_rho_sigma:.2e})")
print(f"  Spearman ρ(ΔΔG,  onset) = {rho_ddg:+.4f}  (p = {p_rho_ddg:.2e})")
print(f"\n  n = {len(has_onset)} mutations with known onset age")

# ── σ-drift predicted vs actual onset ─────────────────────────────────────
has_both = has_onset.dropna(subset=["onset_predicted"]).copy()
r_pred, p_pred = stats.pearsonr(has_both["onset_predicted"], has_both["onset_actual"])
mae = np.mean(np.abs(has_both["onset_predicted"] - has_both["onset_actual"]))
rmse = np.sqrt(np.mean((has_both["onset_predicted"] - has_both["onset_actual"])**2))

print("\n" + "=" * 72)
print("σ-DRIFT MODEL: PREDICTED vs ACTUAL ONSET AGE")
print("=" * 72)
print(f"  Pearson r(predicted, actual) = {r_pred:+.4f}  (p = {p_pred:.2e})")
print(f"  MAE  = {mae:.1f} years")
print(f"  RMSE = {rmse:.1f} years")
print(f"  n = {len(has_both)} mutations with both predicted and actual onset")

# ── Bootstrap 95% CI for MAE ─────────────────────────────────────────
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

print(f"\n  Bootstrap 95% CI for MAE ({n_boot} resamples):")
print(f"    MAE = {mae:.1f} years  [95% CI: {mae_ci_lo:.1f} – {mae_ci_hi:.1f}]")

# ── Drift-rate sensitivity: Spearman ρ is rank-invariant ─────────────
# Because onset_predicted = baseline + (1 − σ_mut) / drift_rate × 10,
# changing drift_rate is a monotonic rescaling of (1 − σ_mut), so the
# rank ordering (and hence Spearman ρ) is invariant to drift_rate.
drift_rates = [0.01, 0.02, 0.03, 0.04, 0.05]
print(f"\n  Drift-rate sensitivity (Spearman ρ preserved across rates):")
for dr in drift_rates:
    preds = []
    actuals = []
    for m in ttr_mutations:
        if m["onset_age"] is None:
            continue
        ddg_kJ = m["ddg_kcal"] * 4.184
        sm = sigma_wt * np.exp(ddg_kJ / NRT)
        if sm >= 1.0:
            po = BASELINE_AGE
        elif sm < sigma_wt:
            continue
        else:
            po = BASELINE_AGE + (1.0 - sm) / dr * 10
        preds.append(po)
        actuals.append(m["onset_age"])
    rho_dr, _ = stats.spearmanr(preds, actuals)
    mae_dr = np.mean(np.abs(np.array(preds) - np.array(actuals)))
    print(f"    drift={dr:.2f}/decade  →  Spearman ρ = {rho_dr:+.4f},  MAE = {mae_dr:.1f} yr")

# ── Classification summary ────────────────────────────────────────────────
print("\n" + "=" * 72)
print("σ CLASSIFICATION SUMMARY")
print("=" * 72)
for cat in ["protective", "pathogenic (σ<1)", "severely destabilised (σ≥1)"]:
    subset = df[df["category"] == cat]
    if len(subset) > 0:
        print(f"\n  {cat}: {len(subset)} mutation(s)")
        for _, row in subset.iterrows():
            onset_str = f"onset {row['onset_actual']}" if row['onset_actual'] is not None else "no onset"
            print(f"    {row['mutation']:8s}  σ={row['sigma']:.4f}  ΔΔG={row['ddg_kcal']:+.1f} kcal/mol  {onset_str}  [{row['phenotype']}]")

# ── Why σ adds value beyond ΔΔG ───────────────────────────────────────────
print("\n" + "=" * 72)
print("WHY σ ADDS VALUE BEYOND ΔΔG ALONE")
print("=" * 72)
print("""
  1. UNIVERSAL THRESHOLD: σ = 1.0 marks the folding/unfolding boundary for
     ANY protein, regardless of chain length N or intrinsic stability.
     ΔΔG thresholds are protein-specific (e.g., 3 kcal/mol is severe for
     TTR but moderate for a larger, more stable protein).

  2. LENGTH NORMALISATION: σ divides by N, so disorder propensity is
     per-residue and comparable across TTR (127 aa) vs APP (770 aa)
     vs SOD1 (153 aa) vs any other amyloidogenic protein.

  3. AGE-DEPENDENT PREDICTION: The σ-drift model
       σ(age) = σ_mut + drift × (age − 30)/10
     provides a mechanistic basis for why the same mutation causes
     disease at different ages — proteostasis capacity declines with age.

  4. MONOTONIC EQUIVALENCE: For a single protein, σ is a monotonic
     transform of ΔΔG, so correlations are identical in magnitude.
     The advantage is interpretive, not statistical, within one protein.
""")

# ── Save CSV ──────────────────────────────────────────────────────────────
csv_path = "/home/ffai/code/papers/ttr_results.csv"
df.to_csv(csv_path, index=False)
print(f"Results saved to {csv_path}")

# ── Figure ────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(2, 2, figsize=(10, 9))
fig.suptitle("TTR Mutation Validation of the σ Framework", fontsize=14, fontweight="bold", y=0.98)

# Colour by phenotype
pheno_colours = {
    "FAP": "#2166ac", "FAC": "#b2182b", "CNS": "#762a83",
    "FAP+FAC": "#e08214", "mixed": "#666666", "protective": "#1b7837",
}

def pheno_color(p):
    return pheno_colours.get(p, "#999999")

# ── Panel A: σ vs onset age ───────────────────────────────────────────────
ax = axes[0, 0]
colours_a = [pheno_color(p) for p in has_onset["phenotype"]]
ax.scatter(has_onset["sigma"], has_onset["onset_actual"], c=colours_a, s=50, edgecolors="k", linewidths=0.5, zorder=3)
# regression line
slope, intercept, _, _, _ = stats.linregress(has_onset["sigma"], has_onset["onset_actual"])
x_fit = np.linspace(has_onset["sigma"].min() - 0.005, has_onset["sigma"].max() + 0.005, 100)
ax.plot(x_fit, slope * x_fit + intercept, "k--", linewidth=1, alpha=0.7)
ax.set_xlabel("σ (disorder propensity)", fontsize=10)
ax.set_ylabel("Clinical onset age (years)", fontsize=10)
ax.set_title(f"A. σ vs onset age\nr = {r_sigma_onset:.3f}", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel B: ΔΔG vs onset age ────────────────────────────────────────────
ax = axes[0, 1]
colours_b = [pheno_color(p) for p in has_onset["phenotype"]]
ax.scatter(has_onset["ddg_kcal"], has_onset["onset_actual"], c=colours_b, s=50, edgecolors="k", linewidths=0.5, zorder=3)
slope2, intercept2, _, _, _ = stats.linregress(has_onset["ddg_kcal"], has_onset["onset_actual"])
x_fit2 = np.linspace(has_onset["ddg_kcal"].min() - 0.2, has_onset["ddg_kcal"].max() + 0.2, 100)
ax.plot(x_fit2, slope2 * x_fit2 + intercept2, "k--", linewidth=1, alpha=0.7)
ax.set_xlabel("ΔΔG (kcal/mol, + = destabilising)", fontsize=10)
ax.set_ylabel("Clinical onset age (years)", fontsize=10)
ax.set_title(f"B. ΔΔG vs onset age\nr = {r_ddg_onset:.3f}", fontsize=11)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel C: predicted vs actual onset ────────────────────────────────────
ax = axes[1, 0]
colours_c = [pheno_color(p) for p in has_both["phenotype"]]
ax.scatter(has_both["onset_actual"], has_both["onset_predicted"], c=colours_c, s=50, edgecolors="k", linewidths=0.5, zorder=3)
lims = [15, 80]
ax.plot(lims, lims, "k-", linewidth=1, alpha=0.4, label="perfect prediction")
ax.set_xlabel("Actual onset age (years)", fontsize=10)
ax.set_ylabel("Predicted onset age (years)", fontsize=10)
ax.set_title(f"C. σ-drift onset prediction\nr = {r_pred:.3f}, MAE = {mae:.1f} yr [95% CI: {mae_ci_lo:.1f}–{mae_ci_hi:.1f}]", fontsize=11)
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.legend(loc="lower right", frameon=False, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# ── Panel D: σ by phenotype (strip plot) ──────────────────────────────────
ax = axes[1, 1]
pheno_order = ["FAP", "FAC", "FAP+FAC", "CNS", "mixed", "protective"]
pheno_present = [p for p in pheno_order if p in df["phenotype"].values]
positions = []
sigmas_by_pheno = []
colors_d = []
for i, ph in enumerate(pheno_present):
    vals = df[df["phenotype"] == ph]["sigma"].values
    sigmas_by_pheno.append(vals)
    positions.append(i)
    colors_d.append(pheno_colours.get(ph, "#999999"))

bp = ax.boxplot(sigmas_by_pheno, positions=positions, widths=0.4,
                patch_artist=True, showfliers=False,
                medianprops=dict(color="black", linewidth=1.5))
for patch, col in zip(bp["boxes"], colors_d):
    patch.set_facecolor(col)
    patch.set_alpha(0.35)

# overlay individual points with jitter
rng = np.random.default_rng(42)
for i, (ph, vals) in enumerate(zip(pheno_present, sigmas_by_pheno)):
    jitter = rng.uniform(-0.12, 0.12, size=len(vals))
    ax.scatter([i] * len(vals) + jitter, vals, c=pheno_colours.get(ph, "#999"),
               s=40, edgecolors="k", linewidths=0.5, zorder=3)

ax.axhline(y=sigma_wt, color="green", linestyle=":", linewidth=1, label=f"σ_wt = {sigma_wt:.3f}")
ax.axhline(y=1.0, color="red", linestyle=":", linewidth=1, label="σ = 1.0 (threshold)")
ax.set_xticks(range(len(pheno_present)))
ax.set_xticklabels(pheno_present, fontsize=9)
ax.set_ylabel("σ (disorder propensity)", fontsize=10)
ax.set_title("D. σ by phenotype category", fontsize=11)
ax.legend(loc="upper left", frameon=False, fontsize=8)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

plt.tight_layout(rect=[0, 0, 1, 0.95])
fig_path = "/home/ffai/code/papers/paper5_submission/fig_ttr.pdf"
fig.savefig(fig_path, dpi=300, bbox_inches="tight")
print(f"Figure saved to {fig_path}")

print("\n" + "=" * 72)
print("VALIDATION COMPLETE")
print("=" * 72)
