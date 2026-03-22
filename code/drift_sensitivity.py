#!/usr/bin/env python3
"""
Sensitivity analysis of the sigma-drift model across different drift rates.

The sigma-drift model predicts disease onset when sigma(age) >= 1.0, where:
    sigma(age) = sigma_base + drift_rate * (age - 30) / 10

Solving for onset age:
    onset_age = 30 + (1.0 - sigma_base) / drift_rate * 10

Key result: Because onset_age is a monotonically decreasing LINEAR function of
sigma_base for any positive drift_rate, the RANKING of mutations by predicted
onset age is perfectly preserved across all drift rates (Spearman rho = 1.0
by construction). The drift rate affects WHEN disease occurs, but not WHO
gets sick first.
"""

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import os

# ---------------------------------------------------------------------------
# Mutation data
# ---------------------------------------------------------------------------
ttr_mutations = [
    {"name": "L55P (TTR)",  "sigma": 0.975},
    {"name": "D18G (TTR)",  "sigma": 0.970},
    {"name": "A25T (TTR)",  "sigma": 0.968},
    {"name": "Y114C (TTR)", "sigma": 0.960},
    {"name": "V30M (TTR)",  "sigma": 0.955},
    {"name": "A36P (TTR)",  "sigma": 0.945},
    {"name": "L58H (TTR)",  "sigma": 0.940},
    {"name": "I84S (TTR)",  "sigma": 0.935},
    {"name": "T60A (TTR)",  "sigma": 0.930},
    {"name": "S50R (TTR)",  "sigma": 0.925},
    {"name": "E54K (TTR)",  "sigma": 0.920},
    {"name": "F64L (TTR)",  "sigma": 0.915},
    {"name": "G47R (TTR)",  "sigma": 0.910},
    {"name": "V122I (TTR)", "sigma": 0.900},
    {"name": "Y78F (TTR)",  "sigma": 0.895},
    {"name": "H88R (TTR)",  "sigma": 0.890},
    {"name": "E89K (TTR)",  "sigma": 0.885},
    {"name": "V71A (TTR)",  "sigma": 0.880},
    {"name": "R104H (TTR)", "sigma": 0.870},
    {"name": "A97S (TTR)",  "sigma": 0.860},
    {"name": "G6S (TTR)",   "sigma": 0.850},
    {"name": "V14A (TTR)",  "sigma": 0.845},
]

ab_mutations = [
    {"name": "Swedish (K670N/M671L)", "sigma": 0.965},
    {"name": "Arctic (E693G)",        "sigma": 0.955},
    {"name": "Italian (E693K)",       "sigma": 0.950},
    {"name": "Iowa (D694N)",          "sigma": 0.948},
    {"name": "Dutch (E693Q)",         "sigma": 0.945},
    {"name": "Flemish (A692G)",       "sigma": 0.940},
    {"name": "Piedmont (L705V)",      "sigma": 0.930},
    {"name": "Osaka (E693\u0394)",    "sigma": 0.960},
    {"name": "A673V",                 "sigma": 0.935},
    {"name": "H677R (English)",       "sigma": 0.925},
]

# ---------------------------------------------------------------------------
# Drift scenarios
# ---------------------------------------------------------------------------
drift_rates = {
    "slow":   0.02,   # per decade
    "medium": 0.03,   # paper default
    "fast":   0.05,   # per decade
}

def onset_age(sigma_base, drift_rate):
    """Predicted onset age given sigma_base and drift_rate (per decade)."""
    return 30.0 + (1.0 - sigma_base) / drift_rate * 10.0

# ---------------------------------------------------------------------------
# Compute predictions for all mutations x all drift rates
# ---------------------------------------------------------------------------
all_mutations = ttr_mutations + ab_mutations
records = []
for m in all_mutations:
    row = {
        "mutation": m["name"],
        "sigma": m["sigma"],
        "family": "TTR" if "(TTR)" in m["name"] else "A\u03b2",
    }
    for label, dr in drift_rates.items():
        row[f"onset_{label}"] = onset_age(m["sigma"], dr)
    row["range"] = row["onset_slow"] - row["onset_fast"]
    records.append(row)

df = pd.DataFrame(records)

# ---------------------------------------------------------------------------
# Print results table
# ---------------------------------------------------------------------------
print("=" * 100)
print("SIGMA-DRIFT SENSITIVITY ANALYSIS")
print("=" * 100)
print(f"\nDrift rates tested: slow={drift_rates['slow']}/decade, "
      f"medium={drift_rates['medium']}/decade, fast={drift_rates['fast']}/decade")
print(f"\nOnset model: onset_age = 30 + (1.0 - sigma) / drift_rate x 10\n")

header = f"{'Mutation':<28s} {'sigma':>6s} {'Slow':>8s} {'Medium':>8s} {'Fast':>8s} {'Range':>8s}"
print(header)
print("-" * len(header))

for fam in ["TTR", "A\u03b2"]:
    sub = df[df["family"] == fam]
    if fam == "A\u03b2":
        print()
        print(f"--- A\u03b2 mutations ---")
    for _, r in sub.iterrows():
        print(f"{r['mutation']:<28s} {r['sigma']:>6.3f} "
              f"{r['onset_slow']:>8.1f} {r['onset_medium']:>8.1f} "
              f"{r['onset_fast']:>8.1f} {r['range']:>8.1f}")

# ---------------------------------------------------------------------------
# Rank correlations
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("SPEARMAN RANK CORRELATIONS BETWEEN DRIFT SCENARIOS")
print("=" * 100)

pairs = [("slow", "medium"), ("slow", "fast"), ("medium", "fast")]
for a, b in pairs:
    rho, pval = spearmanr(df[f"onset_{a}"], df[f"onset_{b}"])
    print(f"  {a:>6s} vs {b:<6s}:  rho = {rho:.4f}  (p = {pval:.2e})")

print("\n  NOTE: Spearman rho = 1.0000 by construction. The onset age formula")
print("  onset = 30 + (1 - sigma) / drift_rate * 10  is a monotonically")
print("  decreasing LINEAR function of sigma for any positive drift_rate.")
print("  A linear rescaling preserves all ranks. Therefore, the ranking of")
print("  mutations by predicted disease onset is IDENTICAL across all drift")
print("  rates. The drift rate affects WHEN disease occurs, not WHO gets sick first.")

# ---------------------------------------------------------------------------
# Clinical sensitivity summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("CLINICAL SENSITIVITY: HOW ONSET AGES SHIFT WITH DRIFT RATE")
print("=" * 100)

example_sigmas = [0.975, 0.950, 0.925, 0.900, 0.870, 0.850]
print(f"\n{'sigma':>8s} {'Slow (0.02)':>12s} {'Medium (0.03)':>14s} {'Fast (0.05)':>12s} {'Range':>8s}")
print("-" * 58)
for s in example_sigmas:
    slow = onset_age(s, 0.02)
    med  = onset_age(s, 0.03)
    fast = onset_age(s, 0.05)
    print(f"{s:>8.3f} {slow:>12.1f} {med:>14.1f} {fast:>12.1f} {slow - fast:>8.1f}")

print("\n  The medium drift rate (0.03/decade) typically produces onset ages")
print("  in the 30-80 year range for high-sigma mutations, consistent with")
print("  clinical observations for familial amyloidoses.")

# ---------------------------------------------------------------------------
# Summary statistics
# ---------------------------------------------------------------------------
print("\n" + "=" * 100)
print("ONSET AGE RANGE (max-min across drift rates) PER MUTATION")
print("=" * 100)
print(f"\n  Mean range across mutations:   {df['range'].mean():.1f} years")
print(f"  Min range (highest sigma):     {df['range'].min():.1f} years  "
      f"({df.loc[df['range'].idxmin(), 'mutation']})")
print(f"  Max range (lowest sigma):      {df['range'].max():.1f} years  "
      f"({df.loc[df['range'].idxmax(), 'mutation']})")
print(f"\n  Higher-sigma mutations show SMALLER absolute ranges because")
print(f"  (1 - sigma) is small, so the scaling effect of drift_rate is modest.")
print(f"  Lower-sigma mutations have larger (1 - sigma), amplifying the")
print(f"  drift-rate sensitivity in absolute years.")

# ---------------------------------------------------------------------------
# Save CSV
# ---------------------------------------------------------------------------
csv_path = "/home/ffai/code/papers/drift_sensitivity_results.csv"
df.to_csv(csv_path, index=False, float_format="%.2f")
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# Figure
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5.5))

# Colours
colours = {"slow": "#2166ac", "medium": "#b2182b", "fast": "#d6604d"}
labels  = {"slow": "Slow (0.02/dec)", "medium": "Medium (0.03/dec)", "fast": "Fast (0.05/dec)"}
markers = {"TTR": "o", "A\u03b2": "s"}

# --- Left panel: onset age vs sigma for each drift rate ---
sigma_cont = np.linspace(0.84, 0.98, 200)
for key in ["slow", "medium", "fast"]:
    dr = drift_rates[key]
    ax1.plot(sigma_cont, onset_age(sigma_cont, dr),
             color=colours[key], lw=2, label=labels[key], zorder=2)

# Scatter points (medium only to avoid clutter)
for fam, marker in markers.items():
    sub = df[df["family"] == fam]
    ax1.scatter(sub["sigma"], sub["onset_medium"], marker=marker,
                edgecolors=colours["medium"], facecolors="white",
                s=40, linewidths=1.2, zorder=3,
                label=f"{fam} mutations" if fam == "TTR" else f"A\u03b2 mutations")

ax1.set_xlabel(r"$\sigma_{\mathrm{base}}$", fontsize=12)
ax1.set_ylabel("Predicted onset age (years)", fontsize=12)
ax1.set_title("A. Onset age vs. $\\sigma$ across drift rates", fontsize=12, fontweight="bold", loc="left")
ax1.legend(fontsize=8, frameon=False, loc="upper right")
ax1.set_xlim(0.84, 0.98)
ax1.set_ylim(25, 110)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.tick_params(labelsize=10)

# --- Right panel: fan / envelope plot ---
df_sorted = df.sort_values("sigma", ascending=False).reset_index(drop=True)
x_pos = np.arange(len(df_sorted))

# Envelope
ax2.fill_between(x_pos,
                  df_sorted["onset_fast"].values,
                  df_sorted["onset_slow"].values,
                  color="#d1e5f0", alpha=0.7, label="Range (slow\u2013fast)", zorder=1)

# Medium line
ax2.plot(x_pos, df_sorted["onset_medium"].values,
         color=colours["medium"], lw=2, marker=".", markersize=5,
         label="Medium (0.03/dec)", zorder=3)

# Slow and fast edges
ax2.plot(x_pos, df_sorted["onset_slow"].values,
         color=colours["slow"], lw=1, ls="--", label="Slow (0.02/dec)", zorder=2)
ax2.plot(x_pos, df_sorted["onset_fast"].values,
         color=colours["fast"], lw=1, ls="--", label="Fast (0.05/dec)", zorder=2)

ax2.set_xticks(x_pos)
ax2.set_xticklabels(df_sorted["mutation"].values, rotation=70, ha="right", fontsize=6.5)
ax2.set_ylabel("Predicted onset age (years)", fontsize=12)
ax2.set_title("B. Onset envelope across drift rates", fontsize=12, fontweight="bold", loc="left")
ax2.legend(fontsize=8, frameon=False, loc="upper left")
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.tick_params(labelsize=10)
ax2.set_xlim(-0.5, len(df_sorted) - 0.5)

fig.tight_layout()

fig_path = "/home/ffai/code/papers/paper5_submission/fig_drift.pdf"
fig.savefig(fig_path, bbox_inches="tight", dpi=300)
plt.close(fig)
print(f"Figure saved to {fig_path}")

print("\n" + "=" * 100)
print("CONCLUSION")
print("=" * 100)
print("""
The sigma-drift model's mutation RANKING is perfectly robust to drift-rate
uncertainty (Spearman rho = 1.0 across all pairwise scenario comparisons).
This is guaranteed by construction: onset_age is a monotonically decreasing
linear function of sigma_base for any positive drift_rate, so ranks cannot change.

What DOES change with the drift rate is the absolute predicted onset age.
The medium rate (0.03/decade) produces clinically plausible onset ages for
known pathogenic TTR and Abeta mutations, but the framework's value lies
primarily in its ability to rank mutation severity — a property that is
entirely independent of the drift-rate parameter.

Key takeaway for reviewers: the drift rate is a calibration parameter that
affects the timescale but not the ordering. Any positive drift rate yields
the same clinical ranking of mutations.
""")
