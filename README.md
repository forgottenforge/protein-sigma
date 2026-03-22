# ForgottenForge - protein-sigma
![Python](https://img.shields.io/badge/Python-3.11-blue?logo=python)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.en.html)
[![License: Commercial](https://img.shields.io/badge/License-Commercial-orange)](LICENSE-COMMERCIAL.txt)

Code and data for:

**A contraction index for protein stability predicts disease onset in stability-driven amyloidoses and identifies its own boundary of validity**

ForgottenForge (2026)

---

## What this is

The stability index $\sigma = D \cdot \gamma$ formalizes protein folding as a contraction problem. $D$ is the fraction of productive conformational moves; $\gamma$ is the step-wise contraction ratio. $\sigma < 1$: the protein folds. $\sigma > 1$: the native basin loses contractivity. $\sigma = 1$: the critical point.

We test a specific hypothesis: $\sigma$ predicts disease onset **when disease is predominantly driven by thermodynamic destabilization of a globular fold**, and fails when other mechanisms dominate.

### Positive validation

| Protein | Disease | n | Spearman $\rho$ | MAE (yr) |
|---------|---------|---|-----------------|----------|
| TTR | Cardiac/polyneuropathy amyloidosis | 24 | -0.977 | 9.2 |
| LYZ | Hereditary systemic amyloidosis | 6 | -0.714 | 7.2 |
| Gelsolin | Finnish amyloidosis | 3 | -1.000 | 3.8 |

### Pre-specified negative controls

| Protein | Disease | $\rho$ | Result | Mechanism |
|---------|---------|--------|--------|-----------|
| SOD1 | ALS | +0.28 | Fails (wrong sign) | Gain-of-toxic-function |
| PRNP | CJD/FFI/GSS | -0.29 | Fails (n.s.) | Templated conversion |
| A$\beta$42 | Alzheimer's | n/a | Not applicable | IDP; no stable fold |

---

## Repository structure

```
code/                          # All simulation and analysis scripts
  protein_folding_cbfi.py      # Thermodynamic sigma(T) for real proteins
  protein_telescope_v2.py      # Single-basin Go model validation
  protein_dual_basin.py        # Dual-basin model: alpha scan, transient sigma
  protein_dual_basin_intervention.py  # Chaperone + stabilizer interventions
  protein_alzheimer_mutations.py      # APP mutation encoding onto alpha axis
  protein_validation.py        # Experimental ddG cross-validation (37 mutations)
  protein_largescale_validation.py    # 20-protein consistency check, VUS predictions
  robustness_sweep.py          # Parameter sweep: 72 combinations of N, S, contacts
  ttr_validation.py            # TTR onset prediction (main validation)
  lyz_validation.py            # Lysozyme validation
  gelsolin_validation.py       # Gelsolin validation
  sod1_validation.py           # SOD1/ALS negative control
  prnp_validation.py           # Prion disease negative control
  structure_based_ddg.py       # DynaMut2 + ESM-1v cross-method check for Abeta42
  hdx_proxy.py                 # HDX-MS/NMR order parameter proxy for sigma
  drift_sensitivity.py         # Drift-rate sensitivity (0.02-0.05/decade)
  benchmark_early_warning.py   # Benchmark: sigma vs 4 alternative indicators
  paper5_figures.py            # Figure generation

data/                          # Result files (CSV)
  ttr_results.csv              # TTR: 25 mutations, sigma, onset, phenotype
  lyz_results.csv              # Lysozyme: 6 mutations
  gelsolin_results.csv         # Gelsolin: 5 mutations
  sod1_results.csv             # SOD1: 10 mutations (negative control)
  prnp_results.csv             # PRNP: 7 mutations (negative control)
  structure_ddg_results.csv    # DynaMut2 vs ESM-1v for Abeta42
  drift_sensitivity_results.csv  # Onset envelopes across drift rates
  robustness_results.csv       # 72-combination parameter sweep
  hdx_results.csv              # HDX-MS proxy validation


```

---

## Requirements

```
Python >= 3.8
numpy
scipy
matplotlib
```

No molecular dynamics software, no GPU, no special hardware. Every script runs on a laptop.

---

## Reproducing results

All scripts use fixed random seeds. To reproduce the main results:

```bash
# TTR validation (main result)
python code/ttr_validation.py

# Negative controls
python code/sod1_validation.py
python code/prnp_validation.py

# Lysozyme + Gelsolin
python code/lyz_validation.py
python code/gelsolin_validation.py

# Full parameter sweep (72 combinations, ~10 min)
python code/robustness_sweep.py

# Benchmark against 4 alternative indicators
python code/benchmark_early_warning.py

# Drift-rate sensitivity
python code/drift_sensitivity.py

# All figures
python code/paper5_figures.py
```

---

## Key numbers

- **TTR**: $\rho(\sigma, \text{onset}) = -0.977$, $p < 10^{-15}$, MAE = 9.2 yr (bootstrap 95% CI: 7.0-11.4)
- **Drift rate**: 0.03/decade from published chaperone decline data, **not fit to onset data**
- **Early warning**: 72/72 parameter combinations, mean lead 36%
- **Benchmark**: All 5 indicators achieve 72/72. $\sigma$ is not unique at detection. Its contribution is the interpretive framework.
- **$\sigma$ is a monotonic transform of $\Delta\Delta G$** for a single protein. The correlation magnitude $\rho(\sigma, \text{onset}) = \rho(\Delta\Delta G, \text{onset})$. This is a mathematical fact, stated explicitly.

---

## What this is not

- Not a clinical diagnostic tool
- Not a universal predictor of all protein misfolding diseases
- Not a replacement for experimental $\Delta\Delta G$ measurement
- The Go model is a 20-residue lattice model with 8 states per residue. It captures topology, not sequence-specific energetics.

The framework works where disease **is** thermodynamic instability. Where it isn't (SOD1, prions, IDPs), $\sigma$ correctly has nothing to say.

---

## License

APGL 3.0 or later + Commercial

## Contact

ForgottenForge
nfo@forgottenforge.xyz
