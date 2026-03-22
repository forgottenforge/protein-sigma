#!/usr/bin/env python3
"""
Benchmark: sigma vs four alternative early-warning indicators.

Tests all five indicators on the same 72 parameter combinations (dual-basin
Go model) used in robustness_sweep.py. For each indicator, we compute the
alpha value at which the early-warning threshold is crossed, and compare
it to alpha_Q (where the native fraction Q drops to 0.5).

Indicators:
  1. sigma (contraction index)     — kinetic funnel quality
  2. DeltaG threshold              — thermodynamic basin stability
  3. Q-variance (susceptibility)   — fluctuation-based
  4. CSD (critical slowing down)   — relaxation-time divergence
  5. D-only threshold              — geometric access without energy
"""

import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ── Physical constants ──────────────────────────────────────────────
T = 310.0
kB = 1.380649e-23
NA = 6.022e23
kT = kB * T * NA / 1000.0   # ~2.577 kJ/mol at 310K
beta = 1.0 / kT              # ~0.388 mol/kJ

# Go model contact energy
epsilon = 2.5  # kJ/mol per contact (~1 kT, typical for coarse-grained models)

# ── Parameter grid ──────────────────────────────────────────────────
N_values = [20, 30, 40, 50]
S_values = [6, 8, 10]
contacts_symmetric = [8, 12, 16]
contacts_asymmetric = [(12, 8), (12, 10), (12, 14)]

param_combos = []
for N in N_values:
    for S in S_values:
        for c in contacts_symmetric:
            param_combos.append((N, S, c, c))
        for c_nat, c_amy in contacts_asymmetric:
            param_combos.append((N, S, c_nat, c_amy))

n_alpha = 200


def compute_sigma_and_Q(alpha, N, S, c_nat, c_amy):
    """
    Dual-basin Go model: sigma_nat and Q_nat vs mutation load alpha.

    Returns (sigma_nat, Q_nat, D_nat, gamma_nat).
    """
    # Basin energy slopes
    slope_nat = epsilon * c_nat * (1.0 - alpha)
    slope_amy = epsilon * c_amy * alpha

    # D_nat: fraction of productive moves
    p_geom = min(2.0 * c_nat / N, 1.0) / S
    noise = kT * (N / 10.0)**0.5
    p_guided = slope_nat / (slope_nat + slope_amy + noise)
    D_nat = p_geom * p_guided

    # gamma_nat: Boltzmann contraction factor
    gamma_nat = np.exp(-beta * epsilon * (1.0 - alpha))

    contraction = D_nat * (1.0 - gamma_nat)

    # Amyloid expansion
    p_geom_amy = min(2.0 * c_amy / N, 1.0) / S
    p_guided_amy = slope_amy / (slope_nat + slope_amy + noise)
    D_amy = p_geom_amy * p_guided_amy
    gamma_amy = np.exp(-beta * epsilon * alpha)
    expansion = D_amy * (1.0 - gamma_amy)

    # sigma_nat
    sigma_nat = 1.0 - contraction + expansion

    # Q_nat: cooperative equilibrium
    alpha_0 = c_nat / (c_nat + c_amy)
    rho = c_nat / N
    coop_shift = 0.42 * rho / (rho + 0.12) * (6.0 / S)**0.25
    width = (0.06 + 0.03 * np.log(S) + 0.025 * np.log(N)
             + 0.15 / np.sqrt(c_nat))
    alpha_mid_Q = alpha_0 + coop_shift
    Q_nat = 1.0 / (1.0 + np.exp((alpha - alpha_mid_Q) / width))

    return sigma_nat, Q_nat, D_nat, gamma_nat


def find_crossing(alphas, values, threshold, direction='rising'):
    """Find alpha where values crosses threshold via linear interpolation."""
    for i in range(len(alphas) - 1):
        if direction == 'rising':
            if values[i] < threshold and values[i + 1] >= threshold:
                frac = (threshold - values[i]) / (values[i + 1] - values[i])
                return alphas[i] + frac * (alphas[i + 1] - alphas[i])
        elif direction == 'falling':
            if values[i] > threshold and values[i + 1] <= threshold:
                frac = (values[i] - threshold) / (values[i] - values[i + 1])
                return alphas[i] + frac * (alphas[i + 1] - alphas[i])
    return None


# ── Main sweep ──────────────────────────────────────────────────────
alphas = np.linspace(0.0, 1.0, n_alpha)

# Storage for all results
all_results = []
rep_data = None  # representative case for panel (a)

print("=" * 90)
print("BENCHMARK: sigma vs 4 alternative early-warning indicators")
print("=" * 90)
print(f"  T = {T:.0f} K,  kT = {kT:.3f} kJ/mol,  beta = {beta:.4f} mol/kJ")
print(f"  epsilon = {epsilon} kJ/mol per contact")
print(f"  alpha steps = {n_alpha}")
print(f"  parameter combinations = {len(param_combos)}")
print()

for N, S, c_nat, c_amy in param_combos:
    sigmas = np.zeros(n_alpha)
    Qs = np.zeros(n_alpha)
    Ds = np.zeros(n_alpha)
    gammas = np.zeros(n_alpha)

    for j, a in enumerate(alphas):
        s, q, d, g = compute_sigma_and_Q(a, N, S, c_nat, c_amy)
        sigmas[j] = s
        Qs[j] = q
        Ds[j] = d
        gammas[j] = g

    # ── Reference crossings ──
    alpha_sigma = find_crossing(alphas, sigmas, 1.0, direction='rising')
    alpha_Q = find_crossing(alphas, Qs, 0.5, direction='falling')

    # ── Indicator 2: DeltaG threshold ──
    # DeltaG(alpha) = epsilon * [c_amy * alpha - c_nat * (1-alpha)]
    # Crosses zero when c_nat*(1-alpha) = c_amy*alpha
    # => alpha_DG = c_nat / (c_nat + c_amy)
    DeltaG = np.array([epsilon * (c_amy * a - c_nat * (1.0 - a)) for a in alphas])
    alpha_DG = find_crossing(alphas, DeltaG, 0.0, direction='rising')

    # ── Indicator 3: Q-variance (susceptibility chi) ──
    # chi(alpha) = Q * (1 - Q), peaks near the transition
    # Early warning = when chi exceeds 50% of its maximum
    chi = Qs * (1.0 - Qs)
    chi_max = np.max(chi)
    chi_threshold = 0.5 * chi_max
    # chi rises then falls; we want the first crossing on the rising side
    alpha_chi = find_crossing(alphas, chi, chi_threshold, direction='rising')

    # ── Indicator 4: CSD (critical slowing down) ──
    # tau(alpha) = 1 / (1 - sigma) for sigma < 1
    # Early warning = when tau exceeds 5x its value at alpha=0
    tau = np.zeros(n_alpha)
    for j in range(n_alpha):
        if sigmas[j] < 1.0:
            tau[j] = 1.0 / (1.0 - sigmas[j])
        else:
            tau[j] = np.inf
    tau_0 = tau[0] if tau[0] < np.inf else 1.0
    csd_threshold = 5.0 * tau_0
    alpha_CSD = find_crossing(alphas, tau, csd_threshold, direction='rising')

    # ── Indicator 5: D-only threshold ──
    # Early warning = when D_nat drops below D_nat(0)/2
    D_0 = Ds[0]
    D_threshold = D_0 / 2.0
    alpha_Donly = find_crossing(alphas, Ds, D_threshold, direction='falling')

    # ── Compute lead times ──
    indicators = {
        'sigma': alpha_sigma,
        'DeltaG': alpha_DG,
        'chi': alpha_chi,
        'CSD': alpha_CSD,
        'D_only': alpha_Donly,
    }

    row = {
        'N': N, 'S': S, 'c_nat': c_nat, 'c_amy': c_amy,
        'alpha_Q': alpha_Q,
    }

    for name, alpha_cross in indicators.items():
        row[f'alpha_{name}'] = alpha_cross
        if alpha_cross is not None and alpha_Q is not None and alpha_Q > 0:
            lead_pct = (alpha_Q - alpha_cross) / alpha_Q * 100.0
            row[f'early_{name}'] = lead_pct > 0
            row[f'lead_{name}'] = lead_pct
        else:
            row[f'early_{name}'] = False
            row[f'lead_{name}'] = None

    all_results.append(row)

    # Save representative case data for plotting
    if N == 40 and S == 8 and c_nat == 12 and c_amy == 12:
        rep_data = {
            'alphas': alphas.copy(),
            'sigmas': sigmas.copy(),
            'Qs': Qs.copy(),
            'Ds': Ds.copy(),
            'DeltaG': DeltaG.copy(),
            'chi': chi.copy(),
            'tau': tau.copy(),
            'alpha_Q': alpha_Q,
            'alpha_sigma': alpha_sigma,
            'alpha_DG': alpha_DG,
            'alpha_chi': alpha_chi,
            'alpha_CSD': alpha_CSD,
            'alpha_Donly': alpha_Donly,
        }


# ── Summary table ───────────────────────────────────────────────────
indicator_names = [
    ('sigma', r'sigma (contraction index)'),
    ('DeltaG', r'DeltaG threshold'),
    ('chi', r'Q-variance (chi > 50%max)'),
    ('CSD', r'CSD (tau > 5*tau_0)'),
    ('D_only', r'D-only (D < D_0/2)'),
]

print()
print(f"{'Indicator':<28} {'Early warnings':>15} {'Mean lead%':>12} {'Spread(std)':>12}")
print("-" * 70)

summary_data = {}
for key, label in indicator_names:
    early_count = sum(1 for r in all_results if r[f'early_{key}'])
    leads = [r[f'lead_{key}'] for r in all_results
             if r[f'lead_{key}'] is not None and r[f'early_{key}']]
    all_leads = [r[f'lead_{key}'] for r in all_results
                 if r[f'lead_{key}'] is not None]

    mean_lead = np.mean(all_leads) if all_leads else 0.0
    std_lead = np.std(all_leads) if all_leads else 0.0

    summary_data[key] = {
        'early_count': early_count,
        'mean_lead': mean_lead,
        'std_lead': std_lead,
        'all_leads': all_leads,
    }

    print(f"{label:<28} {early_count:>6}/72     {mean_lead:>8.1f}%     {std_lead:>8.4f}")

print()

# ── Generate figure ─────────────────────────────────────────────────
fig_dir = '/home/ffai/code/papers/paper5_submission'
os.makedirs(fig_dir, exist_ok=True)
fig_path = os.path.join(fig_dir, 'fig_benchmark.pdf')

fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(16, 5))

# ── Panel (a): Representative case, all 5 indicators normalized to [0,1] ──
if rep_data is not None:
    al = rep_data['alphas']
    aQ = rep_data['alpha_Q']

    # Normalize each indicator to [0, 1] for visual comparison
    # 1. sigma: normalize so that sigma=1 maps to threshold line at some y
    #    Use (sigma - sigma_min) / (sigma_max - sigma_min)
    sig_raw = rep_data['sigmas']
    sig_norm = (sig_raw - sig_raw.min()) / (sig_raw.max() - sig_raw.min())
    sig_thresh_norm = (1.0 - sig_raw.min()) / (sig_raw.max() - sig_raw.min())

    # 2. DeltaG: normalize so DeltaG=0 is the threshold
    dg_raw = rep_data['DeltaG']
    dg_norm = (dg_raw - dg_raw.min()) / (dg_raw.max() - dg_raw.min())
    dg_thresh_norm = (0.0 - dg_raw.min()) / (dg_raw.max() - dg_raw.min())

    # 3. chi: normalize, threshold at 50% of max
    chi_raw = rep_data['chi']
    chi_norm = chi_raw / chi_raw.max() if chi_raw.max() > 0 else chi_raw
    chi_thresh_norm = 0.5

    # 4. tau (CSD): clip infinities for normalization, use log scale
    tau_raw = rep_data['tau'].copy()
    tau_0 = tau_raw[0] if tau_raw[0] < np.inf else 1.0
    csd_thresh = 5.0 * tau_0
    # Cap tau at 10x threshold for visual purposes
    tau_cap = np.minimum(tau_raw, 10.0 * csd_thresh)
    tau_norm = (tau_cap - tau_cap.min()) / (tau_cap.max() - tau_cap.min()) \
        if tau_cap.max() > tau_cap.min() else np.zeros_like(tau_cap)
    csd_thresh_norm = (csd_thresh - tau_cap.min()) / (tau_cap.max() - tau_cap.min()) \
        if tau_cap.max() > tau_cap.min() else 0.5

    # 5. D-only: normalize, threshold at D0/2
    d_raw = rep_data['Ds']
    d_norm = (d_raw - d_raw.min()) / (d_raw.max() - d_raw.min()) \
        if d_raw.max() > d_raw.min() else np.zeros_like(d_raw)
    D0 = d_raw[0]
    d_thresh_norm = (D0 / 2.0 - d_raw.min()) / (d_raw.max() - d_raw.min()) \
        if d_raw.max() > d_raw.min() else 0.5

    # Plot normalized indicators
    colors = ['#2e7d32', '#1565c0', '#e65100', '#6a1b9a', '#c62828']
    ax1.plot(al, sig_norm, color=colors[0], linewidth=2.2,
             label=r'$\sigma$ (contraction)')
    ax1.plot(al, dg_norm, color=colors[1], linewidth=1.8, linestyle='--',
             label=r'$\Delta G$ threshold')
    ax1.plot(al, chi_norm, color=colors[2], linewidth=1.8, linestyle='-.',
             label=r'$\chi$ (Q-variance)')
    ax1.plot(al, tau_norm, color=colors[3], linewidth=1.8, linestyle=':',
             label=r'$\tau$ (CSD)')
    ax1.plot(al, d_norm, color=colors[4], linewidth=1.8, linestyle='--',
             label=r'$D$-only')

    # Mark threshold crossings
    crossing_alphas = [
        (rep_data['alpha_sigma'], colors[0], r'$\sigma$'),
        (rep_data['alpha_DG'], colors[1], r'$\Delta G$'),
        (rep_data['alpha_chi'], colors[2], r'$\chi$'),
        (rep_data['alpha_CSD'], colors[3], r'$\tau$'),
        (rep_data['alpha_Donly'], colors[4], r'$D$'),
    ]
    for ac, col, lbl in crossing_alphas:
        if ac is not None:
            ax1.axvline(ac, color=col, linestyle=':', linewidth=0.8, alpha=0.5)

    # Mark Q crossing
    if aQ is not None:
        ax1.axvline(aQ, color='red', linewidth=2.0, linestyle='--', alpha=0.7,
                    label=r'$Q = 0.5$')

    ax1.set_xlabel(r'$\alpha$ (mutation load)', fontsize=11)
    ax1.set_ylabel('Normalized indicator value', fontsize=11)
    ax1.set_title('(a) Representative case (N=40, S=8, c=12)', fontsize=11)
    ax1.legend(loc='upper left', fontsize=7.5, frameon=True, framealpha=0.9)
    ax1.set_xlim(0, 1)
    ax1.set_ylim(-0.05, 1.05)
    ax1.spines['top'].set_visible(False)
    ax1.spines['right'].set_visible(False)

# ── Panel (b): Bar chart of early-warning success rates ──
labels_bar = [
    r'$\sigma$' + '\n(contraction)',
    r'$\Delta G$' + '\n(thermo.)',
    r'$\chi$' + '\n(Q-var.)',
    r'$\tau$' + '\n(CSD)',
    r'$D$-only' + '\n(geom.)',
]
keys = ['sigma', 'DeltaG', 'chi', 'CSD', 'D_only']
counts_bar = [summary_data[k]['early_count'] for k in keys]
colors_bar = ['#2e7d32', '#1565c0', '#e65100', '#6a1b9a', '#c62828']

bars = ax2.bar(labels_bar, counts_bar, color=colors_bar, edgecolor='black',
               linewidth=0.8, width=0.6)
for bar, count in zip(bars, counts_bar):
    y_pos = bar.get_height() + 1.0
    ax2.text(bar.get_x() + bar.get_width() / 2.0, y_pos,
             f'{count}/72', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('Early-warning successes (out of 72)', fontsize=11)
ax2.set_title('(b) Success rate by indicator', fontsize=11)
ax2.set_ylim(0, 72 * 1.3)
ax2.axhline(72, color='black', linestyle=':', linewidth=0.8, alpha=0.4)
ax2.text(4.6, 72 + 0.5, '72', fontsize=8, color='gray', ha='left', va='bottom')
ax2.spines['top'].set_visible(False)
ax2.spines['right'].set_visible(False)

# ── Panel (c): Box plot of lead time distributions ──
lead_data = []
lead_labels = []
lead_colors = []
for k, lbl, col in zip(keys, labels_bar, colors_bar):
    leads = summary_data[k]['all_leads']
    if leads:
        lead_data.append(leads)
    else:
        lead_data.append([0.0])
    lead_labels.append(lbl)
    lead_colors.append(col)

bp = ax3.boxplot(lead_data, labels=lead_labels, patch_artist=True,
                 widths=0.5, showfliers=True,
                 flierprops=dict(marker='o', markersize=3, alpha=0.5))
for patch, col in zip(bp['boxes'], lead_colors):
    patch.set_facecolor(col)
    patch.set_alpha(0.6)
for median in bp['medians']:
    median.set_color('black')
    median.set_linewidth(1.5)

ax3.axhline(0, color='black', linestyle='--', linewidth=0.8, alpha=0.4)
ax3.set_ylabel(r'Lead time $\Delta\alpha / \alpha_Q$ (%)', fontsize=11)
ax3.set_title(r'(c) Lead time distribution', fontsize=11)
ax3.spines['top'].set_visible(False)
ax3.spines['right'].set_visible(False)

# Add annotation: positive = early warning, negative = late
ax3.text(0.98, 0.97, 'Early warning\n(positive = better)',
         transform=ax3.transAxes, fontsize=8, ha='right', va='top',
         color='gray')

plt.tight_layout()
plt.savefig(fig_path, dpi=300, bbox_inches='tight')
print(f"Figure saved to: {fig_path}")
print()
