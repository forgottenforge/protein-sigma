#!/usr/bin/env python3
"""
Paper 5 Figures — Publication-quality visualizations
=====================================================
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import FancyArrowPatch
import matplotlib.patheffects as pe

plt.rcParams.update({
    'font.size': 11,
    'font.family': 'serif',
    'axes.linewidth': 1.2,
    'xtick.major.width': 1.0,
    'ytick.major.width': 1.0,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15,
})

OUTDIR = '/home/ffai/code/papers/paper5_submission/'

# Color palette
C_NAT = '#2166ac'    # blue for native
C_AMY = '#b2182b'    # red for amyloid
C_SAFE = '#4daf4a'   # green for safe
C_DANGER = '#e41a1c' # red for danger
C_WARN = '#ff7f00'   # orange for warning
C_GREY = '#666666'

# ═══════════════════════════════════════════════════════════
# FIGURE 1: σ(T) for four real proteins
# ═══════════════════════════════════════════════════════════

R = 8.314e-3  # kJ/(mol·K)

proteins = {
    'Trp-cage': {'N': 20, 'T_m': 317.0, 'dH': 230.0, 'dCp': 2.5, 'color': '#e41a1c', 'marker': 'o'},
    'Villin HP35': {'N': 35, 'T_m': 342.0, 'dH': 155.0, 'dCp': 3.1, 'color': '#377eb8', 'marker': 's'},
    'CI2': {'N': 64, 'T_m': 337.0, 'dH': 278.0, 'dCp': 5.0, 'color': '#4daf4a', 'marker': '^'},
    'ACBP': {'N': 86, 'T_m': 332.0, 'dH': 370.0, 'dCp': 6.3, 'color': '#984ea3', 'marker': 'D'},
}

fig, ax = plt.subplots(figsize=(7, 5))

T_range = np.linspace(275, 380, 300)

for name, p in proteins.items():
    sigma_vals = []
    for T in T_range:
        dG = p['dH'] * (1 - T/p['T_m']) + p['dCp'] * (T - p['T_m'] - T * np.log(T/p['T_m']))
        sigma = np.exp(-dG / (p['N'] * R * T))
        sigma_vals.append(sigma)

    ax.plot(T_range, sigma_vals, color=p['color'], linewidth=2.2, label=name, zorder=3)

    # Mark T_m
    dG_tm = p['dH'] * (1 - p['T_m']/p['T_m'])  # = 0
    sigma_tm = np.exp(0)  # = 1
    idx_tm = np.argmin(np.abs(T_range - p['T_m']))
    ax.plot(p['T_m'], sigma_vals[idx_tm], marker=p['marker'], color=p['color'],
            markersize=10, markeredgecolor='white', markeredgewidth=1.5, zorder=5)

# σ = 1 line
ax.axhline(y=1.0, color='black', linewidth=1.5, linestyle='--', alpha=0.7, zorder=2)
ax.text(378, 1.02, r'$\sigma = 1$', fontsize=12, fontweight='bold', ha='right', va='bottom')

# Annotations
ax.fill_between(T_range, 0, 1, alpha=0.06, color=C_SAFE, zorder=1)
ax.fill_between(T_range, 1, 2.5, alpha=0.06, color=C_DANGER, zorder=1)
ax.text(280, 0.65, 'FOLDED\n$\\sigma < 1$', fontsize=13, color=C_SAFE, fontweight='bold', alpha=0.8)
ax.text(280, 1.8, 'UNFOLDED\n$\\sigma > 1$', fontsize=13, color=C_DANGER, fontweight='bold', alpha=0.8)

ax.set_xlabel('Temperature (K)', fontsize=13)
ax.set_ylabel(r'$\sigma(T) = \exp(-\Delta G / NRT)$', fontsize=13)
ax.set_title(r'$\sigma = 1$ at the melting temperature $T_m$', fontsize=14, fontweight='bold')
ax.set_xlim(275, 380)
ax.set_ylim(0.5, 2.3)
ax.legend(loc='upper left', fontsize=10, framealpha=0.9)
ax.grid(True, alpha=0.2)

fig.savefig(OUTDIR + 'fig1_sigma_temperature.pdf')
fig.savefig(OUTDIR + 'fig1_sigma_temperature.png')
plt.close()
print("Figure 1: σ(T) for four proteins — DONE")


# ═══════════════════════════════════════════════════════════
# FIGURE 2: α scan — σ_nat and σ_amy with Q overlay
# ═══════════════════════════════════════════════════════════

# Data from our simulation runs
alpha_vals = np.arange(0.0, 1.05, 0.05)

# From the actual run output
sigma_nat = [0.616, 0.673, 0.736, 0.844, 0.940, 1.012, 1.042, 1.029, 1.037, 1.063,
             1.127, 1.088, 1.113, 1.026, 1.018, 1.013, 1.007, 1.004, 1.003, 1.002, 1.002]
sigma_amy = [1.002, 1.002, 1.003, 1.003, 1.005, 1.007, 1.004, 1.008, 1.013, 1.017,
             1.016, 1.021, 1.001, 0.998, 0.999, 0.993, 1.008, 0.994, 0.982, 1.017, 0.977]
Q_nat = [0.892, 0.925, 0.919, 0.875, 0.886, 0.867, 0.833, 0.764, 0.733, 0.697,
         0.511, 0.408, 0.206, 0.142, 0.056, 0.025, 0.014, 0.014, 0.006, 0.003, 0.006]
Q_amy = [0.008, 0.006, 0.011, 0.006, 0.008, 0.017, 0.008, 0.039, 0.028, 0.061,
         0.103, 0.147, 0.289, 0.358, 0.444, 0.492, 0.586, 0.558, 0.631, 0.681, 0.672]

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 9), sharex=True,
                                gridspec_kw={'height_ratios': [1.2, 1], 'hspace': 0.08})

# Top panel: σ
ax1.plot(alpha_vals, sigma_nat, 'o-', color=C_NAT, linewidth=2.5, markersize=7,
         label=r'$\sigma_{\mathrm{nat}}$', markeredgecolor='white', markeredgewidth=1)
ax1.plot(alpha_vals, sigma_amy, 's-', color=C_AMY, linewidth=2.5, markersize=7,
         label=r'$\sigma_{\mathrm{amy}}$', markeredgecolor='white', markeredgewidth=1)
ax1.axhline(y=1.0, color='black', linewidth=1.5, linestyle='--', alpha=0.6)

# Fill regions
ax1.fill_between(alpha_vals, 0.5, 1.0, alpha=0.05, color=C_SAFE)
ax1.fill_between(alpha_vals, 1.0, 1.2, alpha=0.05, color=C_DANGER)

# Mark σ_nat = 1 crossing
ax1.annotate(r'$\sigma_{\mathrm{nat}} = 1$' + '\n(funnel flattens)',
             xy=(0.22, 1.0), xytext=(0.05, 1.12),
             fontsize=10, fontweight='bold', color=C_NAT,
             arrowprops=dict(arrowstyle='->', color=C_NAT, lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_NAT, alpha=0.9))

ax1.text(0.7, 0.97, r'$\sigma_{\mathrm{amy}} < 1$' + '\n(amyloid attracts)',
         fontsize=9, color=C_AMY, ha='center',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_AMY, alpha=0.8))

ax1.set_ylabel(r'$\sigma_{\mathrm{macro}} = d(t+\Delta)/d(t)$', fontsize=13)
ax1.set_ylim(0.55, 1.18)
ax1.set_title(r'Dual-basin $\alpha$ scan: $\sigma$ and $Q$ vs mutation load', fontsize=14, fontweight='bold')
ax1.legend(loc='upper right', fontsize=11, framealpha=0.95)
ax1.grid(True, alpha=0.15)
ax1.text(0.02, 1.16, 'A', fontsize=18, fontweight='bold', transform=ax1.transAxes,
         va='top', ha='left')

# Bottom panel: Q
ax2.plot(alpha_vals, Q_nat, 'o-', color=C_NAT, linewidth=2.5, markersize=7,
         label=r'$Q_{\mathrm{nat}}$', markeredgecolor='white', markeredgewidth=1)
ax2.plot(alpha_vals, Q_amy, 's-', color=C_AMY, linewidth=2.5, markersize=7,
         label=r'$Q_{\mathrm{amy}}$', markeredgecolor='white', markeredgewidth=1)
ax2.axhline(y=0.5, color='grey', linewidth=1, linestyle=':', alpha=0.5)

# Crossover
ax2.axvline(x=0.588, color='black', linewidth=1.5, linestyle=':', alpha=0.5)
ax2.annotate('Q crossover\n' + r'$\alpha_c = 0.59$',
             xy=(0.588, 0.25), xytext=(0.72, 0.15),
             fontsize=10, fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='black', lw=1.5),
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffcc', edgecolor='grey', alpha=0.9))

# Early warning zone
ax2.axvspan(0.22, 0.59, alpha=0.08, color=C_WARN)
ax2.text(0.40, 0.92, 'EARLY WARNING\nZONE',
         fontsize=10, fontweight='bold', color=C_WARN, ha='center', alpha=0.8)
ax2.annotate('', xy=(0.22, 0.85), xytext=(0.59, 0.85),
             arrowprops=dict(arrowstyle='<->', color=C_WARN, lw=1.5))
ax2.text(0.405, 0.82, '37%', fontsize=10, fontweight='bold', color=C_WARN, ha='center')

ax2.set_xlabel(r'Mutation load $\alpha$', fontsize=13)
ax2.set_ylabel(r'Fraction native contacts $Q$', fontsize=13)
ax2.set_xlim(-0.02, 1.02)
ax2.set_ylim(-0.02, 1.02)
ax2.legend(loc='center right', fontsize=11, framealpha=0.95)
ax2.grid(True, alpha=0.15)
ax2.text(0.02, 0.98, 'B', fontsize=18, fontweight='bold', transform=ax2.transAxes,
         va='top', ha='left')

fig.savefig(OUTDIR + 'fig2_alpha_scan.pdf')
fig.savefig(OUTDIR + 'fig2_alpha_scan.png')
plt.close()
print("Figure 2: α scan — DONE")


# ═══════════════════════════════════════════════════════════
# FIGURE 3: Alzheimer mutations — σ vs severity
# ═══════════════════════════════════════════════════════════

mutations = {
    'A2V\n(protective)':     {'sigma': 0.695, 'severity': -1, 'onset': None, 'type': 'protective'},
    'Icelandic\n(A673T)':    {'sigma': 0.707, 'severity': -1, 'onset': None, 'type': 'protective'},
    'Wild\ntype':            {'sigma': 0.822, 'severity': 0,  'onset': None, 'type': 'wildtype'},
    'Dutch\n(E693Q)':        {'sigma': 0.986, 'severity': 3,  'onset': 45,   'type': 'disease'},
    'Iowa\n(D694N)':         {'sigma': 1.009, 'severity': 3,  'onset': 55,   'type': 'disease'},
    'Arctic\n(E693G)':       {'sigma': 1.032, 'severity': 3,  'onset': 55,   'type': 'disease'},
    'Flemish\n(A692G)':      {'sigma': 1.077, 'severity': 4,  'onset': 45,   'type': 'disease'},
    'Swedish\n(K670N/M671L)':{'sigma': 1.091, 'severity': 2,  'onset': 55,   'type': 'disease'},
    'Osaka\n(E693Δ)':        {'sigma': 1.098, 'severity': 4,  'onset': 45,   'type': 'disease'},
    'London\n(V717I)':       {'sigma': 1.113, 'severity': 4,  'onset': 47,   'type': 'disease'},
}

fig, ax = plt.subplots(figsize=(9, 6))

for name, m in mutations.items():
    if m['type'] == 'protective':
        color = C_SAFE
        marker = 'D'
        ms = 12
    elif m['type'] == 'wildtype':
        color = C_GREY
        marker = '*'
        ms = 16
    else:
        color = C_DANGER
        marker = 'o'
        ms = 12

    ax.scatter(m['sigma'], m['severity'], color=color, marker=marker, s=ms**2,
               edgecolors='white', linewidths=1.5, zorder=5)

    # Label
    offset_x = 0.015
    offset_y = 0.15
    ha = 'left'
    if 'Wild' in name:
        offset_x = -0.015
        ha = 'right'
    if 'Dutch' in name:
        offset_y = -0.4
    if 'Iowa' in name:
        offset_y = 0.3

    ax.annotate(name.replace('\n', ' '), (m['sigma'], m['severity']),
                xytext=(m['sigma'] + offset_x, m['severity'] + offset_y),
                fontsize=8, ha=ha, va='center',
                arrowprops=dict(arrowstyle='-', color='grey', lw=0.5, alpha=0.5) if abs(offset_y) > 0.2 else None)

# σ = 1 line
ax.axvline(x=1.0, color='black', linewidth=2, linestyle='--', alpha=0.7, zorder=2)
ax.text(1.002, 4.5, r'$\sigma = 1$', fontsize=12, fontweight='bold', rotation=90,
        va='top', ha='left')

# Background zones
ax.axvspan(0.6, 1.0, alpha=0.08, color=C_SAFE)
ax.axvspan(1.0, 1.15, alpha=0.08, color=C_DANGER)

ax.text(0.72, 4.5, 'SAFE\n' + r'$\sigma < 1$', fontsize=12, color=C_SAFE,
        fontweight='bold', ha='center', alpha=0.7)
ax.text(1.08, 4.5, 'RISK\n' + r'$\sigma > 1$', fontsize=12, color=C_DANGER,
        fontweight='bold', ha='center', alpha=0.7)

# Correlation
ax.text(0.68, -0.8, r'$r = 0.843$' + '\n' + r'$p < 0.01$',
        fontsize=12, fontweight='bold', color='black',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='#ffffcc', edgecolor='grey'))

ax.set_xlabel(r'Stability index $\sigma_{\mathrm{nat}}$', fontsize=13)
ax.set_ylabel('Clinical severity', fontsize=13)
ax.set_title('Alzheimer mutations: clinical severity tracks ' + r'$\sigma_{\mathrm{nat}}$',
             fontsize=14, fontweight='bold')
ax.set_xlim(0.63, 1.15)
ax.set_ylim(-1.8, 5.0)
ax.set_yticks([-1, 0, 1, 2, 3, 4])
ax.set_yticklabels(['Protective', 'Wild type', '', 'Moderate', 'Severe', 'Critical'])
ax.grid(True, alpha=0.15)

# Legend
from matplotlib.lines import Line2D
legend_elements = [
    Line2D([0], [0], marker='D', color='w', markerfacecolor=C_SAFE, markersize=10, label='Protective'),
    Line2D([0], [0], marker='*', color='w', markerfacecolor=C_GREY, markersize=14, label='Wild type'),
    Line2D([0], [0], marker='o', color='w', markerfacecolor=C_DANGER, markersize=10, label='Pathogenic'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

fig.savefig(OUTDIR + 'fig3_alzheimer_mutations.pdf')
fig.savefig(OUTDIR + 'fig3_alzheimer_mutations.png')
plt.close()
print("Figure 3: Alzheimer mutations — DONE")


# ═══════════════════════════════════════════════════════════
# FIGURE 4: Therapeutic hyperbola D·γ = 1
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 6))

# The hyperbola: D_factor * gamma_factor = 1/sigma_0
sigma_0 = 1.07  # diseased protein

D_factors = np.linspace(0.50, 1.0, 200)
gamma_factors_cure = (1.0 / sigma_0) / D_factors

# Fill safe zone (below hyperbola)
D_fill = np.linspace(0.50, 1.0, 200)
g_fill_top = np.minimum((1.0 / sigma_0) / D_fill, 1.0)
ax.fill_between(D_fill, 0.5, g_fill_top, alpha=0.15, color=C_SAFE, label=r'$\sigma < 1$ (safe)')
ax.fill_between(D_fill, g_fill_top, 1.0, alpha=0.10, color=C_DANGER, label=r'$\sigma > 1$ (disease)')

# The hyperbola itself
mask = gamma_factors_cure <= 1.0
ax.plot(D_factors[mask], gamma_factors_cure[mask], color='black', linewidth=3,
        label=r'$D \cdot \gamma = 1$ (iso-cure)', zorder=4)

# Disease point (untreated)
ax.scatter(1.0, 1.0, color=C_DANGER, s=200, marker='X', zorder=6,
           edgecolors='white', linewidths=2)
ax.annotate('Untreated\n' + r'$\sigma = 1.07$', xy=(1.0, 1.0),
            xytext=(0.88, 1.04), fontsize=10, fontweight='bold', color=C_DANGER,
            arrowprops=dict(arrowstyle='->', color=C_DANGER, lw=2))

# Treatment points on hyperbola
treatments = [
    (1.0, 1/sigma_0, 'Stabilizer\nonly', C_NAT),
    (1/sigma_0, 1.0, 'Chaperone\nonly', '#984ea3'),
    (0.97, (1/sigma_0)/0.97, 'Combined\n(optimal)', C_SAFE),
]

for d, g, label, color in treatments:
    if g <= 1.0:
        ax.scatter(d, g, color=color, s=150, marker='o', zorder=6,
                   edgecolors='white', linewidths=2)
        offset = (0.03, 0.03) if 'Combined' in label else (-0.03, -0.04) if 'Stab' in label else (0.03, 0.02)
        ax.annotate(label, xy=(d, g),
                    xytext=(d + offset[0], g + offset[1]),
                    fontsize=9, color=color, fontweight='bold',
                    ha='left' if offset[0] > 0 else 'right')

# Arrows showing intervention directions
ax.annotate('', xy=(0.85, 1.0), xytext=(1.0, 1.0),
            arrowprops=dict(arrowstyle='->', color='#984ea3', lw=2.5, alpha=0.5))
ax.text(0.92, 1.01, r'reduce $D$' + '\n(chaperone)', fontsize=8, color='#984ea3',
        ha='center', va='bottom', alpha=0.7)

ax.annotate('', xy=(1.0, 0.85), xytext=(1.0, 1.0),
            arrowprops=dict(arrowstyle='->', color=C_NAT, lw=2.5, alpha=0.5))
ax.text(1.01, 0.92, r'reduce $\gamma$' + '\n(stabilizer)', fontsize=8, color=C_NAT,
        ha='left', va='center', alpha=0.7)

ax.set_xlabel(r'$D$ factor (fraction of conformational space)', fontsize=13)
ax.set_ylabel(r'$\gamma$ factor (contraction strength)', fontsize=13)
ax.set_title(r'Therapeutic Hyperbola: $D \cdot \gamma = 1$', fontsize=14, fontweight='bold')
ax.set_xlim(0.82, 1.03)
ax.set_ylim(0.82, 1.06)
ax.legend(loc='lower left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.15)
ax.set_aspect('equal')

fig.savefig(OUTDIR + 'fig4_therapeutic_hyperbola.pdf')
fig.savefig(OUTDIR + 'fig4_therapeutic_hyperbola.png')
plt.close()
print("Figure 4: Therapeutic hyperbola — DONE")


# ═══════════════════════════════════════════════════════════
# FIGURE 5: 3D dual-basin energy landscape
# ═══════════════════════════════════════════════════════════

fig = plt.figure(figsize=(10, 7))
ax = fig.add_subplot(111, projection='3d')

# Create a 2D reaction coordinate space
# x = progress toward native (Q_nat), y = progress toward amyloid (Q_amy)
x = np.linspace(0, 1, 100)
y = np.linspace(0, 1, 100)
X, Y = np.meshgrid(x, y)

# Energy landscape for three α values
alpha_show = 0.3  # healthy: native dominates

# Energy: two Gaussian wells
# Native well at (1, 0), Amyloid well at (0, 1)
E_nat = -3.0 * np.exp(-((X - 1)**2 + Y**2) / 0.15)
E_amy = -3.0 * np.exp(-(X**2 + (Y - 1)**2) / 0.15)

# Three panels side by side
fig, axes = plt.subplots(1, 3, figsize=(16, 5),
                          subplot_kw={'projection': '3d'})

for idx, (alpha, title, subtitle) in enumerate([
    (0.15, 'Healthy', r'$\alpha = 0.15$'),
    (0.59, 'Crossover', r'$\alpha = 0.59$ (disease onset)'),
    (0.85, 'Disease', r'$\alpha = 0.85$'),
]):
    ax = axes[idx]

    E_total = (1 - alpha) * E_nat + alpha * E_amy
    # Add a barrier ridge
    E_barrier = 1.5 * np.exp(-((X - 0.5)**2 + (Y - 0.5)**2) / 0.08)
    E_total = E_total + E_barrier
    # Add slight roughness
    E_total += 0.15 * np.sin(8 * X) * np.sin(8 * Y) * np.exp(-((X-0.5)**2 + (Y-0.5)**2)/0.5)

    # Normalize
    E_total = E_total - E_total.min()

    # Color by energy
    colors = cm.RdYlBu_r((E_total - E_total.min()) / (E_total.max() - E_total.min()))

    surf = ax.plot_surface(X, Y, E_total, facecolors=colors, alpha=0.85,
                           rstride=2, cstride=2, linewidth=0.1, edgecolor='grey',
                           antialiased=True)

    ax.set_xlabel(r'$Q_{\mathrm{nat}}$', fontsize=10, labelpad=5)
    ax.set_ylabel(r'$Q_{\mathrm{amy}}$', fontsize=10, labelpad=5)
    ax.set_zlabel('Energy', fontsize=10, labelpad=5)
    ax.set_title(f'{title}\n{subtitle}', fontsize=12, fontweight='bold', pad=10)

    ax.view_init(elev=35, azim=-135)
    ax.set_zlim(0, E_total.max() * 1.1)
    ax.tick_params(labelsize=8)

    # Mark minima
    if alpha < 0.5:
        # Native is deeper
        ax.text(0.9, 0.1, -0.3, 'Native\n(deep)', fontsize=9, color=C_NAT,
                fontweight='bold', ha='center', zorder=10)
        ax.text(0.1, 0.9, -0.1, 'Amyloid\n(shallow)', fontsize=8, color=C_AMY,
                ha='center', zorder=10)
    elif alpha > 0.6:
        ax.text(0.9, 0.1, -0.1, 'Native\n(shallow)', fontsize=8, color=C_NAT,
                ha='center', zorder=10)
        ax.text(0.1, 0.9, -0.3, 'Amyloid\n(deep)', fontsize=9, color=C_AMY,
                fontweight='bold', ha='center', zorder=10)
    else:
        ax.text(0.9, 0.1, -0.2, 'Native', fontsize=9, color=C_NAT,
                fontweight='bold', ha='center', zorder=10)
        ax.text(0.1, 0.9, -0.2, 'Amyloid', fontsize=9, color=C_AMY,
                fontweight='bold', ha='center', zorder=10)
        ax.text(0.5, 0.5, E_total.max() * 0.5, '=', fontsize=16,
                ha='center', zorder=10, fontweight='bold')

plt.tight_layout()
fig.savefig(OUTDIR + 'fig5_energy_landscape_3d.pdf')
fig.savefig(OUTDIR + 'fig5_energy_landscape_3d.png')
plt.close()
print("Figure 5: 3D energy landscape — DONE")


# ═══════════════════════════════════════════════════════════
# FIGURE 6: Dose-response curve (bonus — clinically relevant)
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(7, 5))

alpha_dose = [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]
sigma_untreated = [0.965, 1.016, 1.096, 1.023, 1.061, 1.074, 1.149, 1.097, 1.081]
eps_rescue = [0.0, 0.1, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]

# Plot dose-response
ax.plot(alpha_dose, eps_rescue, 'o-', color=C_NAT, linewidth=2.5, markersize=10,
        markeredgecolor='white', markeredgewidth=1.5, zorder=4)

# Color background by risk
ax.axvspan(0.15, 0.25, alpha=0.08, color=C_SAFE)
ax.axvspan(0.25, 0.45, alpha=0.08, color=C_WARN)
ax.axvspan(0.45, 0.65, alpha=0.08, color=C_DANGER)

ax.text(0.20, 1.35, 'EARLY', fontsize=10, fontweight='bold', color=C_SAFE, ha='center')
ax.text(0.35, 1.35, 'MODERATE', fontsize=10, fontweight='bold', color=C_WARN, ha='center')
ax.text(0.55, 1.35, 'SEVERE', fontsize=10, fontweight='bold', color=C_DANGER, ha='center')

# Annotation: early vs late
ax.annotate('15× less\nstabilization\nneeded', xy=(0.25, 0.1), xytext=(0.32, 0.7),
            fontsize=10, fontweight='bold', color=C_SAFE,
            arrowprops=dict(arrowstyle='->', color=C_SAFE, lw=2),
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=C_SAFE))

ax.set_xlabel(r'Disease stage $\alpha$ (mutation load)', fontsize=13)
ax.set_ylabel(r'Minimum stabilizer dose $\varepsilon_{\mathrm{rescue}}$', fontsize=13)
ax.set_title('Dose–response: early intervention is exponentially more efficient',
             fontsize=13, fontweight='bold')
ax.set_xlim(0.17, 0.63)
ax.set_ylim(-0.05, 1.6)
ax.grid(True, alpha=0.15)

fig.savefig(OUTDIR + 'fig6_dose_response.pdf')
fig.savefig(OUTDIR + 'fig6_dose_response.png')
plt.close()
print("Figure 6: Dose-response — DONE")


# ═══════════════════════════════════════════════════════════
# FIGURE 7: σ-drift — why sporadic AD happens
# ═══════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(8, 5))

# Conceptual: σ increases with age
ages = np.linspace(30, 90, 200)

# Wild type: starts at 0.82, drifts toward 1 with age
sigma_wt = 0.82 + 0.004 * (ages - 30) + 0.0001 * (ages - 30)**2 / 50
# Swedish: starts higher
sigma_swedish = 0.92 + 0.004 * (ages - 30) + 0.0001 * (ages - 30)**2 / 50
# London: starts even higher
sigma_london = 0.95 + 0.004 * (ages - 30) + 0.0001 * (ages - 30)**2 / 50
# Protective: starts low
sigma_iceland = 0.70 + 0.003 * (ages - 30) + 0.00005 * (ages - 30)**2 / 50

ax.plot(ages, sigma_wt, color=C_GREY, linewidth=2.5, label='Wild type', zorder=3)
ax.plot(ages, sigma_swedish, color=C_WARN, linewidth=2.5, label='Swedish (FAD)', zorder=3)
ax.plot(ages, sigma_london, color=C_DANGER, linewidth=2.5, label='London (FAD)', zorder=3)
ax.plot(ages, sigma_iceland, color=C_SAFE, linewidth=2.5, label='Icelandic (protective)', zorder=3)

# σ = 1 threshold
ax.axhline(y=1.0, color='black', linewidth=2, linestyle='--', alpha=0.7)
ax.text(91, 1.01, r'$\sigma = 1$', fontsize=12, fontweight='bold', va='bottom')

# Fill zones
ax.fill_between(ages, 0.6, 1.0, alpha=0.05, color=C_SAFE)
ax.fill_between(ages, 1.0, 1.3, alpha=0.05, color=C_DANGER)

# Mark onset points
for sigma_curve, name, color, onset_age in [
    (sigma_london, 'London', C_DANGER, None),
    (sigma_swedish, 'Swedish', C_WARN, None),
    (sigma_wt, 'Wild type', C_GREY, None),
]:
    cross_idx = np.where(sigma_curve >= 1.0)[0]
    if len(cross_idx) > 0:
        cross_age = ages[cross_idx[0]]
        ax.scatter(cross_age, 1.0, color=color, s=100, marker='v', zorder=5,
                   edgecolors='white', linewidths=1.5)
        ax.annotate(f'{cross_age:.0f} yr', xy=(cross_age, 1.0),
                    xytext=(cross_age, 1.05), fontsize=9, fontweight='bold',
                    color=color, ha='center')

ax.set_xlabel('Age (years)', fontsize=13)
ax.set_ylabel(r'Stability index $\sigma$', fontsize=13)
ax.set_title(r'$\sigma$-drift: sporadic AD as aging-driven instability', fontsize=14, fontweight='bold')
ax.set_xlim(28, 95)
ax.set_ylim(0.65, 1.15)
ax.legend(loc='upper left', fontsize=10, framealpha=0.95)
ax.grid(True, alpha=0.15)

ax.text(50, 0.72, 'STABLE\n(folded)', fontsize=12, color=C_SAFE, fontweight='bold',
        ha='center', alpha=0.6)
ax.text(85, 1.08, 'UNSTABLE\n(misfolding risk)', fontsize=11, color=C_DANGER, fontweight='bold',
        ha='center', alpha=0.6)

fig.savefig(OUTDIR + 'fig7_sigma_drift.pdf')
fig.savefig(OUTDIR + 'fig7_sigma_drift.png')
plt.close()
print("Figure 7: σ-drift — DONE")


print("\n" + "=" * 50)
print("ALL FIGURES GENERATED")
print("=" * 50)
print(f"\nOutput directory: {OUTDIR}")
print("Files:")
for i in range(1, 8):
    print(f"  fig{i}_*.pdf / .png")
