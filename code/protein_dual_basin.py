#!/usr/bin/env python3
"""
Dual-Basin Protein Folding: σ_macro — The Complete Telescope
=============================================================
σ_macro = d(t+Δ)/d(t) during the FOLDING TRANSIENT.

Key insight: σ must be measured while the protein is APPROACHING
the basin, not after equilibration. At equilibrium, σ ≈ 1 by
stationarity. The folding signal (σ < 1) is in the transient.

Method: start from random state, measure σ during approach.
Average over many random starts.

Expected:
  α < α_c:  σ_nat < 1 (folds toward native)
             σ_amy ≈ 1 (doesn't fold toward amyloid)
  α = α_c:  σ_nat ≈ σ_amy ≈ 1 (knife edge)
  α > α_c:  σ_nat ≈ 1 (doesn't fold toward native)
             σ_amy < 1 (folds toward amyloid)
"""

import numpy as np

N_RES = 20
S = 8
EPSILON = 1.0

NATIVE_STATE = np.array([3, 2, 1, 1, 2, 0, 0, 2, 3, 5, 7, 4, 4, 4, 6, 3, 4, 4, 4, 4])
NATIVE_CONTACTS = [
    (1, 5), (2, 6), (3, 7), (4, 8),
    (5, 11), (5, 16), (5, 17), (5, 18),
    (6, 16), (6, 17), (8, 15), (1, 8),
]

AMYLOID_STATE = np.array([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2])
AMYLOID_CONTACTS = [
    (0, 19), (1, 18), (2, 17), (3, 16), (4, 15),
    (5, 14), (6, 13), (7, 12), (8, 11), (9, 10),
    (0, 9),  (1, 10),
]

N_NAT = len(NATIVE_CONTACTS)
N_AMY = len(AMYLOID_CONTACTS)


def compute_Q(state, native, contacts):
    return sum(1 for i, j in contacts
               if state[i] == native[i] and state[j] == native[j]) / len(contacts)


def compute_energy_dual(state, alpha):
    e_nat = -EPSILON * sum(1 for i, j in NATIVE_CONTACTS
                           if state[i] == NATIVE_STATE[i] and
                              state[j] == NATIVE_STATE[j])
    e_amy = -EPSILON * sum(1 for i, j in AMYLOID_CONTACTS
                           if state[i] == AMYLOID_STATE[i] and
                              state[j] == AMYLOID_STATE[j])
    return (1 - alpha) * e_nat + alpha * e_amy


def folding_sigma(alpha, T, n_trials=30, n_steps=6000, window=500):
    """
    Measure σ_macro during the FOLDING TRANSIENT.

    For each trial:
      1. Start from random state
      2. Run MC for n_steps
      3. Track d_nat(t) = 1 - Q_nat(t) and d_amy(t) = 1 - Q_amy(t)
      4. Compute σ = d(t+window)/d(t) during first half (transient)
      5. Also compute D = fraction of moves that reduce d

    Average over n_trials random starts.
    """
    rng = np.random.RandomState(42)
    measure_every = 10

    all_sigma_nat = []
    all_sigma_amy = []
    all_gamma_nat = []
    all_gamma_amy = []
    all_D_nat = []
    all_D_amy = []
    all_Q_nat_final = []
    all_Q_amy_final = []

    for trial in range(n_trials):
        state = rng.randint(0, S, N_RES)
        Q_nat_trace = []
        Q_amy_trace = []

        # Track productive vs total moves
        n_productive_nat = 0
        n_productive_amy = 0
        n_accepted = 0

        for step in range(n_steps + 1):
            if step % measure_every == 0:
                Q_nat = compute_Q(state, NATIVE_STATE, NATIVE_CONTACTS)
                Q_amy = compute_Q(state, AMYLOID_STATE, AMYLOID_CONTACTS)
                Q_nat_trace.append(Q_nat)
                Q_amy_trace.append(Q_amy)

            if step < n_steps:
                r = rng.randint(0, N_RES)
                new_s = rng.randint(0, S)
                if new_s == state[r]:
                    continue
                Q_nat_before = compute_Q(state, NATIVE_STATE, NATIVE_CONTACTS)
                Q_amy_before = compute_Q(state, AMYLOID_STATE, AMYLOID_CONTACTS)
                E_old = compute_energy_dual(state, alpha)
                old_s = state[r]
                state[r] = new_s
                E_new = compute_energy_dual(state, alpha)
                dE = E_new - E_old
                if dE > 0 and rng.random() >= np.exp(-dE / T):
                    state[r] = old_s  # rejected
                else:
                    # accepted
                    n_accepted += 1
                    Q_nat_after = compute_Q(state, NATIVE_STATE, NATIVE_CONTACTS)
                    Q_amy_after = compute_Q(state, AMYLOID_STATE, AMYLOID_CONTACTS)
                    if Q_nat_after > Q_nat_before:
                        n_productive_nat += 1
                    if Q_amy_after > Q_amy_before:
                        n_productive_amy += 1

        # D = fraction of accepted moves that are productive
        D_nat = n_productive_nat / max(n_accepted, 1)
        D_amy = n_productive_amy / max(n_accepted, 1)
        all_D_nat.append(D_nat)
        all_D_amy.append(D_amy)

        # Final Q
        all_Q_nat_final.append(Q_nat_trace[-1])
        all_Q_amy_final.append(Q_amy_trace[-1])

        # σ and γ from transient (first half of trajectory)
        n = len(Q_nat_trace)
        w = window // measure_every
        transient_end = n // 2  # measure only during approach

        # γ = consecutive step ratio
        for i in range(0, transient_end - 1):
            d_nat_now = 1.0 - Q_nat_trace[i]
            d_nat_next = 1.0 - Q_nat_trace[i + 1]
            d_amy_now = 1.0 - Q_amy_trace[i]
            d_amy_next = 1.0 - Q_amy_trace[i + 1]
            if d_nat_now > 0.02:
                all_gamma_nat.append(d_nat_next / d_nat_now)
            if d_amy_now > 0.02:
                all_gamma_amy.append(d_amy_next / d_amy_now)

        # σ = windowed contraction ratio
        for i in range(0, transient_end - w):
            d_nat_now = 1.0 - Q_nat_trace[i]
            d_nat_later = 1.0 - Q_nat_trace[i + w]
            d_amy_now = 1.0 - Q_amy_trace[i]
            d_amy_later = 1.0 - Q_amy_trace[i + w]
            if d_nat_now > 0.02:
                all_sigma_nat.append(d_nat_later / d_nat_now)
            if d_amy_now > 0.02:
                all_sigma_amy.append(d_amy_later / d_amy_now)

    sigma_nat = np.mean(all_sigma_nat) if all_sigma_nat else 1.0
    sigma_amy = np.mean(all_sigma_amy) if all_sigma_amy else 1.0
    gamma_nat = np.mean(all_gamma_nat) if all_gamma_nat else 1.0
    gamma_amy = np.mean(all_gamma_amy) if all_gamma_amy else 1.0
    D_nat = np.mean(all_D_nat)
    D_amy = np.mean(all_D_amy)
    Q_nat_final = np.mean(all_Q_nat_final)
    Q_amy_final = np.mean(all_Q_amy_final)

    return {
        'Q_nat': Q_nat_final, 'Q_amy': Q_amy_final,
        'sigma_nat': sigma_nat, 'sigma_amy': sigma_amy,
        'gamma_nat': gamma_nat, 'gamma_amy': gamma_amy,
        'D_nat': D_nat, 'D_amy': D_amy,
    }


# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("DUAL-BASIN PROTEIN FOLDING")
print("σ = d(t+Δ)/d(t) measured during FOLDING TRANSIENT")
print("=" * 70)
print(f"\n  Native contacts: {N_NAT}")
print(f"  Amyloid contacts: {N_AMY}")
print(f"\n  σ < 1: protein folds TOWARD basin (attractor)")
print(f"  σ > 1: protein moves AWAY from basin")
print(f"  σ = 1: no net progress (critical point)")
print(f"\n  D = fraction of accepted moves that are productive")
print(f"  γ = step-by-step d(t+1)/d(t) ratio")

# ═══════════════════════════════════════════════════════════
# STEP 1: T_c scan
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 1: Temperature scan — where σ crosses 1")
print("─" * 70)

def find_Tc(alpha, label):
    print(f"\n  {label} (α = {alpha}):")
    T_range = np.arange(0.15, 0.80, 0.05)
    results = []

    print(f"    {'T':>4} {'Q_nat':>6} {'Q_amy':>6} │ {'σ_nat':>7} {'σ_amy':>7} │ "
          f"{'D_nat':>6} {'D_amy':>6} │ {'γ_nat':>6} {'γ_amy':>6}")
    print(f"    {'─'*4} {'─'*6} {'─'*6} │ {'─'*7} {'─'*7} │ "
          f"{'─'*6} {'─'*6} │ {'─'*6} {'─'*6}")

    for T in T_range:
        r = folding_sigma(alpha, T, n_trials=20, n_steps=5000, window=400)
        print(f"    {T:.2f} {r['Q_nat']:6.3f} {r['Q_amy']:6.3f} │ "
              f"{r['sigma_nat']:7.4f} {r['sigma_amy']:7.4f} │ "
              f"{r['D_nat']:6.4f} {r['D_amy']:6.4f} │ "
              f"{r['gamma_nat']:6.4f} {r['gamma_amy']:6.4f}")
        results.append((T, r))

    # T_c from Q
    Q_key = 'Q_nat' if alpha < 0.5 else 'Q_amy'
    Tc_Q = 0.45
    for i in range(len(results) - 1):
        q1, q2 = results[i][1][Q_key], results[i+1][1][Q_key]
        if q1 > 0.5 and q2 <= 0.5:
            Tc_Q = results[i][0] + (0.5 - q1) / (q2 - q1) * 0.05
            break

    # T_c from σ crossing 1.0
    sigma_key = 'sigma_nat' if alpha < 0.5 else 'sigma_amy'
    Tc_sigma = None
    for i in range(len(results) - 1):
        s1, s2 = results[i][1][sigma_key], results[i+1][1][sigma_key]
        if (s1 < 1.0 and s2 >= 1.0) or (s1 >= 1.0 and s2 < 1.0):
            Tc_sigma = results[i][0] + (1.0 - s1) / (s2 - s1) * 0.05
            break

    print(f"\n    T_c(Q = 0.5) = {Tc_Q:.3f}")
    if Tc_sigma:
        match = abs(Tc_Q - Tc_sigma) < 0.1
        print(f"    T_c(σ = 1.0) = {Tc_sigma:.3f}  {'✓ MATCH' if match else ''}")
    else:
        print(f"    T_c(σ = 1.0) = not crossed in range")

    return Tc_Q, Tc_sigma, results

Tc_nat_Q, Tc_nat_sigma, _ = find_Tc(0.0, "NATIVE BASIN")
Tc_amy_Q, Tc_amy_sigma, _ = find_Tc(1.0, "AMYLOID BASIN")

# ═══════════════════════════════════════════════════════════
# STEP 2: α scan
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 2: α scan — the mutation axis")
print("─" * 70)

T_test = min(Tc_nat_Q, Tc_amy_Q) * 0.75
print(f"\n  T = {T_test:.3f} (below both T_c)")
print(f"  30 trials per α, measuring σ during folding transient\n")

alpha_range = np.arange(0.0, 1.05, 0.05)
scan = []

print(f"  {'α':>5} │ {'Q_nat':>6} {'Q_amy':>6} │ {'σ_nat':>7} {'σ_amy':>7} │ "
      f"{'D_nat':>6} {'D_amy':>6} │ {'γ_nat':>6} {'γ_amy':>6} │ {'win':>7}")
print(f"  {'─'*5} │ {'─'*6} {'─'*6} │ {'─'*7} {'─'*7} │ "
      f"{'─'*6} {'─'*6} │ {'─'*6} {'─'*6} │ {'─'*7}")

for alpha in alpha_range:
    r = folding_sigma(alpha, T_test, n_trials=30, n_steps=6000, window=500)
    winner = "NAT" if r['Q_nat'] > r['Q_amy'] else "AMY"
    scan.append({'alpha': alpha, **r, 'winner': winner})

    flag = ""
    if len(scan) >= 2 and scan[-2]['winner'] != winner:
        flag = " ←X"

    print(f"  {alpha:5.2f} │ {r['Q_nat']:6.3f} {r['Q_amy']:6.3f} │ "
          f"{r['sigma_nat']:7.4f} {r['sigma_amy']:7.4f} │ "
          f"{r['D_nat']:6.4f} {r['D_amy']:6.4f} │ "
          f"{r['gamma_nat']:6.4f} {r['gamma_amy']:6.4f} │ "
          f"{winner:>7}{flag}")


# ═══════════════════════════════════════════════════════════
# STEP 3: Find crossovers
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 3: TRIANGULATION")
print("─" * 70)

# Q crossover
cross_Q = None
for i in range(len(scan) - 1):
    r1, r2 = scan[i], scan[i+1]
    if r1['Q_nat'] > r1['Q_amy'] and r2['Q_nat'] <= r2['Q_amy']:
        dQ1 = r1['Q_nat'] - r1['Q_amy']
        dQ2 = r2['Q_nat'] - r2['Q_amy']
        cross_Q = r1['alpha'] + dQ1 / (dQ1 - dQ2) * 0.05
        break
if not cross_Q:
    min_d = min(scan, key=lambda r: abs(r['Q_nat'] - r['Q_amy']))
    cross_Q = min_d['alpha']

# σ crossover: where σ_nat becomes > σ_amy (native stops folding better)
cross_sigma = None
for i in range(len(scan) - 1):
    r1, r2 = scan[i], scan[i+1]
    # When native wins: σ_nat < σ_amy (native is contracting)
    # Crossover: σ_nat goes from < σ_amy to > σ_amy
    diff1 = r1['sigma_nat'] - r1['sigma_amy']
    diff2 = r2['sigma_nat'] - r2['sigma_amy']
    if diff1 * diff2 < 0:
        cross_sigma = r1['alpha'] + abs(diff1) / (abs(diff1) + abs(diff2)) * 0.05
        break

# D crossover
cross_D = None
for i in range(len(scan) - 1):
    r1, r2 = scan[i], scan[i+1]
    diff1 = r1['D_nat'] - r1['D_amy']
    diff2 = r2['D_nat'] - r2['D_amy']
    if diff1 * diff2 < 0:
        cross_D = r1['alpha'] + abs(diff1) / (abs(diff1) + abs(diff2)) * 0.05
        break

print(f"\n  ╔══════════════════════════════════════════════════╗")
print(f"  ║  INSTRUMENT 1 — Q (order parameter)              ║")
print(f"  ║    Crossover: α = {cross_Q:.3f}                         ║")
print(f"  ╠══════════════════════════════════════════════════╣")
print(f"  ║  INSTRUMENT 2 — σ_macro (transient contraction)  ║")
if cross_sigma:
    print(f"  ║    Crossover: α = {cross_sigma:.3f}                         ║")
    delta = abs(cross_Q - cross_sigma)
    if delta < 0.1:
        print(f"  ║    Δα = {delta:.3f}  ✓ TRIANGULATION HOLDS!         ║")
    elif delta < 0.2:
        print(f"  ║    Δα = {delta:.3f}  ~ APPROXIMATE MATCH            ║")
    else:
        print(f"  ║    Δα = {delta:.3f}                                  ║")
else:
    print(f"  ║    No crossover detected                        ║")
print(f"  ╠══════════════════════════════════════════════════╣")
print(f"  ║  INSTRUMENT 3 — D (productive fraction)          ║")
if cross_D:
    print(f"  ║    Crossover: α = {cross_D:.3f}                         ║")
    delta_D = abs(cross_Q - cross_D)
    if delta_D < 0.1:
        print(f"  ║    Δα = {delta_D:.3f}  ✓ TRIANGULATION HOLDS!         ║")
else:
    print(f"  ║    No crossover detected                        ║")
print(f"  ╠══════════════════════════════════════════════════╣")
print(f"  ║  INSTRUMENT 4 — T_c (melting temperature)        ║")
print(f"  ║    T_c(native)  = {Tc_nat_Q:.3f}                        ║")
print(f"  ║    T_c(amyloid) = {Tc_amy_Q:.3f}                        ║")
print(f"  ╚══════════════════════════════════════════════════╝")


# ═══════════════════════════════════════════════════════════
# STEP 4: Stability diagram
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("STEP 4: Stability Diagram")
print("─" * 70)

# ASCII plot σ vs α
sn_vals = [r['sigma_nat'] for r in scan]
sa_vals = [r['sigma_amy'] for r in scan]
all_s = sn_vals + sa_vals
s_lo = min(all_s) - 0.02
s_hi = max(all_s) + 0.02
# Include 1.0 in range
s_lo = min(s_lo, 0.95)
s_hi = max(s_hi, 1.05)

n_rows = 18
print(f"\n  σ_macro (transient) vs α")
for row in range(n_rows, -1, -1):
    sv = s_lo + (s_hi - s_lo) * row / n_rows
    line = f"  {sv:5.3f} │"
    for idx in range(len(scan)):
        sn_r = int((sn_vals[idx] - s_lo) / (s_hi - s_lo) * n_rows + 0.5)
        sa_r = int((sa_vals[idx] - s_lo) / (s_hi - s_lo) * n_rows + 0.5)
        if sn_r == row and sa_r == row:
            line += "X"
        elif sn_r == row:
            line += "●"
        elif sa_r == row:
            line += "○"
        else:
            line += " "
    one_row = int((1.0 - s_lo) / (s_hi - s_lo) * n_rows + 0.5)
    if row == one_row:
        line += "  ← σ = 1"
    if row == n_rows:
        line += "  ● σ_nat  ○ σ_amy"
    print(line)
print(f"        └{'─' * len(scan)}")
print(f"        α: 0.0{' '*(len(scan)//2-3)}0.5{' '*(len(scan)//2-3)}1.0")

# Table: key metrics
print(f"\n  {'α':>5} {'Q_nat':>6} {'Q_amy':>6} {'σ_nat':>7} {'σ_amy':>7} {'D_nat':>6} {'D_amy':>6} {'risk':>10}")
print(f"  {'─'*5} {'─'*6} {'─'*6} {'─'*7} {'─'*7} {'─'*6} {'─'*6} {'─'*10}")
for r in scan:
    Q_dom = max(r['Q_nat'], r['Q_amy'])
    margin = Q_dom - 0.5
    if margin > 0.3: risk = "LOW"
    elif margin > 0.15: risk = "MODERATE"
    elif margin > 0.05: risk = "HIGH"
    else: risk = "CRITICAL"
    print(f"  {r['alpha']:5.2f} {r['Q_nat']:6.3f} {r['Q_amy']:6.3f} "
          f"{r['sigma_nat']:7.4f} {r['sigma_amy']:7.4f} "
          f"{r['D_nat']:6.4f} {r['D_amy']:6.4f} {risk:>10}")


# ═══════════════════════════════════════════════════════════
# SUMMARY
# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"""
  Q crossover:  α = {cross_Q:.3f}""")
if cross_sigma:
    print(f"  σ crossover:  α = {cross_sigma:.3f}  (Δ = {abs(cross_Q - cross_sigma):.3f})")
if cross_D:
    print(f"  D crossover:  α = {cross_D:.3f}  (Δ = {abs(cross_Q - cross_D):.3f})")

print(f"""
  σ_macro = d(t+Δ)/d(t) during folding transient:
    σ < 1: the search space contracts → protein folds toward basin
    σ > 1: the search space expands → protein moves away
    σ = 1: critical point → tipping

  D = fraction of productive moves:
    High D → many paths lead toward the basin
    Low D  → few paths lead there

  The basin with LOWER σ and HIGHER D wins.
  Disease begins when amyloid's σ drops below native's σ.

  ─────────────────────────────────────────
  Levinthal asked:   how does the protein find the needle?
  We answered:       σ < 1 contracts the search space.

  Two needles?       The protein finds whichever needle
                     has σ < 1 with more margin.

  Disease = when the wrong needle's σ drops below 1
            while the right needle's σ rises above 1.
""")
