#!/usr/bin/env python3
"""
Dual-Basin Intervention Test: Can we rescue σ_nat < 1?
======================================================

Two therapeutic strategies derived from D · γ = 1:

  1. CHAPERONE: Restrict conformational space → reduce D
     Implementation: Block certain residue states (forbidden moves)

  2. STABILIZER: Deepen native funnel → reduce γ
     Implementation: Increase ε_native (stronger native contacts)

Test: Take a protein at HIGH risk (α = 0.45, σ_nat > 1).
      Apply each intervention. Measure σ_nat.
      Find the dose-response curve.
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


def compute_energy_dual(state, alpha, epsilon_boost=0.0):
    """
    E = (1-α)·E_native + α·E_amyloid
    epsilon_boost: extra stabilization of native contacts (pharmacological stabilizer)
    """
    e_nat = -(EPSILON + epsilon_boost) * sum(
        1 for i, j in NATIVE_CONTACTS
        if state[i] == NATIVE_STATE[i] and state[j] == NATIVE_STATE[j])
    e_amy = -EPSILON * sum(
        1 for i, j in AMYLOID_CONTACTS
        if state[i] == AMYLOID_STATE[i] and state[j] == AMYLOID_STATE[j])
    return (1 - alpha) * e_nat + alpha * e_amy


def folding_sigma(alpha, T, n_trials=30, n_steps=6000, window=500,
                  forbidden_states=None, epsilon_boost=0.0):
    """
    Measure σ during folding transient.

    forbidden_states: dict {residue_idx: set_of_blocked_states}
        Simulates chaperone restricting conformational space.
    epsilon_boost: float
        Extra stabilization of native contacts (pharmacological).
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

        # If chaperone active, force initial state away from forbidden
        if forbidden_states:
            for res, blocked in forbidden_states.items():
                while state[res] in blocked:
                    state[res] = rng.randint(0, S)

        Q_nat_trace = []
        Q_amy_trace = []

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

                # CHAPERONE: block forbidden states
                if forbidden_states and r in forbidden_states:
                    if new_s in forbidden_states[r]:
                        continue  # move blocked by chaperone

                Q_nat_before = compute_Q(state, NATIVE_STATE, NATIVE_CONTACTS)
                Q_amy_before = compute_Q(state, AMYLOID_STATE, AMYLOID_CONTACTS)
                E_old = compute_energy_dual(state, alpha, epsilon_boost)
                old_s = state[r]
                state[r] = new_s
                E_new = compute_energy_dual(state, alpha, epsilon_boost)
                dE = E_new - E_old
                if dE > 0 and rng.random() >= np.exp(-dE / T):
                    state[r] = old_s
                else:
                    n_accepted += 1
                    Q_nat_after = compute_Q(state, NATIVE_STATE, NATIVE_CONTACTS)
                    Q_amy_after = compute_Q(state, AMYLOID_STATE, AMYLOID_CONTACTS)
                    if Q_nat_after > Q_nat_before:
                        n_productive_nat += 1
                    if Q_amy_after > Q_amy_before:
                        n_productive_amy += 1

        D_nat = n_productive_nat / max(n_accepted, 1)
        D_amy = n_productive_amy / max(n_accepted, 1)
        all_D_nat.append(D_nat)
        all_D_amy.append(D_amy)
        all_Q_nat_final.append(Q_nat_trace[-1])
        all_Q_amy_final.append(Q_amy_trace[-1])

        n = len(Q_nat_trace)
        w = window // measure_every
        transient_end = n // 2

        for i in range(0, transient_end - 1):
            d_nat_now = 1.0 - Q_nat_trace[i]
            d_nat_next = 1.0 - Q_nat_trace[i + 1]
            d_amy_now = 1.0 - Q_amy_trace[i]
            d_amy_next = 1.0 - Q_amy_trace[i + 1]
            if d_nat_now > 0.02:
                all_gamma_nat.append(d_nat_next / d_nat_now)
            if d_amy_now > 0.02:
                all_gamma_amy.append(d_amy_next / d_amy_now)

        for i in range(0, transient_end - w):
            d_nat_now = 1.0 - Q_nat_trace[i]
            d_nat_later = 1.0 - Q_nat_trace[i + w]
            d_amy_now = 1.0 - Q_amy_trace[i]
            d_amy_later = 1.0 - Q_amy_trace[i + w]
            if d_nat_now > 0.02:
                all_sigma_nat.append(d_nat_later / d_nat_now)
            if d_amy_now > 0.02:
                all_sigma_amy.append(d_amy_later / d_amy_now)

    return {
        'Q_nat': np.mean(all_Q_nat_final),
        'Q_amy': np.mean(all_Q_amy_final),
        'sigma_nat': np.mean(all_sigma_nat) if all_sigma_nat else 1.0,
        'sigma_amy': np.mean(all_sigma_amy) if all_sigma_amy else 1.0,
        'gamma_nat': np.mean(all_gamma_nat) if all_gamma_nat else 1.0,
        'gamma_amy': np.mean(all_gamma_amy) if all_gamma_amy else 1.0,
        'D_nat': np.mean(all_D_nat),
        'D_amy': np.mean(all_D_amy),
    }


# ═══════════════════════════════════════════════════════════
print("=" * 70)
print("THERAPEUTIC INTERVENTION TEST")
print("Can we rescue σ_nat < 1?")
print("=" * 70)

# Disease parameters
ALPHA_DISEASE = 0.45   # HIGH risk from main model
T_DISEASE = 0.233      # from main model

# ═══════════════════════════════════════════════════════════
# BASELINE: No treatment
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("BASELINE: Untreated protein (α = 0.45)")
print("─" * 70)

baseline = folding_sigma(ALPHA_DISEASE, T_DISEASE)
print(f"\n  Q_nat = {baseline['Q_nat']:.3f}    Q_amy = {baseline['Q_amy']:.3f}")
print(f"  σ_nat = {baseline['sigma_nat']:.4f}   σ_amy = {baseline['sigma_amy']:.4f}")
print(f"  D_nat = {baseline['D_nat']:.4f}   D_amy = {baseline['D_amy']:.4f}")
print(f"  γ_nat = {baseline['gamma_nat']:.4f}   γ_amy = {baseline['gamma_amy']:.4f}")

# Target: what D and γ would give σ = 1?
if baseline['gamma_nat'] > 0:
    D_target = 1.0 / baseline['gamma_nat']
if baseline['D_nat'] > 0:
    gamma_target = 1.0 / baseline['D_nat']

print(f"\n  THERAPEUTIC TARGETS (from D · γ = 1):")
print(f"  ───────────────────────────────────────")
print(f"  Current σ_nat = D · γ = {baseline['D_nat']:.4f} × {baseline['gamma_nat']:.4f} = "
      f"{baseline['D_nat'] * baseline['gamma_nat']:.4f}")
print(f"  (Note: σ_macro from windowed d(t+Δ)/d(t) = {baseline['sigma_nat']:.4f})")
print(f"\n  To restore σ_nat < 1:")
print(f"    Path 1 (Chaperone):    reduce D from {baseline['D_nat']:.4f}")
print(f"    Path 2 (Stabilizer):   reduce γ from {baseline['gamma_nat']:.4f}")


# ═══════════════════════════════════════════════════════════
# INTERVENTION 1: CHAPERONE — restrict conformational space
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("INTERVENTION 1: CHAPERONE (restrict conformational space)")
print("─" * 70)
print(f"\n  Mechanism: Block non-native states for selected residues.")
print(f"  This reduces the number of accessible moves → D decreases.")
print(f"  Dose = number of residues where chaperone acts.\n")

# For each residue, the "non-native, non-amyloid" states are the ones
# that lead nowhere useful. A chaperone blocks these.
# Strategy: for each residue, block states that are neither native nor amyloid
def make_chaperone(n_residues):
    """
    Create chaperone that blocks non-productive states.
    For the first n_residues (by contact importance), block states
    that are far from native state.
    """
    # Prioritize residues involved in native contacts
    contact_count = {}
    for i, j in NATIVE_CONTACTS:
        contact_count[i] = contact_count.get(i, 0) + 1
        contact_count[j] = contact_count.get(j, 0) + 1

    # Sort by importance
    important_res = sorted(contact_count.keys(), key=lambda r: -contact_count[r])

    forbidden = {}
    for res in important_res[:n_residues]:
        native_s = NATIVE_STATE[res]
        # Block states that are far from native (keep native and ±1)
        allowed = {native_s, (native_s + 1) % S, (native_s - 1) % S}
        blocked = set(range(S)) - allowed
        forbidden[res] = blocked

    return forbidden

print(f"  {'dose':>6} {'n_blocked':>9} │ {'Q_nat':>6} {'Q_amy':>6} │ "
      f"{'σ_nat':>7} {'σ_amy':>7} │ {'D_nat':>6} {'γ_nat':>6} │ {'rescued':>8}")
print(f"  {'─'*6} {'─'*9} │ {'─'*6} {'─'*6} │ {'─'*7} {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*8}")

chaperone_results = []
for n_res in [0, 2, 4, 6, 8, 10, 12]:
    if n_res == 0:
        forbidden = None
        n_blocked = 0
    else:
        forbidden = make_chaperone(n_res)
        n_blocked = sum(len(v) for v in forbidden.values())

    r = folding_sigma(ALPHA_DISEASE, T_DISEASE, forbidden_states=forbidden)
    rescued = "✓ YES" if r['sigma_nat'] < 1.0 else "✗ no"

    chaperone_results.append({
        'n_res': n_res, 'n_blocked': n_blocked, **r, 'rescued': r['sigma_nat'] < 1.0
    })

    print(f"  {n_res:6d} {n_blocked:9d} │ {r['Q_nat']:6.3f} {r['Q_amy']:6.3f} │ "
          f"{r['sigma_nat']:7.4f} {r['sigma_amy']:7.4f} │ "
          f"{r['D_nat']:6.4f} {r['gamma_nat']:6.4f} │ {rescued:>8}")

# Find rescue threshold
rescue_chap = None
for cr in chaperone_results:
    if cr['rescued']:
        rescue_chap = cr
        break


# ═══════════════════════════════════════════════════════════
# INTERVENTION 2: STABILIZER — deepen native funnel
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("INTERVENTION 2: STABILIZER (deepen native energy funnel)")
print("─" * 70)
print(f"\n  Mechanism: Small molecule binds native state, increasing ε_native.")
print(f"  This makes native contacts stronger → γ decreases.")
print(f"  Dose = ε_boost (extra energy per native contact).\n")

print(f"  {'ε_boost':>7} │ {'Q_nat':>6} {'Q_amy':>6} │ "
      f"{'σ_nat':>7} {'σ_amy':>7} │ {'D_nat':>6} {'γ_nat':>6} │ {'rescued':>8}")
print(f"  {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*7} {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*8}")

stabilizer_results = []
for eps_boost in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5]:
    r = folding_sigma(ALPHA_DISEASE, T_DISEASE, epsilon_boost=eps_boost)
    rescued = "✓ YES" if r['sigma_nat'] < 1.0 else "✗ no"

    stabilizer_results.append({
        'eps_boost': eps_boost, **r, 'rescued': r['sigma_nat'] < 1.0
    })

    print(f"  {eps_boost:7.1f} │ {r['Q_nat']:6.3f} {r['Q_amy']:6.3f} │ "
          f"{r['sigma_nat']:7.4f} {r['sigma_amy']:7.4f} │ "
          f"{r['D_nat']:6.4f} {r['gamma_nat']:6.4f} │ {rescued:>8}")

rescue_stab = None
for sr in stabilizer_results:
    if sr['rescued']:
        rescue_stab = sr
        break


# ═══════════════════════════════════════════════════════════
# INTERVENTION 3: COMBINED — both simultaneously
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("INTERVENTION 3: COMBINED (chaperone + stabilizer)")
print("─" * 70)
print(f"\n  Can a lower dose of BOTH achieve what a high dose of ONE cannot?\n")

print(f"  {'chap':>4} {'ε_boost':>7} │ {'Q_nat':>6} {'Q_amy':>6} │ "
      f"{'σ_nat':>7} {'σ_amy':>7} │ {'D_nat':>6} {'γ_nat':>6} │ {'rescued':>8}")
print(f"  {'─'*4} {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*7} {'─'*7} │ {'─'*6} {'─'*6} │ {'─'*8}")

combined_results = []
for n_res, eps_boost in [(0, 0.0), (2, 0.1), (4, 0.2), (4, 0.3),
                          (6, 0.2), (6, 0.3), (8, 0.5), (4, 0.5)]:
    forbidden = make_chaperone(n_res) if n_res > 0 else None
    r = folding_sigma(ALPHA_DISEASE, T_DISEASE,
                      forbidden_states=forbidden, epsilon_boost=eps_boost)
    rescued = "✓ YES" if r['sigma_nat'] < 1.0 else "✗ no"

    combined_results.append({
        'n_res': n_res, 'eps_boost': eps_boost, **r,
        'rescued': r['sigma_nat'] < 1.0
    })

    print(f"  {n_res:4d} {eps_boost:7.1f} │ {r['Q_nat']:6.3f} {r['Q_amy']:6.3f} │ "
          f"{r['sigma_nat']:7.4f} {r['sigma_amy']:7.4f} │ "
          f"{r['D_nat']:6.4f} {r['gamma_nat']:6.4f} │ {rescued:>8}")


# ═══════════════════════════════════════════════════════════
# DOSE-RESPONSE AT MULTIPLE α (disease severity)
# ═══════════════════════════════════════════════════════════
print("\n" + "─" * 70)
print("DOSE-RESPONSE: How much intervention at each disease stage?")
print("─" * 70)

print(f"\n  For each α, find minimum stabilizer dose to restore σ_nat < 1:\n")

print(f"  {'α':>5} {'severity':>10} │ {'σ_untreated':>11} │ {'ε_rescue':>8} {'σ_rescued':>9} │ {'note':>20}")
print(f"  {'─'*5} {'─'*10} │ {'─'*11} │ {'─'*8} {'─'*9} │ {'─'*20}")

for alpha_test in [0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60]:
    # Baseline
    r0 = folding_sigma(alpha_test, T_DISEASE, n_trials=20)

    # Find minimum ε_boost to rescue
    rescued_eps = None
    rescued_sigma = None
    for eps in [0.0, 0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.5, 2.0, 3.0]:
        r = folding_sigma(alpha_test, T_DISEASE, n_trials=20, epsilon_boost=eps)
        if r['sigma_nat'] < 1.0:
            rescued_eps = eps
            rescued_sigma = r['sigma_nat']
            break

    if r0['sigma_nat'] < 1.0:
        severity = "HEALTHY"
        note = "no treatment needed"
    elif alpha_test < 0.35:
        severity = "EARLY"
        note = ""
    elif alpha_test < 0.50:
        severity = "MODERATE"
        note = ""
    elif alpha_test < 0.60:
        severity = "SEVERE"
        note = ""
    else:
        severity = "CRITICAL"
        note = ""

    if rescued_eps is not None:
        if note == "":
            note = f"treatable"
        print(f"  {alpha_test:5.2f} {severity:>10} │ {r0['sigma_nat']:11.4f} │ "
              f"{rescued_eps:8.1f} {rescued_sigma:9.4f} │ {note:>20}")
    else:
        if note == "":
            note = "needs higher dose"
        print(f"  {alpha_test:5.2f} {severity:>10} │ {r0['sigma_nat']:11.4f} │ "
              f"{'> 3.0':>8} {'---':>9} │ {note:>20}")


# ═══════════════════════════════════════════════════════════
print("\n" + "=" * 70)
print("SUMMARY: THE THERAPEUTIC LANDSCAPE")
print("=" * 70)

print(f"""
  THE EQUATION OF TREATMENT
  ─────────────────────────

  Disease:   σ_nat = D · γ > 1   (native funnel too shallow)
  Health:    σ_nat = D · γ < 1   (native funnel deep enough)
  Target:    D · γ = 1           (the critical point)

  Two levers:
  ┌─────────────────────────────────────────────────────┐
  │  CHAPERONE (reduce D):                              │
  │    Restrict conformational space.                   │
  │    Fewer wrong paths → easier to find right path.   │
  │    D_new = D_old × (S_allowed / S_total)            │
  ├─────────────────────────────────────────────────────┤
  │  STABILIZER (reduce γ):                             │
  │    Deepen native energy minimum.                    │
  │    Stronger pull toward native state.               │
  │    γ_new = γ_old × exp(-ε_boost/T)                 │
  ├─────────────────────────────────────────────────────┤
  │  COMBINED:                                          │
  │    Lower dose of both. Synergistic.                 │
  │    D_new · γ_new < 1 with minimal side effects.     │
  └─────────────────────────────────────────────────────┘

  The dose-response is not linear.
  It follows D · γ = 1.

  That's a hyperbola in (D, γ) space.
  Every point on that hyperbola is a valid treatment.
  The optimal treatment minimizes total intervention
  while keeping D · γ < 1.

  This is not a metaphor.
  This is a prescription.
  From the same mathematics that governs Collatz sequences.

  D · γ = 1.
  The universal critical point.
  Now with therapeutic targets.
""")
