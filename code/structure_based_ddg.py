#!/usr/bin/env python3
"""
Structure-based DDG computation for APP/Abeta variants
======================================================

Two independent methods:
  1. ESM-1v (sequence-based ML) — local, fast
  2. DynaMut2 (structure-based) — remote API

For Nature Communications paper: every number must be correct.

Author: Matthias Wurm / ForgottenForge
"""

import numpy as np
import csv
import os
import sys
import time
import requests
import io
import warnings
warnings.filterwarnings("ignore")

# ═══════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════

AB42_SEQ = "DAEFRHDSGYEVHHQKLVFFAEDVGSNKGAIIGLMVGGVVIA"
N_AB42 = 42
SIGMA_WT = 0.822  # from paper
R_KCAL = 0.001987  # kcal/(mol*K)
T_BODY = 310.0  # K (body temperature)

# PDB file (single model extracted from 1IYT NMR structure)
PDB_FILE = "/tmp/1IYT_model1.pdb"
PDB_CHAIN = "A"

# ═══════════════════════════════════════════════════════════
# VARIANT DEFINITIONS
# ═══════════════════════════════════════════════════════════

# 26 VUS from ClinVar (from supplementary Table S11.2)
# APP position -> Abeta position: pos_abeta = pos_APP - 671
# Only variants in Abeta region (APP 672-713) use N=42
# Variants outside Abeta (APP 714+) use full APP domain

VUS_VARIANTS = [
    # (APP_variant, APP_pos, wt_aa, mut_aa, region, paper_ddg, paper_sigma)
    ("G696R", 696, "G", "R", "Abeta", 2.87, 1.117),
    ("D678H", 678, "D", "H", "Abeta", 2.33, 1.094),
    ("G700E", 700, "G", "E", "Abeta", 2.21, 1.089),
    ("K687E", 687, "K", "E", "Abeta", 2.19, 1.088),
    ("G700D", 700, "G", "D", "Abeta", 1.94, 1.078),
    ("V695A", 695, "V", "A", "Abeta", 1.73, 1.069),
    ("A692T", 692, "A", "T", "Abeta", 1.52, 1.061),
    ("V689M", 689, "V", "M", "Abeta", 1.42, 1.056),
    ("V710F", 710, "V", "F", "Abeta", 1.42, 1.056),
    ("G696S", 696, "G", "S", "Abeta", 0.91, 1.036),
    ("G709S", 709, "G", "S", "Abeta", 0.91, 1.036),
    ("V711L", 711, "V", "L", "Abeta", 0.89, 1.035),
    ("K687R", 687, "K", "R", "Abeta", 0.73, 1.029),
    ("V695I", 695, "V", "I", "Abeta", 0.68, 1.027),
    ("V711I", 711, "V", "I", "Abeta", 0.68, 1.027),
    ("V710I", 710, "V", "I", "Abeta", 0.68, 1.027),
    ("E682G", 682, "E", "G", "Abeta", 0.35, 1.014),
    ("A692V", 692, "A", "V", "Abeta", 0.29, 1.011),
    ("L720R", 720, "L", "R", "APP", 3.06, 1.007),
    ("I716T", 716, "I", "T", "APP", 2.57, 1.005),
    # NOTE: G681A listed at APP 681 = Abeta pos 10 = Y (tyrosine), NOT G.
    # Actual G is at APP 680 = Abeta 9. Possible off-by-one in ClinVar annotation.
    # Excluding from structure-based analysis; retained with flag for paper table.
    ("G681A", 681, "G", "A", "Abeta_SKIP", 0.13, 1.005),
    ("T719P", 719, "T", "P", "APP", 2.31, 1.005),
    ("M722K", 722, "M", "K", "APP", 2.30, 1.005),
    ("V717A", 717, "V", "A", "APP", 1.73, 1.004),
    ("T719N", 719, "T", "N", "APP", 1.36, 1.003),
    ("G696V", 696, "G", "V", "Abeta", -0.08, 0.997),
]

# Known pathogenic mutations WITHIN Abeta42 (for calibration)
PATHOGENIC_VARIANTS = [
    # (name, APP_variant, APP_pos, wt_aa, mut_aa)
    ("Icelandic (protective)", "A673T", 673, "A", "T"),
    ("Arctic", "E693G", 693, "E", "G"),
    ("Dutch", "E693Q", 693, "E", "Q"),
    ("Iowa", "D694N", 694, "D", "N"),
    ("Flemish", "A692G", 692, "A", "G"),
    ("Italian", "E693K", 693, "E", "K"),
]

def app_to_abeta(app_pos):
    """Convert APP numbering to Abeta numbering (1-indexed)."""
    return app_pos - 671

def verify_sequence_position(app_pos, expected_wt):
    """Verify that the Abeta sequence has the expected WT amino acid at this position."""
    ab_pos = app_to_abeta(app_pos)
    if 1 <= ab_pos <= 42:
        actual = AB42_SEQ[ab_pos - 1]  # 0-indexed
        if actual != expected_wt:
            print(f"  WARNING: Position {app_pos} (Abeta {ab_pos}): "
                  f"expected {expected_wt}, found {actual} in sequence")
            return False
        return True
    return False  # outside Abeta

# ═══════════════════════════════════════════════════════════
# METHOD 1: ESM-1v
# ═══════════════════════════════════════════════════════════

def compute_esm1v_scores():
    """
    Compute ESM-1v log-likelihood ratios for all variants in Abeta42.

    LLR = log P(mutant | context) - log P(wildtype | context)
    Negative LLR = wildtype preferred = mutation is destabilizing.
    """
    print("\n" + "=" * 70)
    print("METHOD 1: ESM-1v (sequence-based)")
    print("=" * 70)

    import torch
    import esm

    print("  Loading ESM-1v model...")
    model, alphabet = esm.pretrained.esm1v_t33_650M_UR90S_1()
    model.eval()
    batch_converter = alphabet.get_batch_converter()

    data = [("abeta42", AB42_SEQ)]
    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    print(f"  Sequence: {AB42_SEQ}")
    print(f"  Length: {len(AB42_SEQ)}")
    print(f"  Token shape: {batch_tokens.shape}")

    with torch.no_grad():
        results = model(batch_tokens)
    logits = results["logits"]  # [1, seq_len+2, vocab_size]

    # Convert logits to log-probabilities
    log_probs = torch.log_softmax(logits, dim=-1)

    print(f"  Logits shape: {logits.shape}")
    print(f"  Vocabulary size: {logits.shape[-1]}")

    # Verify BOS/EOS positioning
    # ESM tokens: [BOS, D, A, E, F, R, ..., A, EOS]
    # So Abeta position i (1-indexed) -> token index i (BOS is 0)

    # Sanity check: verify WT amino acids match
    print("\n  Sanity check — WT amino acids at each position:")
    all_ok = True
    for i, aa in enumerate(AB42_SEQ):
        token_idx = i + 1  # +1 for BOS
        aa_token_id = alphabet.get_idx(aa)
        # The highest-scoring token should be close to the WT
        top_token = logits[0, token_idx].argmax().item()
        top_aa = alphabet.get_tok(top_token)
        if i < 5 or i >= 37:
            print(f"    Pos {i+1} ({aa}): token_idx={token_idx}, "
                  f"WT token_id={aa_token_id}, "
                  f"top predicted={top_aa}, "
                  f"log_prob(WT)={log_probs[0, token_idx, aa_token_id]:.3f}")

    esm_results = {}

    # Compute LLR for all Abeta-region variants
    all_variants = []

    # VUS variants in Abeta region
    for var_name, app_pos, wt, mut, region, _, _ in VUS_VARIANTS:
        if region == "Abeta":
            ab_pos = app_to_abeta(app_pos)
            all_variants.append((var_name, ab_pos, wt, mut, "VUS"))

    # Known pathogenic in Abeta
    for name, var_name, app_pos, wt, mut in PATHOGENIC_VARIANTS:
        ab_pos = app_to_abeta(app_pos)
        if 1 <= ab_pos <= 42:
            all_variants.append((f"{name} ({var_name})", ab_pos, wt, mut, "pathogenic"))

    print(f"\n  Computing LLR for {len(all_variants)} variants...")
    print(f"\n  {'Variant':<35} {'AbPos':>5} {'WT':>3} {'Mut':>3} {'LLR':>8} {'Type':>12}")
    print(f"  {'─'*35} {'─'*5} {'─'*3} {'─'*3} {'─'*8} {'─'*12}")

    for var_name, ab_pos, wt, mut, vtype in all_variants:
        # Verify WT
        actual_wt = AB42_SEQ[ab_pos - 1]
        if actual_wt != wt:
            print(f"  WARNING: {var_name}: expected WT={wt} at Abeta pos {ab_pos}, "
                  f"found {actual_wt}")
            continue

        token_idx = ab_pos  # BOS at 0, so pos 1 -> token 1
        wt_idx = alphabet.get_idx(wt)
        mut_idx = alphabet.get_idx(mut)

        # LLR = log P(mut) - log P(wt)
        llr = (log_probs[0, token_idx, mut_idx] - log_probs[0, token_idx, wt_idx]).item()

        esm_results[var_name] = {
            "ab_pos": ab_pos,
            "wt": wt,
            "mut": mut,
            "llr": llr,
            "type": vtype,
        }

        print(f"  {var_name:<35} {ab_pos:>5} {wt:>3} {mut:>3} {llr:>8.3f} {vtype:>12}")

    return esm_results


# ═══════════════════════════════════════════════════════════
# METHOD 2: DynaMut2 API
# ═══════════════════════════════════════════════════════════

def submit_dynamut2(mutations_in_abeta):
    """
    Submit mutations to DynaMut2 API and poll for results.

    mutations_in_abeta: list of (variant_name, chain, wt, pos, mut)
    where pos is the PDB residue number (1-42 for Abeta in 1IYT)

    Returns dict mapping mutation string (e.g. "V18M") to DDG value, or empty dict.
    DynaMut2 sign convention: negative = destabilizing, positive = stabilizing.
    """
    import json

    print("\n" + "=" * 70)
    print("METHOD 2: DynaMut2 (structure-based)")
    print("=" * 70)

    # Build mutations list text
    mutations_lines = []
    mutation_map = {}  # maps "V18M" -> variant_name
    for var_name, chain, wt, pos, mut in mutations_in_abeta:
        mut_str = f"{wt}{pos}{mut}"
        mutations_lines.append(f"{chain} {mut_str}")
        mutation_map[mut_str] = var_name

    mutations_text = "\n".join(mutations_lines)
    print(f"\n  Mutations to submit ({len(mutations_lines)}):")
    for line in mutations_lines:
        print(f"    {line}")

    # Read PDB file
    with open(PDB_FILE, "r") as f:
        pdb_content = f.read()

    print(f"\n  PDB file: {PDB_FILE}")
    print(f"  Submitting to DynaMut2 API...")

    url = "https://biosig.lab.uq.edu.au/dynamut2/api/prediction_list"

    try:
        files = {
            "pdb_file": ("1IYT_model1.pdb", io.BytesIO(pdb_content.encode()), "text/plain"),
            "mutations_list": ("mutations.txt", io.BytesIO(mutations_text.encode()), "text/plain"),
        }

        response = requests.post(url, files=files, timeout=120)
        print(f"  Response status: {response.status_code}")

        if response.status_code != 200:
            print(f"  ERROR: API returned status {response.status_code}")
            print(f"  Body: {response.text[:500]}")
            return {}

        data = json.loads(response.text)

        if "error" in data:
            print(f"  API ERROR: {data['error']}")
            return {}

        if "job_id" in data:
            job_id = data["job_id"]
            print(f"  Job ID: {job_id}")
            return poll_dynamut2(job_id, mutation_map)

        # Results came back immediately
        return parse_dynamut2_response(data, mutation_map)

    except requests.exceptions.Timeout:
        print(f"  TIMEOUT: DynaMut2 API did not respond within 120s")
        return {}
    except Exception as e:
        print(f"  ERROR: {e}")
        return {}


def poll_dynamut2(job_id, mutation_map, max_wait=600, poll_interval=20):
    """Poll DynaMut2 for results by job ID."""
    import json

    poll_url = f"https://biosig.lab.uq.edu.au/dynamut2/api/prediction_list?job_id={job_id}"
    print(f"\n  Polling DynaMut2 (max {max_wait}s, every {poll_interval}s)...")
    start = time.time()

    while time.time() - start < max_wait:
        try:
            response = requests.get(poll_url, timeout=60)
            if response.status_code == 200:
                data = json.loads(response.text)

                if isinstance(data, dict) and data.get("status") == "RUNNING":
                    elapsed = time.time() - start
                    print(f"  Still running... ({elapsed:.0f}s elapsed)")
                else:
                    elapsed = time.time() - start
                    print(f"  Results ready after {elapsed:.0f}s")
                    return parse_dynamut2_response(data, mutation_map)
        except Exception as e:
            print(f"  Poll error: {e}")

        time.sleep(poll_interval)

    print(f"  TIMEOUT: DynaMut2 did not complete within {max_wait}s")
    return {}


def parse_dynamut2_response(data, mutation_map):
    """
    Parse DynaMut2 JSON response.
    Returns dict mapping variant_name to DDG value.

    DynaMut2 convention: prediction > 0 = stabilizing, < 0 = destabilizing.
    We NEGATE to match standard DDG convention: positive = destabilizing.
    """
    results = {}

    print(f"\n  DynaMut2 results (raw API values):")
    print(f"  {'Mutation':<10} {'Chain':>5} {'DDG_raw':>8} {'DDG_std':>8} {'Variant':<30}")
    print(f"  {'─'*10} {'─'*5} {'─'*8} {'─'*8} {'─'*30}")

    for key in sorted(data.keys()):
        if key == "results_page" or key == "status" or key == "job_id":
            continue
        entry = data[key]
        if isinstance(entry, dict) and "mutation" in entry:
            mut_str = entry["mutation"]
            chain = entry.get("chain", "?")
            ddg_raw = entry["prediction"]  # DynaMut2: positive = stabilizing

            # NEGATE to standard convention: positive = destabilizing
            ddg_std = -ddg_raw

            var_name = mutation_map.get(mut_str, f"Unknown ({mut_str})")
            results[var_name] = ddg_std

            print(f"  {mut_str:<10} {chain:>5} {ddg_raw:>8.2f} {ddg_std:>+8.2f} {var_name:<30}")

    if "results_page" in data:
        print(f"\n  Results page: {data['results_page']}")

    print(f"\n  NOTE: DDG_raw is DynaMut2 native (positive=stabilizing).")
    print(f"  DDG_std is standard convention (positive=destabilizing).")
    print(f"  Parsed {len(results)} mutations.")

    return results


# ═══════════════════════════════════════════════════════════
# SIGMA COMPUTATION
# ═══════════════════════════════════════════════════════════

def compute_sigma_from_ddg(ddg_kcal, N=42, T=310.0):
    """
    Compute sigma from DDG.
    sigma = sigma_wt * exp(DDG / (N * R * T))

    DDG > 0 means destabilizing -> sigma > sigma_wt
    DDG < 0 means stabilizing -> sigma < sigma_wt
    """
    return SIGMA_WT * np.exp(ddg_kcal / (N * R_KCAL * T))


# ═══════════════════════════════════════════════════════════
# FIGURE GENERATION
# ═══════════════════════════════════════════════════════════

def generate_figures(esm_results, dynamut_results, output_path):
    """Generate publication-quality figure."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from scipy import stats

    # Collect paired data (variants that have both ESM and DynaMut2 scores)
    paired_variants = []
    esm_only = []

    for var_name, esm_data in esm_results.items():
        entry = {
            "name": var_name,
            "ab_pos": esm_data["ab_pos"],
            "wt": esm_data["wt"],
            "mut": esm_data["mut"],
            "llr": esm_data["llr"],
            "type": esm_data["type"],
            "dynamut_ddg": None,
        }

        if dynamut_results and var_name in dynamut_results:
            entry["dynamut_ddg"] = dynamut_results[var_name]
            paired_variants.append(entry)
        else:
            esm_only.append(entry)

    has_dynamut = len(paired_variants) > 0

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Structure-based $\\Delta\\Delta G$ for APP/A$\\beta$42 Variants",
                 fontsize=14, fontweight="bold", y=1.02)

    # ── Panel (a): ESM-1v LLR vs DynaMut2 DDG ──
    ax = axes[0]

    if has_dynamut:
        vus_llr = [v["llr"] for v in paired_variants if v["type"] == "VUS"]
        vus_ddg = [v["dynamut_ddg"] for v in paired_variants if v["type"] == "VUS"]
        path_llr = [v["llr"] for v in paired_variants if v["type"] == "pathogenic"]
        path_ddg = [v["dynamut_ddg"] for v in paired_variants if v["type"] == "pathogenic"]

        if vus_llr:
            ax.scatter(vus_ddg, vus_llr, c="#2196F3", s=50, alpha=0.8,
                      label="VUS", zorder=3, edgecolors="white", linewidth=0.5)
        if path_llr:
            ax.scatter(path_ddg, path_llr, c="#F44336", s=80, alpha=0.9,
                      marker="D", label="Known pathogenic", zorder=4,
                      edgecolors="white", linewidth=0.5)

        # Spearman correlation
        all_llr = [v["llr"] for v in paired_variants]
        all_ddg = [v["dynamut_ddg"] for v in paired_variants]
        if len(all_llr) >= 3:
            rho, pval = stats.spearmanr(all_ddg, all_llr)
            ax.annotate(f"Spearman $\\rho$ = {rho:.3f}\n$p$ = {pval:.2e}",
                       xy=(0.05, 0.95), xycoords="axes fraction",
                       fontsize=10, va="top",
                       bbox=dict(boxstyle="round,pad=0.3", facecolor="wheat", alpha=0.8))

        ax.set_xlabel("DynaMut2 $\\Delta\\Delta G$ (kcal/mol)", fontsize=11)
        ax.set_ylabel("ESM-1v LLR", fontsize=11)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)
        ax.legend(fontsize=10)
    else:
        # ESM-1v only: show LLR distribution
        vus_data = [(v["name"], v["llr"]) for v in esm_only if v["type"] == "VUS"]
        path_data = [(v["name"], v["llr"]) for v in esm_only if v["type"] == "pathogenic"]

        all_data = vus_data + path_data
        all_data.sort(key=lambda x: x[1])

        names = [d[0] for d in all_data]
        llrs = [d[1] for d in all_data]
        colors = ["#2196F3" if d[0] in [v[0] for v in vus_data] else "#F44336"
                  for d in all_data]

        y_pos = range(len(names))
        ax.barh(y_pos, llrs, color=colors, alpha=0.8, edgecolor="white", linewidth=0.5)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([n.split("(")[-1].rstrip(")") if "(" in n else n
                           for n in names], fontsize=7)
        ax.set_xlabel("ESM-1v LLR (negative = destabilizing)", fontsize=11)
        ax.axvline(0, color="gray", linestyle="--", alpha=0.5)

        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [Patch(facecolor="#2196F3", label="VUS"),
                          Patch(facecolor="#F44336", label="Known pathogenic")]
        ax.legend(handles=legend_elements, fontsize=10, loc="lower right")

    ax.set_title("(a) ESM-1v vs DynaMut2", fontsize=12, fontweight="bold")

    # ── Panel (b): VUS ranking by sigma ──
    ax = axes[1]

    # Compute sigma_ESM for VUS variants
    # Use LLR as a proxy: more negative LLR = more destabilizing
    # Convert LLR to approximate DDG: DDG ~ -LLR (rough, same scale)
    # Actually, calibrate: if we have pathogenic data with known effects, use that
    # For now, use the paper's DDG values to compute sigma, and overlay ESM-1v rank

    vus_for_plot = []
    for var_name, app_pos, wt, mut, region, paper_ddg, paper_sigma in VUS_VARIANTS:
        if region != "Abeta":
            continue

        # Get ESM-1v LLR
        esm_llr = None
        for ename, edata in esm_results.items():
            if var_name in ename or ename == var_name:
                esm_llr = edata["llr"]
                break

        # Sigma from paper DDG
        sigma_paper = compute_sigma_from_ddg(paper_ddg)

        # Sigma from ESM-1v: use linear mapping
        # We'll calibrate after collecting all data

        vus_for_plot.append({
            "name": var_name,
            "ab_name": f"{wt}{app_to_abeta(app_pos)}{mut}",
            "paper_ddg": paper_ddg,
            "sigma_paper": sigma_paper,
            "paper_sigma": paper_sigma,
            "esm_llr": esm_llr,
        })

    # Sort by paper sigma (descending for horizontal bars)
    vus_for_plot.sort(key=lambda x: x["paper_sigma"])

    names_plot = [f"{v['name']} ({v['ab_name']})" for v in vus_for_plot]
    sigmas_paper = [v["paper_sigma"] for v in vus_for_plot]

    y_pos = range(len(names_plot))

    # Paper sigma bars
    bars = ax.barh(y_pos, sigmas_paper, color="#2196F3", alpha=0.7,
                   label="$\\sigma$ (paper, generic matrix)", edgecolor="white", linewidth=0.5)

    # If we have ESM data, overlay ESM-derived sigma ranking
    esm_llrs_for_sigma = [v["esm_llr"] for v in vus_for_plot if v["esm_llr"] is not None]
    if esm_llrs_for_sigma:
        # Convert ESM LLR to approximate DDG using calibration
        # LLR is negative for destabilizing, DDG is positive for destabilizing
        # Simple calibration: DDG_esm = -k * LLR
        # Use known pathogenic mutations to calibrate if possible
        # For now: use linear scaling where mean(|LLR|) maps to mean(paper DDG)

        llr_abs = [abs(v["esm_llr"]) for v in vus_for_plot if v["esm_llr"] is not None]
        ddg_abs = [abs(v["paper_ddg"]) for v in vus_for_plot if v["esm_llr"] is not None]

        if np.mean(llr_abs) > 0:
            k_calib = np.mean(ddg_abs) / np.mean(llr_abs)
        else:
            k_calib = 1.0

        sigma_esm_vals = []
        for v in vus_for_plot:
            if v["esm_llr"] is not None:
                ddg_esm = -k_calib * v["esm_llr"]  # negative LLR -> positive DDG
                sigma_esm = compute_sigma_from_ddg(ddg_esm)
                sigma_esm_vals.append(sigma_esm)
            else:
                sigma_esm_vals.append(None)

        # Overlay ESM sigma as markers
        for i, sigma_e in enumerate(sigma_esm_vals):
            if sigma_e is not None:
                ax.plot(sigma_e, i, "D", color="#F44336", markersize=5, alpha=0.8,
                       zorder=5)

        # Dummy for legend
        ax.plot([], [], "D", color="#F44336", markersize=5, label="$\\sigma$ (ESM-1v calibrated)")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names_plot, fontsize=7)
    ax.set_xlabel("$\\sigma$", fontsize=12)
    ax.axvline(1.0, color="red", linestyle="--", alpha=0.7, label="$\\sigma = 1$ (critical)")
    ax.legend(fontsize=8, loc="lower right")
    ax.set_title("(b) VUS ranking (A$\\beta$ region)", fontsize=12, fontweight="bold")

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    print(f"\n  Figure saved: {output_path}")

    # Also save PNG
    png_path = output_path.replace(".pdf", ".png")
    plt.savefig(png_path, dpi=300, bbox_inches="tight")
    print(f"  PNG saved: {png_path}")
    plt.close()


# ═══════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════

def main():
    print("=" * 70)
    print("STRUCTURE-BASED DDG FOR APP/Abeta VARIANTS")
    print("Two independent methods: ESM-1v + DynaMut2")
    print("=" * 70)

    # ── Verify sequence ──
    print(f"\n  Abeta42 sequence: {AB42_SEQ}")
    print(f"  Length: {len(AB42_SEQ)}")

    print(f"\n  Verifying VUS variant positions...")
    for var_name, app_pos, wt, mut, region, _, _ in VUS_VARIANTS:
        if region == "Abeta":
            verify_sequence_position(app_pos, wt)

    print(f"\n  Verifying pathogenic variant positions...")
    for name, var_name, app_pos, wt, mut in PATHOGENIC_VARIANTS:
        ab_pos = app_to_abeta(app_pos)
        if 1 <= ab_pos <= 42:
            actual = AB42_SEQ[ab_pos - 1]
            if actual != wt:
                print(f"  WARNING: {name} ({var_name}): expected {wt} at Abeta {ab_pos}, found {actual}")
            else:
                print(f"  OK: {name} ({var_name}): {wt}{ab_pos} confirmed")

    # ── Method 1: ESM-1v ──
    esm_results = compute_esm1v_scores()

    # ── Method 2: DynaMut2 ──
    # Build mutation list for Abeta variants
    dynamut_mutations = []

    # VUS in Abeta region
    for var_name, app_pos, wt, mut, region, _, _ in VUS_VARIANTS:
        if region == "Abeta":
            ab_pos = app_to_abeta(app_pos)
            if 1 <= ab_pos <= 42:
                dynamut_mutations.append((var_name, PDB_CHAIN, wt, ab_pos, mut))

    # Known pathogenic in Abeta
    for name, var_name, app_pos, wt, mut in PATHOGENIC_VARIANTS:
        ab_pos = app_to_abeta(app_pos)
        if 1 <= ab_pos <= 42:
            full_name = f"{name} ({var_name})"
            dynamut_mutations.append((full_name, PDB_CHAIN, wt, ab_pos, mut))

    # Submit to DynaMut2 (returns dict of variant_name -> DDG)
    dynamut_results = submit_dynamut2(dynamut_mutations)

    # ── Results Table ──
    print("\n\n" + "=" * 70)
    print("RESULTS TABLE")
    print("=" * 70)

    # Header
    print(f"\n  {'Variant':<12} {'Abeta':>7} {'ESM-1v':>8} {'DynaMut2':>10} "
          f"{'sigma_ESM':>10} {'sigma_DM2':>10} {'sigma_paper':>12}")
    print(f"  {'':.<12} {'':.<7} {'LLR':.<8} {'DDG':.<10} "
          f"{'':.<10} {'':.<10} {'(generic)':.<12}")
    print(f"  {'─'*12} {'─'*7} {'─'*8} {'─'*10} {'─'*10} {'─'*10} {'─'*12}")

    csv_rows = []

    # Calibrate ESM-1v: collect LLR values and paper DDG values for Abeta VUS
    calib_llr = []
    calib_ddg = []
    for var_name, app_pos, wt, mut, region, paper_ddg, paper_sigma in VUS_VARIANTS:
        if region != "Abeta":
            continue
        for ename, edata in esm_results.items():
            if ename == var_name:
                calib_llr.append(edata["llr"])
                calib_ddg.append(paper_ddg)
                break

    # Linear regression: DDG = a * LLR + b
    # Or simpler: DDG = -k * LLR (forcing through origin conceptually)
    if calib_llr:
        from scipy import stats as sp_stats
        slope, intercept, r_value, p_value, std_err = sp_stats.linregress(calib_llr, calib_ddg)
        print(f"\n  ESM-1v calibration: DDG = {slope:.3f} * LLR + {intercept:.3f}")
        print(f"  R^2 = {r_value**2:.3f}, p = {p_value:.2e}")
        print(f"  (This calibration maps ESM-1v LLR to approximate DDG scale)\n")
    else:
        slope, intercept = -1.0, 0.0

    # Print all VUS variants
    print(f"\n  --- VUS (Abeta region) ---")
    for var_name, app_pos, wt, mut, region, paper_ddg, paper_sigma in VUS_VARIANTS:
        if region != "Abeta":
            continue

        ab_pos = app_to_abeta(app_pos)
        ab_name = f"{wt}{ab_pos}{mut}"

        # ESM-1v
        esm_llr = None
        for ename, edata in esm_results.items():
            if ename == var_name:
                esm_llr = edata["llr"]
                break

        # DynaMut2
        dm2_ddg = dynamut_results.get(var_name, None)

        # Sigma from ESM-1v (calibrated)
        sigma_esm = None
        if esm_llr is not None:
            ddg_esm = slope * esm_llr + intercept
            sigma_esm = compute_sigma_from_ddg(ddg_esm)

        # Sigma from DynaMut2
        sigma_dm2 = None
        if dm2_ddg is not None:
            sigma_dm2 = compute_sigma_from_ddg(dm2_ddg)

        esm_str = f"{esm_llr:8.3f}" if esm_llr is not None else "    N/A "
        dm2_str = f"{dm2_ddg:10.3f}" if dm2_ddg is not None else "      N/A "
        se_str = f"{sigma_esm:10.4f}" if sigma_esm is not None else "      N/A "
        sd_str = f"{sigma_dm2:10.4f}" if sigma_dm2 is not None else "      N/A "

        print(f"  {var_name:<12} {ab_name:>7} {esm_str} {dm2_str} "
              f"{se_str} {sd_str} {paper_sigma:12.3f}")

        csv_rows.append({
            "variant_APP": var_name,
            "variant_Abeta": ab_name,
            "APP_pos": app_pos,
            "Abeta_pos": ab_pos,
            "region": region,
            "ESM1v_LLR": esm_llr,
            "DynaMut2_DDG": dm2_ddg,
            "sigma_ESM": sigma_esm,
            "sigma_DynaMut2": sigma_dm2,
            "sigma_paper": paper_sigma,
            "paper_DDG": paper_ddg,
            "type": "VUS",
        })

    # Print VUS outside Abeta
    print(f"\n  --- VUS (outside Abeta, APP domain) ---")
    for var_name, app_pos, wt, mut, region, paper_ddg, paper_sigma in VUS_VARIANTS:
        if region != "APP":
            continue
        print(f"  {var_name:<12} {'N/A':>7} {'N/A':>8} {'N/A':>10} "
              f"{'N/A':>10} {'N/A':>10} {paper_sigma:12.3f}")
        csv_rows.append({
            "variant_APP": var_name,
            "variant_Abeta": "N/A (outside Abeta)",
            "APP_pos": app_pos,
            "Abeta_pos": "N/A",
            "region": region,
            "ESM1v_LLR": None,
            "DynaMut2_DDG": None,
            "sigma_ESM": None,
            "sigma_DynaMut2": None,
            "sigma_paper": paper_sigma,
            "paper_DDG": paper_ddg,
            "type": "VUS",
        })

    # Print known pathogenic
    print(f"\n  --- Known pathogenic (Abeta region) ---")
    for name, var_name, app_pos, wt, mut in PATHOGENIC_VARIANTS:
        ab_pos = app_to_abeta(app_pos)
        if ab_pos < 1 or ab_pos > 42:
            continue
        ab_name = f"{wt}{ab_pos}{mut}"
        full_name = f"{name} ({var_name})"

        esm_llr = None
        for ename, edata in esm_results.items():
            if full_name == ename or var_name in ename:
                esm_llr = edata["llr"]
                break

        dm2_ddg = dynamut_results.get(full_name, None)

        sigma_esm = None
        if esm_llr is not None:
            ddg_esm = slope * esm_llr + intercept
            sigma_esm = compute_sigma_from_ddg(ddg_esm)

        sigma_dm2 = None
        if dm2_ddg is not None:
            sigma_dm2 = compute_sigma_from_ddg(dm2_ddg)

        esm_str = f"{esm_llr:8.3f}" if esm_llr is not None else "    N/A "
        dm2_str = f"{dm2_ddg:10.3f}" if dm2_ddg is not None else "      N/A "
        se_str = f"{sigma_esm:10.4f}" if sigma_esm is not None else "      N/A "
        sd_str = f"{sigma_dm2:10.4f}" if sigma_dm2 is not None else "      N/A "

        print(f"  {var_name:<12} {ab_name:>7} {esm_str} {dm2_str} "
              f"{se_str} {sd_str} {'(pathogenic)':>12}")

        csv_rows.append({
            "variant_APP": var_name,
            "variant_Abeta": ab_name,
            "APP_pos": app_pos,
            "Abeta_pos": ab_pos,
            "region": "Abeta",
            "ESM1v_LLR": esm_llr,
            "DynaMut2_DDG": dm2_ddg,
            "sigma_ESM": sigma_esm,
            "sigma_DynaMut2": sigma_dm2,
            "sigma_paper": None,
            "paper_DDG": None,
            "type": "pathogenic" if name != "Icelandic (protective)" else "protective",
        })

    # ── Spearman correlation ──
    print("\n\n" + "=" * 70)
    print("CORRELATION ANALYSIS")
    print("=" * 70)

    from scipy import stats as sp_stats

    # ESM-1v LLR vs paper DDG (for Abeta VUS)
    esm_vals = []
    paper_vals = []
    for row in csv_rows:
        if row["ESM1v_LLR"] is not None and row["paper_DDG"] is not None and row["region"] == "Abeta":
            esm_vals.append(row["ESM1v_LLR"])
            paper_vals.append(row["paper_DDG"])

    if len(esm_vals) >= 3:
        rho, pval = sp_stats.spearmanr(esm_vals, paper_vals)
        print(f"\n  ESM-1v LLR vs Paper DDG (Abeta VUS, n={len(esm_vals)}):")
        print(f"  Spearman rho = {rho:.4f}, p = {pval:.2e}")

        r_pearson, p_pearson = sp_stats.pearsonr(esm_vals, paper_vals)
        print(f"  Pearson r = {r_pearson:.4f}, p = {p_pearson:.2e}")

    # ESM-1v vs DynaMut2 (if available)
    if dynamut_results:
        esm_d = []
        dm2_d = []
        for row in csv_rows:
            if row["ESM1v_LLR"] is not None and row["DynaMut2_DDG"] is not None:
                esm_d.append(row["ESM1v_LLR"])
                dm2_d.append(row["DynaMut2_DDG"])

        if len(esm_d) >= 3:
            rho2, pval2 = sp_stats.spearmanr(esm_d, dm2_d)
            print(f"\n  ESM-1v LLR vs DynaMut2 DDG (n={len(esm_d)}):")
            print(f"  Spearman rho = {rho2:.4f}, p = {pval2:.2e}")

    # ── Save CSV ──
    csv_path = "/home/ffai/code/papers/structure_ddg_results.csv"
    fieldnames = ["variant_APP", "variant_Abeta", "APP_pos", "Abeta_pos", "region",
                  "ESM1v_LLR", "DynaMut2_DDG", "sigma_ESM", "sigma_DynaMut2",
                  "sigma_paper", "paper_DDG", "type"]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(csv_rows)

    print(f"\n  Results saved: {csv_path}")
    print(f"  Total variants: {len(csv_rows)}")

    # ── Generate figure ──
    fig_path = "/home/ffai/code/papers/paper5_submission/fig_structure_ddg.pdf"
    generate_figures(esm_results, dynamut_results, fig_path)

    # ── Summary ──
    print("\n\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    n_esm = sum(1 for r in csv_rows if r["ESM1v_LLR"] is not None)
    n_dm2 = sum(1 for r in csv_rows if r["DynaMut2_DDG"] is not None)

    print(f"\n  Variants analyzed: {len(csv_rows)}")
    print(f"  ESM-1v scores: {n_esm}")
    print(f"  DynaMut2 scores: {n_dm2}")

    if n_dm2 == 0:
        print(f"\n  NOTE: DynaMut2 API did not return results.")
        print(f"  ESM-1v results are complete and can be used independently.")
        print(f"  DynaMut2 results can be added later if the API becomes available.")

    # Top destabilizing by ESM-1v
    esm_ranked = [(r["variant_APP"], r["variant_Abeta"], r["ESM1v_LLR"], r["type"])
                  for r in csv_rows if r["ESM1v_LLR"] is not None]
    esm_ranked.sort(key=lambda x: x[2])  # most negative = most destabilizing

    print(f"\n  Top 10 most destabilizing by ESM-1v (most negative LLR):")
    for i, (vapp, vab, llr, vtype) in enumerate(esm_ranked[:10]):
        print(f"    {i+1:2d}. {vapp:<12} ({vab:<7}) LLR = {llr:+.3f}  [{vtype}]")

    print(f"\n  Bottom 5 (least destabilizing / potentially stabilizing):")
    for i, (vapp, vab, llr, vtype) in enumerate(esm_ranked[-5:]):
        print(f"    {len(esm_ranked)-4+i:2d}. {vapp:<12} ({vab:<7}) LLR = {llr:+.3f}  [{vtype}]")

    # Check: do pathogenic mutations rank higher than most VUS?
    path_ranks = []
    for i, (vapp, vab, llr, vtype) in enumerate(esm_ranked):
        if vtype in ("pathogenic", "protective"):
            path_ranks.append((i+1, vapp, llr, vtype))

    if path_ranks:
        print(f"\n  Known pathogenic/protective mutation ranks (out of {len(esm_ranked)}):")
        for rank, vapp, llr, vtype in path_ranks:
            print(f"    Rank {rank}: {vapp} (LLR={llr:+.3f}) [{vtype}]")


if __name__ == "__main__":
    main()
