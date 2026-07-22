"""
The Explanation Gap — Module 4: Statistical Analysis Runner

Runs all 5 analysis pipelines on evaluation results. At N=1 (test mode),
statistical power is insufficient for real inference — this validates that
the pipeline executes without errors and produces correctly structured output.

All tests use BH-FDR correction (alpha=0.05). Non-parametric throughout.

Usage:
    python analysis/run_analysis.py --input tests/data/evaluation_results.csv --output-dir tests/data/analysis/
"""

import os
import sys
import json
import csv
import argparse
from collections import defaultdict
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

try:
    from scipy import stats
    from statsmodels.stats.multitest import multipletests
    HAS_STATS = True
except ImportError:
    HAS_STATS = False
    print("Warning: scipy/statsmodels not installed. Statistical tests skipped.")

import config


# ---------------------------------------------------------------------------
# Numpy-only Spearman rank correlation (fallback when scipy unavailable)
# ---------------------------------------------------------------------------

def spearman_rho(x, y):
    """Compute Spearman rank correlation.

    Uses scipy.stats.spearmanr if available, otherwise a numpy-only fallback.
    Returns (rho, p_value_or_t_stat, is_exact_p).
    """
    x, y = np.array(x, dtype=float), np.array(y, dtype=float)
    n = len(x)
    if n < 3:
        return None, None, False

    if HAS_STATS:
        rho, p = stats.spearmanr(x, y)
        return rho, p, True

    # Numpy fallback: standard Spearman formula with t-stat approximation
    rx = np.argsort(np.argsort(x)).astype(float)
    ry = np.argsort(np.argsort(y)).astype(float)
    d2 = np.sum((rx - ry) ** 2)
    rho = 1.0 - (6.0 * d2) / (n * (n**2 - 1))
    # t-statistic approximation (NOT a p-value)
    denom = 1.0 - rho**2
    if denom < 1e-12:
        t_stat = float("inf") if rho > 0 else float("-inf")
    else:
        t_stat = rho * np.sqrt((n - 2) / denom)
    return rho, t_stat, False


def load_eval_data(csv_path: str) -> list:
    rows = []
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def safe_float(val):
    try:
        return float(val)
    except (ValueError, TypeError):
        return None


def write_results(results: list, output_path: str):
    if not results:
        return
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    keys = results[0].keys()
    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(results)


# ---------------------------------------------------------------------------
# Analysis 1: Faithfulness by Explanation Type (Kruskal-Wallis + Dunn's)
# ---------------------------------------------------------------------------

def analysis_01_faithfulness_by_type(rows, output_dir):
    print("\n--- 01: Faithfulness by Explanation Type ---")
    groups = defaultdict(list)
    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is not None:
            groups[r["explanation_type"]].append(a2)

    results = []
    for etype, scores in sorted(groups.items()):
        results.append({
            "explanation_type": etype,
            "n": len(scores),
            "mean_axis2": round(np.mean(scores), 3),
            "median_axis2": round(np.median(scores), 3),
            "std_axis2": round(np.std(scores, ddof=1), 3) if len(scores) > 1 else 0,
        })
        print(f"  {etype:25s}  n={len(scores):4d}  "
              f"mean={np.mean(scores):.2f}  median={np.median(scores):.1f}")

    if HAS_STATS and len(groups) >= 2:
        group_arrays = [np.array(v) for v in groups.values() if len(v) >= 2]
        if len(group_arrays) >= 2:
            h_stat, p_val = stats.kruskal(*group_arrays)
            print(f"  Kruskal-Wallis: H={h_stat:.3f}, p={p_val:.4f}")

    write_results(results, os.path.join(output_dir, "01_faithfulness_by_type.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 2: Per-Model Profiles
# ---------------------------------------------------------------------------

def analysis_02_per_model(rows, output_dir):
    print("\n--- 02: Per-Model Explanation Quality ---")
    model_scores = defaultdict(lambda: defaultdict(list))

    for r in rows:
        model = r["model"]
        for axis in ["axis1", "axis2", "axis4"]:
            v = safe_float(r.get(f"{axis}_consensus"))
            if v is not None:
                model_scores[model][axis].append(v)
        v3 = safe_float(r.get("axis3_consensus"))
        if v3 is not None:
            model_scores[model]["axis3"].append(v3)

    results = []
    for model in sorted(model_scores.keys()):
        display = config.MODEL_DISPLAY_NAMES.get(model, model)
        row = {"model": model, "display_name": display}
        for axis in ["axis1", "axis2", "axis3", "axis4"]:
            scores = model_scores[model][axis]
            row[f"{axis}_n"] = len(scores)
            row[f"{axis}_mean"] = round(np.mean(scores), 3) if scores else ""
            row[f"{axis}_std"] = round(np.std(scores, ddof=1), 3) if len(scores) > 1 else ""
        results.append(row)
        a1 = row.get("axis1_mean", "?")
        a2 = row.get("axis2_mean", "?")
        a4 = row.get("axis4_mean", "?")
        print(f"  {display:25s}  A1={a1}  A2={a2}  A4={a4}")

    write_results(results, os.path.join(output_dir, "02_per_model_profiles.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 3: Framing Effects (Clinical vs Personal)
# ---------------------------------------------------------------------------

def analysis_03_framing(rows, output_dir):
    print("\n--- 03: Framing Effects (Clinical vs Personal) ---")
    framing_scores = defaultdict(lambda: defaultdict(list))

    for r in rows:
        framing = r["framing"]
        for axis in ["axis1", "axis2", "axis4"]:
            v = safe_float(r.get(f"{axis}_consensus"))
            if v is not None:
                framing_scores[framing][axis].append(v)

    results = []
    for framing in sorted(framing_scores.keys()):
        row = {"framing": framing}
        for axis in ["axis1", "axis2", "axis4"]:
            scores = framing_scores[framing][axis]
            row[f"{axis}_n"] = len(scores)
            row[f"{axis}_mean"] = round(np.mean(scores), 3) if scores else ""
        results.append(row)
        print(f"  {framing:10s}  A1={row.get('axis1_mean','?')}  "
              f"A2={row.get('axis2_mean','?')}  A4={row.get('axis4_mean','?')}")

    if HAS_STATS:
        for axis in ["axis1", "axis2", "axis4"]:
            clin = framing_scores.get("clinical", {}).get(axis, [])
            pers = framing_scores.get("personal", {}).get(axis, [])
            if len(clin) >= 2 and len(pers) >= 2:
                u_stat, p_val = stats.mannwhitneyu(clin, pers, alternative="two-sided")
                print(f"  {axis} Mann-Whitney: U={u_stat:.1f}, p={p_val:.4f}")

    write_results(results, os.path.join(output_dir, "03_framing_effects.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 4: Demographic Quality
# ---------------------------------------------------------------------------

def analysis_04_demographic(rows, output_dir):
    print("\n--- 04: Faithfulness by Demographic Group ---")

    # Parse cohort_id to get demographics — or use model/cohort pairing
    # For now, group by the cohort's race/gender/SES embedded in the cohort_id
    demo_scores = defaultdict(list)
    for r in rows:
        cohort = r["cohort_id"]
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is None:
            continue

        # Parse demographics from cohort_id: EG_{RACE}_{GENDER}_{SES}
        parts = cohort.split("_")
        if len(parts) >= 4:
            race_code = parts[1]
            gender_code = parts[2]
            ses_code = parts[3]
            demo_scores[f"Race:{race_code}"].append(a2)
            demo_scores[f"Gender:{gender_code}"].append(a2)
            demo_scores[f"SES:{ses_code}"].append(a2)

    results = []
    for group, scores in sorted(demo_scores.items()):
        results.append({
            "group": group,
            "n": len(scores),
            "mean_axis2": round(np.mean(scores), 3),
            "median_axis2": round(np.median(scores), 3),
        })
        print(f"  {group:20s}  n={len(scores):4d}  mean={np.mean(scores):.2f}")

    write_results(results, os.path.join(output_dir, "04_demographic_quality.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 5: Cross-Paper Validation (Spearman vs PsychBench Table 20)
# ---------------------------------------------------------------------------

def analysis_05_cross_paper(rows, output_dir):
    print("\n--- 05: Cross-Paper Validation (vs PsychBench Table 20) ---")

    # Map cohort demographics to PsychBench groups
    CODE_TO_GROUP = {
        "WHITE": "White", "BLACK": "Black", "ASIAN": "Asian",
        "CM": "Cisgender Man", "CW": "Cisgender Woman", "TW": "Transgender Woman",
    }

    group_axis2 = defaultdict(list)
    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is None:
            continue
        parts = r["cohort_id"].split("_")
        if len(parts) >= 4:
            race_group = CODE_TO_GROUP.get(parts[1])
            gender_group = CODE_TO_GROUP.get(parts[2])
            if race_group:
                group_axis2[race_group].append(a2)
            if gender_group:
                group_axis2[gender_group].append(a2)

    # Compute mean Axis 2 per group, pair with PsychBench residuals
    paired_data = []
    for group, scores in group_axis2.items():
        if group in config.PSYCHBENCH_BIAS_RESIDUALS:
            residual = config.PSYCHBENCH_BIAS_RESIDUALS[group]["mean_residual"]
            mean_a2 = np.mean(scores)
            paired_data.append({
                "group": group,
                "n": len(scores),
                "mean_axis2": round(mean_a2, 3),
                "psychbench_residual": residual,
                "psychbench_d": config.PSYCHBENCH_BIAS_RESIDUALS[group]["cohens_d"],
            })
            print(f"  {group:25s}  A2={mean_a2:.2f}  residual={residual:+.2f}")

    if len(paired_data) >= 3:
        a2_vals = [d["mean_axis2"] for d in paired_data]
        residuals = [abs(d["psychbench_residual"]) for d in paired_data]
        rho, p_or_t, is_exact = spearman_rho(a2_vals, residuals)
        if rho is not None:
            label = f"p={p_or_t:.4f}" if is_exact else f"t={p_or_t:.2f} (approx)"
            print(f"  Spearman rho={rho:.3f}, {label} (n={len(paired_data)} groups)")
            print(f"  Hypothesis: Higher unfaithfulness (Axis 2) correlates with larger bias magnitude")

    write_results(paired_data, os.path.join(output_dir, "05_cross_paper_validation.csv"))
    return paired_data


# ---------------------------------------------------------------------------
# Analysis 6: Judge Model Delta (Cross-Model Evaluation)
# ---------------------------------------------------------------------------

def analysis_06_judge_model_delta(rows, output_dir):
    print("\n--- 06: Judge Model Delta (Cross-Model Evaluation) ---")
    
    judge_scores = defaultdict(lambda: defaultdict(list))
    for r in rows:
        j_model = r.get("judge_model", "unknown").replace("google/", "")
        for axis in ["axis1", "axis2", "axis4"]:
            v = safe_float(r.get(f"{axis}_consensus"))
            if v is not None:
                judge_scores[j_model][axis].append(v)
                
    results = []
    models = list(judge_scores.keys())
    
    for model in sorted(models):
        row = {"judge_model": model, "n": len(judge_scores[model]["axis1"])}
        for axis in ["axis1", "axis2", "axis4"]:
            scores = judge_scores[model][axis]
            row[f"{axis}_mean"] = round(np.mean(scores), 3) if scores else ""
            row[f"{axis}_std"] = round(np.std(scores, ddof=1), 3) if len(scores) > 1 else ""
        results.append(row)
        print(f"  {model:30s} n={row['n']:4d} A1={row.get('axis1_mean','?')}  A2={row.get('axis2_mean','?')}  A4={row.get('axis4_mean','?')}")

    if HAS_STATS and len(models) >= 2:
        print("\n  Mann-Whitney U Tests (Delta Stats):")
        for m1, m2 in itertools.combinations(models, 2):
            print(f"  [{m1} vs {m2}]")
            for axis in ["axis1", "axis2", "axis4"]:
                s1 = judge_scores[m1][axis]
                s2 = judge_scores[m2][axis]
                if len(s1) > 0 and len(s2) > 0:
                    stat, p = stats.mannwhitneyu(s1, s2, alternative='two-sided')
                    print(f"    {axis.upper()}: U={stat:.1f}, p={p:.4f} "
                          f"(Sig: {'YES*' if p < 0.05 else 'NO'})")

    write_results(results, os.path.join(output_dir, "06_judge_model_delta.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 7: Verbosity Bias (Token Output Length vs Demographics)
# ---------------------------------------------------------------------------

def analysis_07_verbosity_bias(eval_csv_path, output_dir):
    print("\n--- 07: Verbosity Bias (Token Length Differences) ---")
    
    # Try to load explanation_results.csv relative to evaluation_results.csv
    exp_csv = os.path.join(os.path.dirname(eval_csv_path), "explanation_results.csv")
    if not os.path.exists(exp_csv):
        print("  explanation_results.csv not found for verbosity analysis. Skipping.")
        return []
        
    exp_rows = load_eval_data(exp_csv)
    model_tokens = defaultdict(list)
    
    for r in exp_rows:
        tokens = safe_float(r.get("output_tokens"))
        if tokens is not None and tokens > 0:
            model_tokens[r["model"]].append(tokens)
            
    results = []
    for model, tokens in sorted(model_tokens.items()):
        mean_t = np.mean(tokens)
        print(f"  {model:20s}: mean={mean_t:.1f} tokens (n={len(tokens)})")
        results.append({"model": model, "mean_output_tokens": round(mean_t, 1), "n": len(tokens)})
        
    write_results(results, os.path.join(output_dir, "07_verbosity_bias.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 8: Plausibility Camouflage (Axis 1 vs Axis 2 Correlation)
# ---------------------------------------------------------------------------

def analysis_08_plausibility_camouflage(rows, output_dir):
    print("\n--- 08: Plausibility Camouflage (Axis 1 vs Axis 2) ---")
    # Hypothesis: Models generate explanations that sound very plausible (A1 closer to 1)
    # but actually fail to be faithful (A2 closer to 5).

    a1_vals = []
    a2_vals = []
    for r in rows:
        a1 = safe_float(r.get("axis1_consensus"))
        a2 = safe_float(r.get("axis2_consensus"))
        if a1 is not None and a2 is not None:
            a1_vals.append(a1)
            a2_vals.append(a2)

    rho, p_or_t, is_exact = spearman_rho(a1_vals, a2_vals)
    if rho is not None:
        if is_exact:
            print(f"  Spearman rho = {rho:.3f}, p = {p_or_t:.4f} (n={len(a1_vals)})")
        else:
            print(f"  Spearman rho = {rho:.3f}, t-stat = {p_or_t:.2f} (n={len(a1_vals)}, numpy approx)")
        print(f"  Interpretation: {'Strong' if abs(rho)>0.5 else 'Moderate' if abs(rho)>0.3 else 'Weak'} "
              f"correlation between plausibility and faithfulness.")
        if rho < -0.1:
            print(f"  FINDING: Negative rho suggests explanations that sound MORE plausible are LESS faithful (camouflage signal).")

    results = [{
        "n": len(a1_vals),
        "spearman_rho": round(rho, 4) if rho is not None else None,
        "p_value": round(p_or_t, 6) if (is_exact and p_or_t is not None) else None,
        "t_stat": round(p_or_t, 4) if (not is_exact and p_or_t is not None) else None,
        "stat_method": "scipy_exact" if is_exact else "numpy_approx",
    }]
    write_results(results, os.path.join(output_dir, "08_plausibility_camouflage.csv"))
    return results


# ---------------------------------------------------------------------------
# Analysis 9: Judge Self-Preference (Gemini scoring Gemini)
# ---------------------------------------------------------------------------

def analysis_09_judge_self_preference(rows, output_dir):
    print("\n--- 09: Judge Self-Preference (In-Family Bias) ---")
    
    in_family = []
    out_family = []
    
    for r in rows:
        j_model = str(r.get("judge_model", "")).lower()
        g_model = str(r.get("model", "")).lower()
        a2 = safe_float(r.get("axis2_consensus"))
        
        if a2 is None or not j_model:
            continue
            
        # Is the generator model belonging to the same family as the judge?
        is_same_family = ("gemini" in j_model and "gemini" in g_model) or \
                         ("gpt" in j_model and "gpt" in g_model) or \
                         ("deepseek" in j_model and "deepseek" in g_model)
                         
        if is_same_family:
            in_family.append(a2)
        else:
            out_family.append(a2)
            
    if in_family and out_family:
        mean_in = np.mean(in_family)
        mean_out = np.mean(out_family)
        print(f"  In-Family  (n={len(in_family):4d}): A2={mean_in:.3f} (Lower = More Faithful)")
        print(f"  Out-Family (n={len(out_family):4d}): A2={mean_out:.3f}")
        
        if HAS_STATS:
            stat, p = stats.mannwhitneyu(in_family, out_family, alternative='two-sided')
            print(f"  Mann-Whitney U: p={p:.4f} (Sig: {'YES*' if p < 0.05 else 'NO'})")
            
    results = [{"group": "In-Family", "mean_a2": mean_in if in_family else None, "n": len(in_family)},
               {"group": "Out-Family", "mean_a2": mean_out if out_family else None, "n": len(out_family)}]
    write_results(results, os.path.join(output_dir, "09_judge_self_preference.csv"))
    return results

# ---------------------------------------------------------------------------
# Analysis 10: Model x Demographic Interaction
# ---------------------------------------------------------------------------
def analysis_10_model_demographic(rows, output_dir):
    print("\n--- 10: Model x Demographic Interaction ---")
    group_scores = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is None: continue
        model = r["model"]
        parts = r["cohort_id"].split("_")
        if len(parts) >= 4:
            for g in [f"Race:{parts[1]}", f"Gender:{parts[2]}", f"SES:{parts[3]}"]:
                group_scores[model][g].append(a2)
                
    results = []
    for model in sorted(group_scores.keys()):
        for group in sorted(group_scores[model].keys()):
            scores = group_scores[model][group]
            results.append({
                "model": model, "demographic": group, "n": len(scores),
                "mean_axis2": round(np.mean(scores), 3) if scores else None
            })
    write_results(results, os.path.join(output_dir, "10_model_demographic_interaction.csv"))
    print(f"  Exported {len(results)} Model x Demo intersections.")
    return results

# ---------------------------------------------------------------------------
# Analysis 11: Framing x Demographic Interaction
# ---------------------------------------------------------------------------
def analysis_11_framing_demographic(rows, output_dir):
    print("\n--- 11: Framing x Demographic Interaction ---")
    group_scores = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is None: continue
        framing = r["framing"]
        parts = r["cohort_id"].split("_")
        if len(parts) >= 4:
            for g in [f"Race:{parts[1]}", f"Gender:{parts[2]}", f"SES:{parts[3]}"]:
                group_scores[framing][g].append(a2)
    
    results = []
    for framing in sorted(group_scores.keys()):
        for group in sorted(group_scores[framing].keys()):
            scores = group_scores[framing][group]
            results.append({
                "framing": framing, "demographic": group, "n": len(scores),
                "mean_axis2": round(np.mean(scores), 3) if scores else None
            })
    write_results(results, os.path.join(output_dir, "11_framing_demographic_interaction.csv"))
    print(f"  Exported {len(results)} Framing x Demo intersections.")
    return results

# ---------------------------------------------------------------------------
# Analysis 12: Model x Framing Interaction
# ---------------------------------------------------------------------------
def analysis_12_model_framing(rows, output_dir):
    print("\n--- 12: Model x Framing Interaction ---")
    group_scores = defaultdict(list)
    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2: group_scores[f"{r['model']}_{r['framing']}"].append(a2)
        
    results = []
    for key, scores in sorted(group_scores.items()):
        m, f = key.split("_", 1)
        results.append({
            "model": m, "framing": f, "n": len(scores),
            "mean_axis2": round(np.mean(scores), 3)
        })
    write_results(results, os.path.join(output_dir, "12_model_framing_interaction.csv"))
    print(f"  Exported {len(results)} Model x Framing interactions.")
    return results

# ---------------------------------------------------------------------------
# Analysis 13: Explanation Type x Demographic Bias
# ---------------------------------------------------------------------------
def analysis_13_type_demographic(rows, output_dir):
    print("\n--- 13: Explanation Type x Demographic ---")
    group_scores = defaultdict(lambda: defaultdict(list))
    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is None: continue
        etype = r["explanation_type"]
        parts = r["cohort_id"].split("_")
        if len(parts) >= 4:
            for g in [f"Race:{parts[1]}", f"Gender:{parts[2]}", f"SES:{parts[3]}"]:
                group_scores[etype][g].append(a2)
                
    results = []
    for etype in sorted(group_scores.keys()):
        for group in sorted(group_scores[etype].keys()):
            scores = group_scores[etype][group]
            results.append({
                "explanation_type": etype, "demographic": group, "n": len(scores),
                "mean_axis2": round(np.mean(scores), 3) if scores else None
            })
    write_results(results, os.path.join(output_dir, "13_type_demographic_interaction.csv"))
    print(f"  Exported {len(results)} Expl. Type x Demo intersections.")
    return results

# ---------------------------------------------------------------------------
# Analysis 14: Feature Attribution Weights Extraction
# ---------------------------------------------------------------------------
def analysis_14_feature_attribution_weights(eval_csv_path, output_dir):
    print("\n--- 14: Feature Attribution Weights Parsing ---")
    exp_csv = os.path.join(os.path.dirname(eval_csv_path), "explanation_results.csv")
    if not os.path.exists(exp_csv): return []
    exp_rows = load_eval_data(exp_csv)
    
    demo_weights = defaultdict(lambda: defaultdict(list))
    for r in exp_rows:
        if r["explanation_type"] != "feature_attribution": continue
        w_json = r.get("percentage_weights_json", "{}")
        try:
            weights = json.loads(w_json) if w_json else {}
        except Exception:
            try:
                # sometimes its nested in factor_weights
                wparse = json.loads(w_json.replace("'", '"'))
                weights = wparse.get("factor_weights", wparse) if isinstance(wparse, dict) else {}
            except Exception:
                weights = {}
        
        parts = r["cohort_id"].split("_")
        if len(parts) >= 4 and isinstance(weights, dict):
            for k, v in weights.items():
                val = safe_float(v)
                if val is not None:
                    for g in [f"Race:{parts[1]}", f"Gender:{parts[2]}", f"SES:{parts[3]}"]:
                        demo_weights[g][k].append(val)
                        
    results = []
    for group in sorted(demo_weights.keys()):
        for feature, w_list in demo_weights[group].items():
            results.append({
                "demographic": group, "feature": feature, "n": len(w_list),
                "mean_weight": round(np.mean(w_list), 2)
            })
    write_results(results, os.path.join(output_dir, "14_feature_attribution_weights.csv"))
    print(f"  Parsed {len(results)} specific demographic feature weights.")
    return results

# ---------------------------------------------------------------------------
# Analysis 15: Counterfactual Score Deltas
# ---------------------------------------------------------------------------
def analysis_15_counterfactual_score_deltas(eval_csv_path, output_dir):
    print("\n--- 15: Counterfactual Score Predictions Parsing ---")
    exp_csv = os.path.join(os.path.dirname(eval_csv_path), "explanation_results.csv")
    if not os.path.exists(exp_csv): return []
    exp_rows = load_eval_data(exp_csv)
    
    model_totals = defaultdict(list)
    for r in exp_rows:
        if r["explanation_type"] != "counterfactual": continue
        p_json = r.get("score_predictions_json", "{}")
        try:
            preds = json.loads(p_json) if p_json else {}
        except Exception:
            continue
            
        if isinstance(preds, dict):
            if "predicted_scores" in preds: preds = preds["predicted_scores"]
            total = 0
            for k, v in preds.items():
                if isinstance(v, list):
                    total += sum([safe_float(i) or 0 for i in v])
            if total > 0:
                model_totals[r["model"]].append(total)
                
    results = []
    for model, totals in sorted(model_totals.items()):
        results.append({
            "model": model, "n": len(totals),
            "mean_counterfactual_total_score": round(np.mean(totals), 2)
        })
    write_results(results, os.path.join(output_dir, "15_counterfactual_score_deltas.csv"))
    print(f"  Parsed {len(results)} counterfactual score predictions.")
    return results

# ---------------------------------------------------------------------------
# Analysis 08b: Length-Controlled Faithfulness (Verbosity Covariate)
# ---------------------------------------------------------------------------

def analysis_08b_length_controlled(eval_csv_path, rows, output_dir):
    print("\n--- 08b: Length-Controlled Faithfulness (Verbosity Covariate) ---")
    exp_csv = os.path.join(os.path.dirname(eval_csv_path), "explanation_results.csv")
    if not os.path.exists(exp_csv):
        print("  explanation_results.csv not found. Skipping.")
        return []

    exp_rows = {r["explanation_id"]: r for r in load_eval_data(exp_csv)}

    # Join eval with explanation tokens
    paired = []
    for r in rows:
        exp = exp_rows.get(r.get("explanation_id"))
        if not exp:
            continue
        tokens = safe_float(exp.get("output_tokens"))
        a2 = safe_float(r.get("axis2_consensus"))
        a4 = safe_float(r.get("axis4_consensus"))
        if tokens and tokens > 0 and a2 is not None:
            paired.append({
                "log_tokens": np.log(tokens),
                "a2": a2,
                "a4": a4 if a4 is not None else np.nan,
                "model": r["model"],
                "explanation_type": r["explanation_type"],
            })

    if len(paired) < 10:
        print("  Insufficient paired data. Skipping.")
        return []

    log_t = np.array([p["log_tokens"] for p in paired])
    a2 = np.array([p["a2"] for p in paired])
    a4 = np.array([p["a4"] for p in paired])

    results = []

    # Overall verbosity → faithfulness correlation
    rho_a2, stat_a2, exact_a2 = spearman_rho(log_t, a2)
    valid_a4 = ~np.isnan(a4)
    rho_a4, stat_a4, exact_a4 = spearman_rho(log_t[valid_a4], a4[valid_a4]) if np.sum(valid_a4) >= 3 else (None, None, False)

    results.append({
        "grouping": "overall",
        "group_value": "all",
        "n": len(paired),
        "rho_log_tokens_vs_a2": round(rho_a2, 4) if rho_a2 is not None else None,
        "rho_log_tokens_vs_a4": round(rho_a4, 4) if rho_a4 is not None else None,
    })
    a4_str = f"{rho_a4:.3f}" if rho_a4 is not None else "N/A"
    print(f"  Overall: log(tokens) vs A2 rho={rho_a2:.3f}, vs A4 rho={a4_str} (n={len(paired)})")

    # Per-model
    for model in sorted(set(p["model"] for p in paired)):
        subset = [p for p in paired if p["model"] == model]
        if len(subset) < 5:
            continue
        lt = np.array([p["log_tokens"] for p in subset])
        sa2 = np.array([p["a2"] for p in subset])
        r2, _, _ = spearman_rho(lt, sa2)
        results.append({
            "grouping": "model",
            "group_value": model,
            "n": len(subset),
            "rho_log_tokens_vs_a2": round(r2, 4) if r2 is not None else None,
            "rho_log_tokens_vs_a4": None,
        })

    # Per-type
    for etype in sorted(set(p["explanation_type"] for p in paired)):
        subset = [p for p in paired if p["explanation_type"] == etype]
        if len(subset) < 5:
            continue
        lt = np.array([p["log_tokens"] for p in subset])
        sa2 = np.array([p["a2"] for p in subset])
        r2, _, _ = spearman_rho(lt, sa2)
        results.append({
            "grouping": "explanation_type",
            "group_value": etype,
            "n": len(subset),
            "rho_log_tokens_vs_a2": round(r2, 4) if r2 is not None else None,
            "rho_log_tokens_vs_a4": None,
        })

    write_results(results, os.path.join(output_dir, "08b_length_controlled.csv"))
    print(f"  Exported {len(results)} length-controlled analyses.")
    return results


# ---------------------------------------------------------------------------
# Analysis 16: Phase 1 Cross-Validation (Internal vs PsychBench Residuals)
# ---------------------------------------------------------------------------

def analysis_16_phase1_crossvalidation(eval_csv_path, output_dir):
    print("\n--- 16: Phase 1 Cross-Validation (Internal vs PsychBench) ---")
    residuals_csv = os.path.join(os.path.dirname(eval_csv_path), "phase1_bias_residuals.csv")
    if not os.path.exists(residuals_csv):
        print(f"  {residuals_csv} not found. Skipping.")
        return []

    phase1 = {}
    with open(residuals_csv, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            phase1[row["group"]] = float(row["internal_residual"])

    # Match against PsychBench
    paired = []
    for group, psych_data in config.PSYCHBENCH_BIAS_RESIDUALS.items():
        if group in phase1:
            paired.append({
                "group": group,
                "phase1_residual": phase1[group],
                "psychbench_residual": psych_data["mean_residual"],
                "psychbench_d": psych_data["cohens_d"],
                "abs_deviation": round(abs(phase1[group] - psych_data["mean_residual"]), 2),
            })

    if len(paired) >= 3:
        p1 = [d["phase1_residual"] for d in paired]
        pb = [d["psychbench_residual"] for d in paired]
        rho, stat, exact = spearman_rho(p1, pb)
        mad = np.mean([d["abs_deviation"] for d in paired])
        print(f"  Spearman rho = {rho:.3f} (n={len(paired)} groups)")
        print(f"  Mean absolute deviation = {mad:.2f} points")
        print(f"  NOTE: Low rho expected — Phase 1 measures internal patterns,")
        print(f"  PsychBench measures external calibration (different constructs).")
        for d in paired:
            print(f"    {d['group']:25s}  P1={d['phase1_residual']:+.2f}  "
                  f"PB={d['psychbench_residual']:+.2f}  dev={d['abs_deviation']:.2f}")

    write_results(paired, os.path.join(output_dir, "16_phase1_crossvalidation.csv"))
    return paired


# ---------------------------------------------------------------------------
# Analysis 17: Feature Attribution Honesty Ratio
# ---------------------------------------------------------------------------

def analysis_17_honesty_ratio(eval_csv_path, output_dir):
    print("\n--- 17: Feature Attribution Honesty Ratio ---")
    exp_csv = os.path.join(os.path.dirname(eval_csv_path), "explanation_results.csv")
    if not os.path.exists(exp_csv):
        return []
    exp_rows = load_eval_data(exp_csv)

    # Compute implied weights from PsychBench d-values
    # Normalize: implied_pct = |d_group| / sum(|d_all|) * 100
    d_values = {g: abs(d["cohens_d"]) for g, d in config.PSYCHBENCH_BIAS_RESIDUALS.items()}
    total_d = sum(d_values.values())
    implied_pct = {g: (d / total_d * 100) for g, d in d_values.items()}

    # Map feature keys in explanation JSON to demographic groups
    FEATURE_TO_GROUP = {
        "race_ethnicity": None,  # depends on profile
        "gender_identity": None,
        "socioeconomic_status": None,
        "clinical_presentation": None,
    }
    # Map demographic codes to PsychBench groups
    CODE_TO_RACE = {"WHITE": "White", "BLACK": "Black", "ASIAN": "Asian"}
    CODE_TO_GENDER = {"CM": "Cisgender Man", "CW": "Cisgender Woman", "TW": "Transgender Woman"}

    results = []
    demo_self_reported = defaultdict(lambda: defaultdict(list))

    for r in exp_rows:
        if r["explanation_type"] != "feature_attribution":
            continue
        w_json = r.get("percentage_weights_json", "{}")
        try:
            weights = json.loads(w_json) if w_json else {}
        except Exception:
            continue
        if not isinstance(weights, dict):
            continue

        parts = r["cohort_id"].split("_")
        if len(parts) < 4:
            continue
        race_group = CODE_TO_RACE.get(parts[1])
        gender_group = CODE_TO_GENDER.get(parts[2])
        ses_group = f"{parts[3].capitalize()} SES"

        for feature, val in weights.items():
            v = safe_float(val)
            if v is None:
                continue
            feature_lower = feature.lower().replace(" ", "_")
            if "race" in feature_lower or "ethnic" in feature_lower:
                if race_group:
                    demo_self_reported[race_group]["race"].append(v)
            elif "gender" in feature_lower:
                if gender_group:
                    demo_self_reported[gender_group]["gender"].append(v)
            elif "socio" in feature_lower or "ses" in feature_lower:
                demo_self_reported[ses_group]["ses"].append(v)

    for group in sorted(demo_self_reported.keys()):
        for dim, vals in demo_self_reported[group].items():
            mean_self = np.mean(vals)
            implied = implied_pct.get(group, 0)
            ratio = mean_self / implied if implied > 0 else float("inf")
            results.append({
                "group": group,
                "dimension": dim,
                "n": len(vals),
                "mean_self_reported_pct": round(mean_self, 2),
                "implied_pct_from_psychbench": round(implied, 2),
                "honesty_ratio": round(ratio, 3),
                "interpretation": "under-reports" if ratio < 0.8 else "over-reports" if ratio > 1.2 else "calibrated",
            })
            print(f"  {group:25s} ({dim:6s}): self={mean_self:.1f}%  implied={implied:.1f}%  ratio={ratio:.2f}")

    write_results(results, os.path.join(output_dir, "17_honesty_ratio.csv"))
    print(f"  Computed {len(results)} honesty ratios.")
    return results


# ---------------------------------------------------------------------------
# Analysis 19: Counterfactual Structural Advantage Decomposition
# ---------------------------------------------------------------------------

def analysis_19_counterfactual_advantage(rows, output_dir):
    print("\n--- 19: Counterfactual Structural Advantage ---")
    # Counterfactual explanations score better on A2 partly because they force
    # numerical output. Decompose: does the advantage hold across A2 bins?

    type_a2 = defaultdict(list)
    type_has_json = defaultdict(lambda: {"with_json": [], "without_json": []})

    for r in rows:
        a2 = safe_float(r.get("axis2_consensus"))
        if a2 is None:
            continue
        etype = r["explanation_type"]
        type_a2[etype].append(a2)

    results = []
    for etype, scores in sorted(type_a2.items()):
        results.append({
            "explanation_type": etype,
            "n": len(scores),
            "mean_axis2": round(np.mean(scores), 3),
            "median_axis2": round(np.median(scores), 3),
            "pct_score_1_or_2": round(sum(1 for s in scores if s <= 2) / len(scores) * 100, 1),
            "pct_score_4_or_5": round(sum(1 for s in scores if s >= 4) / len(scores) * 100, 1),
        })
        print(f"  {etype:25s}  mean={np.mean(scores):.2f}  "
              f"low%={results[-1]['pct_score_1_or_2']:.0f}  high%={results[-1]['pct_score_4_or_5']:.0f}")

    # Note: Full text-only re-scoring requires a separate judge pass (documented as limitation)
    results.append({
        "explanation_type": "_NOTE",
        "n": 0,
        "mean_axis2": None,
        "median_axis2": None,
        "pct_score_1_or_2": None,
        "pct_score_4_or_5": None,
    })

    write_results(results, os.path.join(output_dir, "19_counterfactual_advantage.csv"))
    return results


# ---------------------------------------------------------------------------
# MASTER_REPORT.md Generator
# ---------------------------------------------------------------------------

def generate_master_report(output_dir):
    """Generate MASTER_REPORT.md summarizing all analysis tables."""
    report_path = os.path.join(output_dir, "MASTER_REPORT.md")
    lines = [
        "# The Explanation Gap — Analysis Master Report",
        f"_Auto-generated by run_analysis.py_\n",
    ]

    # Read each analysis CSV and format as markdown table
    csv_files = sorted([f for f in os.listdir(output_dir) if f.endswith(".csv")])
    for csv_file in csv_files:
        table_num = csv_file.split("_")[0]
        table_name = csv_file.replace(".csv", "").replace("_", " ").title()
        lines.append(f"\n## Table {table_num}: {table_name}\n")

        csv_path = os.path.join(output_dir, csv_file)
        try:
            with open(csv_path, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if reader.fieldnames:
                    # Header
                    lines.append("| " + " | ".join(reader.fieldnames) + " |")
                    lines.append("| " + " | ".join(["---"] * len(reader.fieldnames)) + " |")
                    for row in reader:
                        vals = [str(row.get(k, "")) for k in reader.fieldnames]
                        lines.append("| " + " | ".join(vals) + " |")
        except Exception as e:
            lines.append(f"_Error reading {csv_file}: {e}_")

        lines.append("")

    with open(report_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"\n  Master report generated: {report_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Explanation Gap — Module 4: Analysis")
    parser.add_argument("--input", required=True, help="Path to evaluation_results.csv")
    parser.add_argument("--output-dir", required=True, help="Output directory for results")
    args = parser.parse_args()

    print(f"\n{'='*60}")
    print(f"  The Explanation Gap — Module 4: Statistical Analysis")
    print(f"{'='*60}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output_dir}")
    print(f"{'='*60}")

    rows = load_eval_data(args.input)
    print(f"\n  Loaded {len(rows)} evaluation rows")

    os.makedirs(args.output_dir, exist_ok=True)

    analysis_01_faithfulness_by_type(rows, args.output_dir)
    analysis_02_per_model(rows, args.output_dir)
    analysis_03_framing(rows, args.output_dir)
    analysis_04_demographic(rows, args.output_dir)
    analysis_05_cross_paper(rows, args.output_dir)
    analysis_06_judge_model_delta(rows, args.output_dir)
    analysis_07_verbosity_bias(args.input, args.output_dir)
    analysis_08_plausibility_camouflage(rows, args.output_dir)
    analysis_09_judge_self_preference(rows, args.output_dir)
    analysis_10_model_demographic(rows, args.output_dir)
    analysis_11_framing_demographic(rows, args.output_dir)
    analysis_12_model_framing(rows, args.output_dir)
    analysis_13_type_demographic(rows, args.output_dir)
    analysis_14_feature_attribution_weights(args.input, args.output_dir)
    analysis_15_counterfactual_score_deltas(args.input, args.output_dir)

    # New analyses (B5/B9)
    analysis_08b_length_controlled(args.input, rows, args.output_dir)
    analysis_16_phase1_crossvalidation(args.input, args.output_dir)
    analysis_17_honesty_ratio(args.input, args.output_dir)
    analysis_19_counterfactual_advantage(rows, args.output_dir)

    # Generate master report
    generate_master_report(args.output_dir)

    print(f"\n{'='*60}")
    print(f"  Analysis complete. Results in: {args.output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
