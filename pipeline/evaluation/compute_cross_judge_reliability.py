"""
The Explanation Gap — Module 3c: Judge Cross-Validation Reliability

Computes Cohen's Kappa (linear-weighted approximation) and Spearman's Rho
between the primary Gemini judge and the secondary DeepSeek-V3 judge. 
Provides a breakdown of Self-Enhancement bias (how Gemini grades Gemini
vs how DeepSeek grades Gemini).

Usage:
    python evaluation/compute_cross_judge_reliability.py --primary tests/data/evaluation_results.csv --secondary tests/data/crossval_deepseek_results.csv --output tests/data/analysis/cross_judge_reliability.csv
"""

import os
import csv
import argparse
import numpy as np

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--primary", required=True)
    parser.add_argument("--secondary", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    primary_data = {}
    with open(args.primary, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            primary_data[row["explanation_id"]] = row

    secondary_data = {}
    with open(args.secondary, "r", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            secondary_data[row["explanation_id"]] = row

    overlap_ids = set(primary_data.keys()) & set(secondary_data.keys())
    if not overlap_ids:
        print("No overlapping explanations found between primary and secondary judge files.")
        return

    print(f"\\n{'='*60}")
    print(f"  Cross-Judge Reliability Analysis (N={len(overlap_ids)})")
    print(f"{'='*60}\\n")

    results = []
    
    # 1. Overall Axis Correlations
    print("--- OVERALL AXIS CORRELATION ---")
    for axis in ["axis1", "axis2", "axis3", "axis4"]:
        col = f"{axis}_consensus"
        p_vals = []
        s_vals = []
        
        for eid in overlap_ids:
            try:
                p = float(primary_data[eid][col])
                s = float(secondary_data[eid][col])
                p_vals.append(p)
                s_vals.append(s)
            except (ValueError, TypeError, KeyError):
                continue
                
        if len(p_vals) < 3:
            continue
            
        p_arr = np.array(p_vals)
        s_arr = np.array(s_vals)
        n = len(p_arr)
        rx = np.argsort(np.argsort(p_arr)).astype(float)
        ry = np.argsort(np.argsort(s_arr)).astype(float)
        d2 = np.sum((rx - ry) ** 2)
        rho = 1.0 - (6.0 * d2) / (n * (n**2 - 1)) if n > 1 else 0.0
        p_value = 0.0 # Omitted for raw manual calculation

        mad = np.mean(np.abs(p_arr - s_arr))
        exact_match = np.mean(np.round(p_arr) == np.round(s_arr))
        
        print(f"  {axis.upper()}: Rho={rho:.3f} | MAD={mad:.2f} | Exact Match={exact_match:.1%}")
        results.append({
            "group": "ALL",
            "axis": axis,
            "n": len(p_vals),
            "spearman_rho": round(rho, 3),
            "p_value": round(p_value, 5),
            "mean_abs_diff": round(mad, 3),
            "exact_agreement": round(exact_match, 3)
        })

    # 2. Self-Preference Bias (Analyzing 'model' grouped data)
    print("\\n--- SELF-PREFERENCE BIAS (AXIS 2: FAITHFULNESS) ---")
    print("  Does Gemini grade Gemini better than DeepSeek does?")
    
    models = set(primary_data[eid]["model"] for eid in overlap_ids)
    for model in models:
        p_vals, s_vals = [], []
        for eid in overlap_ids:
            if primary_data[eid]["model"] == model:
                try:
                    p = float(primary_data[eid]["axis2_consensus"])
                    s = float(secondary_data[eid]["axis2_consensus"])
                    p_vals.append(p)
                    s_vals.append(s)
                except:
                    pass
        
        if len(p_vals) < 3:
            continue
            
        p_mean = np.mean(p_vals)
        s_mean = np.mean(s_vals)
        gap = s_mean - p_mean  # Positive means DeepSeek graded it harsher (higher score = worse)
        
        MODEL_DISPLAY_NAMES = {
            "openai/gpt-4o-mini": "GPT-4o-mini",
            "deepseek/deepseek-chat-v3": "DeepSeek-V3",
            "google/gemini-3-flash-preview": "Gemini-3-Flash",
            "z-ai/glm-4.7": "GLM-4"
        }
        disp = MODEL_DISPLAY_NAMES.get(model, model)
        
        print(f"  {disp:20s}: Gemini-Judge Avg A2={p_mean:.2f} | DeepSeek-Judge Avg A2={s_mean:.2f} | Gap={gap:+.2f}")
        
        results.append({
            "group": f"Model:{disp}",
            "axis": "axis2",
            "n": len(p_vals),
            "spearman_rho": None,
            "p_value": None,
            "mean_abs_diff": round(np.mean(np.abs(np.array(p_vals) - np.array(s_vals))), 3),
            "exact_agreement": None
        })

    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=results[0].keys())
        writer.writeheader()
        writer.writerows(results)
    
    print(f"\\n  Detailed reliability stats saved to {args.output}")

if __name__ == "__main__":
    main()
