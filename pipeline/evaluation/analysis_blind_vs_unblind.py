"""
The Explanation Gap — Module 4c: Blind vs Unblind Self-Preference Analysis

This script loads data from standard (unblinded) and blinded (test) judge runs 
for a 4x4 matrix comparison (Gemini, GPT-4o-mini, DeepSeek-V3, GLM-4.7) to measure 
the true geographic alignment gaps and stylistic self-recognition bias.
"""

import os
import csv
import math
import statistics
import argparse
from typing import List, Dict

def calculate_cohens_d(group1: List[float], group2: List[float]) -> float:
    n1, n2 = len(group1), len(group2)
    if n1 < 2 or n2 < 2: return 0.0
    var1 = statistics.variance(group1)
    var2 = statistics.variance(group2)
    pooled_sd = math.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    if pooled_sd == 0: return 0.0
    return (statistics.mean(group1) - statistics.mean(group2)) / pooled_sd

def extract_axis2_scores(csv_path: str, judge_name: str) -> Dict[str, Dict[str, List[float]]]:
    if not os.path.exists(csv_path): return None
        
    target_models = {
        "gemini": "google/gemini-3-flash-preview",
        "deepseek": "deepseek/deepseek-chat-v3",
        "gpt": "openai/gpt-4o-mini",
        "glm": "z-ai/glm-4.7"
    }
    
    in_group_target = target_models.get(judge_name.lower())
    scores = {"in_group": [], "out_group": []}
    
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if not row.get("axis2_consensus"): continue
            val = float(row["axis2_consensus"])
            model_evaluated = row["model"]
            
            if model_evaluated == in_group_target:
                scores["in_group"].append(val)
            else:
                scores["out_group"].append(val)
    return scores

def format_stats(scores: Dict[str, List[float]]) -> str:
    if not scores or len(scores["in_group"]) == 0 or len(scores["out_group"]) == 0:
        return "| N/A | N/A | N/A | N/A | N/A | N/A |"
        
    in_n = len(scores["in_group"])
    in_mean = statistics.mean(scores["in_group"])
    out_n = len(scores["out_group"])
    out_mean = statistics.mean(scores["out_group"])
    gap = in_mean - out_mean
    d = calculate_cohens_d(scores["in_group"], scores["out_group"])
    
    return f"| {in_mean:.3f} | {in_n} | {out_mean:.3f} | {out_n} | {gap:+.3f} | {d:+.3f} |"

def format_section(model_name: str, unblind_scores: dict, blind_scores: dict) -> List[str]:
    lines = []
    lines.append(f"## {model_name} Judge")
    lines.append(f"| Condition | In-Group ({model_name.split()[0]}) | N | Out-Group | N | Bias Gap | Cohen's d |")
    lines.append("|-----------|----------------|---|-----------|---|----------|-----------|")
    
    u_stats = format_stats(unblind_scores) if unblind_scores else "| File missing | - | - | - | - | - |"
    b_stats = format_stats(blind_scores)   if blind_scores   else "| File missing | - | - | - | - | - |"
    
    lines.append(f"| Unblinded (Baseline) {u_stats}")
    lines.append(f"| **Blinded (Test)** {b_stats}")
    lines.append("")
    return lines

def main():
    parser = argparse.ArgumentParser()
    # Unblinded files
    parser.add_argument("--u-gem", default="tests/data/evaluation_results.csv")
    parser.add_argument("--u-ds",  default="tests/data/crossval_deepseek_results.csv")
    parser.add_argument("--u-gpt", default="tests/data/crossval_gpt4omini_results.csv")
    parser.add_argument("--u-glm", default="tests/data/crossval_glm4_results.csv")
    # Blinded files
    parser.add_argument("--b-gem", default="tests/data/blind_gemini_results.csv")
    parser.add_argument("--b-ds",  default="tests/data/blind_deepseek_results.csv")
    parser.add_argument("--b-gpt", default="tests/data/blind_gpt_results.csv")
    parser.add_argument("--b-glm", default="tests/data/blind_glm_results.csv")
    parser.add_argument("--out",   default="reports/blind_vs_unblind_analysis.md")
    args = parser.parse_args()
    
    print("Loading geographic dual-method data...")
    gem_unblind = extract_axis2_scores(args.u_gem, "gemini")
    gem_blind   = extract_axis2_scores(args.b_gem, "gemini")
    
    gpt_unblind = extract_axis2_scores(args.u_gpt, "gpt")
    gpt_blind   = extract_axis2_scores(args.b_gpt, "gpt")

    ds_unblind  = extract_axis2_scores(args.u_ds, "deepseek")
    ds_blind    = extract_axis2_scores(args.b_ds, "deepseek")
    
    glm_unblind = extract_axis2_scores(args.u_glm, "glm")
    glm_blind   = extract_axis2_scores(args.b_glm, "glm")

    report = []
    report.append("# Efficacy of Prompt Anonymization on LLM-as-Judge Self-Preference Bias")
    report.append("This analysis compares Axis 2 (Faithfulness) scoring patterns before and after stripping explicit model labels (`Model: google/gemini-...`) from the evaluation prompt.")
    report.append("\n*(Note: Axis 2 uses a penalty scale 1-5, where lower is better. A negative gap means the judge scored the in-group better than the out-group).*")
    report.append("\n### Western Cluster")
    report.extend(format_section("Gemini 3.0 Flash", gem_unblind, gem_blind))
    report.extend(format_section("GPT-4o-mini", gpt_unblind, gpt_blind))
    
    report.append("### Eastern Cluster")
    report.extend(format_section("DeepSeek-V3", ds_unblind, ds_blind))
    report.extend(format_section("GLM-4.7", glm_unblind, glm_blind))

    md_text = "\n".join(report)
    
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(md_text)
        
    print(f"\nAnalysis generated and saved to {args.out}")
    print("--------------------------------------------------")
    print(md_text)
    print("--------------------------------------------------")

if __name__ == "__main__":
    main()
