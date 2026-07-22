# Explainability Gap — Full Manuscript Validation Ledger
_Every numeric claim in PilotPaperTex/main.tex checked against source data. Generated after the 3-vote re-judge._

## Legend
✅ validates exactly · ⚠️ typo/rounding · 🔄 needs 3-vote anchor update · ❌ separate pipeline (not this re-judge)

---

## A. Judge-INDEPENDENT claims (generation scores) — should be stable

| Loc | Claim | Source check | Status |
|---|---|---|---|
| L85 | 1,430 valid PHQ-8 rows, grand mean 8.02, SD 4.04 | 1,430 ✓, mean 8.01, **SD 4.14** | ⚠️ SD is 4.14 not 4.04 (likely transposed-digit typo); mean 8.01 vs 8.02 rounding |
| L96-97 | SES: High 713/5.12/2.54, Low 717/10.90/3.32 | exact | ✅ |
| L99-101 | Gender: CM 467/6.44, CW 481/7.50, TW 482/10.06 | exact | ✅ |
| L103-105 | Race: Asian 483/7.03, Black 482/8.45, White 465/8.58 | exact | ✅ |
| L107-110 | Model: DS 342/7.08, GLM 350/8.10, GPT 378/8.71, Gem 360/8.10 | exact | ✅ |
| L112-113 | Framing: Clinical 716/8.37, Personal 714/7.66 | exact | ✅ |
| L127-132 | Variance: SES 48.75%, gender 13.33%, race 2.89%, model 2.22% | exact | ✅ |
| L166,L175-180 | GLM tokens 807.2/600/11228; DS 222.9, GPT 263.0, Gem 238.7 | exact | ✅ |
| L328 | PHQ-8 SES gap 5.78 pts, d=1.95 | 5.78 ✓ | ✅ |

**Judge-independent core is solid** — the FA inversion, SES dominance, and effect-size floor all reproduce exactly. Only fix: the 4.04→4.14 SD typo.

---

## B. Judge-DEPENDENT claims — need 3-vote internal-anchor update

### Per-axis descriptives (Table, L198-201)
| Axis | OLD (single-judge, external) | NEW (3-vote, internal) |
|---|---|---|
| A1 Perception | 1.14 / SD 0.48 | **1.16 / 0.52** |
| A2 Faithfulness | 2.66 / 1.50 / 31.28% ≥4 | **2.53 / 1.54 / 29.0%** |
| A3 Counterfactual | 492 / 2.31 / 1.37 | **494 / 2.16 / 1.30** |
| A4 Fairwashing | 3.49 / 1.47 / 61.55% ≥4 | **3.02 / 1.57 / 40.3%** |

### Per-type table (L228-231) — **ordering changes**
| Type | OLD A2/A4 | NEW 3-vote A2/A4 |
|---|---|---|
| Contrastive | 2.23 / 2.75 | **2.28 / 2.95** |
| Abductive | 2.58 / 3.55 | **1.29 / 2.25** (now most faithful) |
| Counterfactual | 2.61 / 3.59 | **2.18 / 2.80** |
| Feature Attribution | 3.10 / 3.89 | **4.29 / 4.04** (now least faithful, by far) |

### Demographic effects (L256-261) — **two effects change qualitatively**
| Row | OLD | NEW 3-vote |
|---|---|---|
| A2 Race | H=43.7 ε²=0.039, d +0.51 | **H=9.8 ε²=0.007, d +0.22** (weaker) |
| A2 SES | d +0.25 | **d +0.20** |
| A4 Race | H=25.4 ε²=0.022, d +0.36 | **H=3.7 ε²=0.002, d −0.02 — now NS** (effect vanishes) |
| A4 Gender | H=9.74 ε²=0.007, d +0.27 | **H=64.9 ε²=0.059, d +0.50** (much stronger) |
| A4 SES | d +0.29 | **d +0.45** (stronger) |

### By-model (L213-215)
| | OLD | NEW 3-vote |
|---|---|---|
| A2 model | H=24.6 ε²=0.020 | **H=17.4 ε²=0.013** |
| A4 model | H=73.7 ε²=0.066 | **H=49.2 ε²=0.043** |

### Interaction / framing
| Loc | OLD | NEW 3-vote |
|---|---|---|
| L268 FA A4 High-SES | 4.20 | **4.16** |
| L270 framing A4 | clinical 3.31 vs personal 3.61 | **3.04 vs 3.00 (gap collapses)** |

---

## C. Separate-pipeline claims — NOT touched by this re-judge

| Loc | Claim | Note |
|---|---|---|
| L276-298 | Cross-judge leniency, inter-rater ρ, self-preference d's | From the 108-item cross-validation sample (crossval_*.csv) — a different study, unaffected by the anchor |
| L293-298 | Self-preference table (Gem d=−0.50 p.026 …) | ❌ Also carries a pre-existing d-formula discrepancy vs self_preference_bias_stats.csv (d=−0.70); part of L303 blinding-leak territory |
| L303 | Blind leak-free self-preference | ❌ Needs the blind leak-free re-run (RUNBOOK step 3) |

---

## Bottom line
- **All judge-independent claims validate** (one SD typo).
- **The 3-vote re-judge reshapes the judge-dependent tables**, and two demographic effects change qualitatively: the A4 *race* effect disappears, the A4 *gender* effect strengthens sharply. These are interpretation-level changes, not just number swaps.
- **Cross-judge / self-preference claims are a separate pipeline** and should not be overwritten with re-judge numbers.


---

## APPLIED (main_rejudge_edited.tex, this pass)
All B-category (judge-dependent) tables and prose updated to 3-vote internal-anchor consensus; SD typo fixed; FA prose/caption/abstract updated to the completed 4-model GLM regen. Specifically:
- Per-axis descriptives table (L198-201) + n 1,074→1,072
- Per-type table (L228-231) + L217 prose (Abductive now best, DeepSeek-FA worst cell) + L320 discussion
- Demographic effects table (L256-261): A4 race→ns, A4 gender→ε²0.059, A2 race weakened
- L245 race/SES prose; L268 SES-by-type gaps; L270 framing (now null moderator)
- By-model L213-215 (DeepSeek now worst on both axes) + L209 3-vote pointer
- Token↔A2 table L177-180 + L166 prose (GLM ρ +0.17→−0.02, null)
- FA inversion L138-140 + caption + abstract L41 (GLM 350/350, SES 5.1%/clinical 88.8%)
- Distribution shape L207 (A4 floor/ceiling/≥4); A1 %named 87.71→88.0; SD 4.04→4.14; grand mean 8.02→8.01
- Section 2.4 A4-SES cue values

**65 lines changed. Integrity: math-$ parity even, braces balanced, line count preserved, 0 raw REJUDGE-UPDATE tags.**

## NOT changed (by design)
- Cross-judge / leniency / self-preference (L276-298): separate 108-item pipeline
- L303 blind leak-free self-preference: needs the RUNBOOK-step-3 re-run
- All judge-independent Phase-1 tables: validated exact, untouched
