# Re-judge flag reconciliation (3-VOTE) — 9 REJUDGE-UPDATE flags in PilotPaperTex/main.tex
_Values from `rejudge_3vote_consensus.csv` (internal-anchor, three-vote consensus, n=1,072) and `fa_weights_4model.csv` (GLM FA regen). Edited manuscript: `main_rejudge_edited.tex`; patch: `main_rejudge.diff`._

**8 discharged · 1 not dischargeable by this run (needs a different pipeline).**

| # | Flag | Was (external anchor) | Now (3-vote internal anchor) |
|---|---|---|---|
| L41 | Abstract A2 faithfulness mean | 2.66 | **2.53** ✅ |
| L42 | Abstract A4 fairwashing headline | 3.49 / 61.6% ≥4 / 52.2% name-FW | **3.02 / 40.3% / 31.6%** ✅ |
| L75 | Vote structure | flag: after 3-vote, drop single-vote caveat | **DISCHARGED — 94.1% 3-vote, 5.9% 2-vote, 0% single** ✅ |
| L207 | Joint A1/A4 distribution | 52.23 / 25.88 / 30.26 | **31.6 / 31.7 / 22.7** ✅ |
| L245 | SES contrast on A4 | 3.71 vs 3.29, d=+0.29 | **3.37 vs 2.68, d=+0.45 — STRENGTHENS** ✅ |
| L266 | 18-cohort A4 extremes | top ASIAN_CM_HIGH 4.31; "4 of 5 bottom Low-SES" | **top ASIAN_CM_HIGH 4.38; bottom-5 now 2 Low/3 High — split RETRACTED** ✅ |
| L303 | Blind leak-free self-preference | flag: values from patched blind re-run | **NOT DISCHARGED — needs blind leak-free re-run (RUNBOOK step 3)** ❌ |
| L318 | FA inversion + joint signal | 63.4%/17.2% (3-model); "52% joint" | **joint→31.6%; +4-model FA (GLM regen 350/350)** ✅ |
| L326 | ~47% single-vote caveat | flag: remove after 3-vote run | **DISCHARGED — set uniformly multi-vote; caveat removed** ✅ |

## 1-vote vs 3-vote (aggregates were stable — validates the earlier 1-vote decision)
| Metric | 1-vote | 3-vote |
|---|---|---|
| A4 fairwashing mean | 3.01 | **3.02** |
| A2 faithfulness mean | 2.52 | **2.53** |
| A4 %≥4 | 47.7% | **40.3%** (averaging pulls borderline 4s below threshold) |
| Name-and-fairwash | 37.9% | **31.6%** |

The **means are essentially identical** across 1- vs 3-vote; only the ≥4 *prevalence* metrics tighten, because a 3-vote mean rarely lands exactly on 4 when one vote dips. The paper's *mean* headlines are robust; if you cite prevalence (%≥4), use the 3-vote 40.3%.

## Judge consistency (3-vote cells)
Mean within-cell SD: **A4 = 0.49, A2 = 0.34, A1 = 0.07** (on a 1–5 scale). A1 (demographic perception) is near-unanimous; A4 (fairwashing) carries the most judge-to-judge spread, as expected for the most interpretive axis.

## Still open (one flag + consistency items)
- **L303** needs the blind leak-free self-preference re-run (RUNBOOK step 3) — a different pipeline (blinded prompt, self-vs-other paired comparison), not the fairwashing re-judge.
- **Results Table (~line 199)** and **Fig 3 caption** still carry old-anchor A2/A4 values — outside the 9 flags but worth updating for whole-paper consistency.

## Apply
`git apply main_rejudge.diff` from the repo root, or drop in `main_rejudge_edited.tex`.

_Cost: 3-vote run (votes 2+3 for 1,074 cells, vote 1 reused from the prior full re-judge) = $4.97; 2,085/2,148 calls parsed (97.1%)._


---
## Full-manuscript validation pass (beyond the 9 flags)
After discharging the flags, every numeric claim in main.tex was validated against source data and all judge-dependent tables/prose were updated to 3-vote consensus. See `VALIDATION_LEDGER.md` for the claim-by-claim audit. Net: 65 lines changed; judge-independent core validates exactly (one SD typo 4.04→4.14 fixed); two demographic effects change qualitatively (A4 race → ns, A4 gender → ε²0.059); per-type ordering flips (Abductive best); worst model flips (GPT→DeepSeek); GLM token-A2 correlation flips +0.17→−0.02 (null). Cross-judge/self-preference section untouched (separate pipeline).
