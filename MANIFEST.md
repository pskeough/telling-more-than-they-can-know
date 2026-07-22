# MANIFEST — explanation-gap export

Curated allowlist export, staged 2026-07-23. Source project:
`C:\Research\ExplinationGap\UpdatedExplinationGap` (originals untouched).
Secrets scan: **clean** (`_pipeline\scan-secrets.ps1`, 2026-07-23).

## Provenance gate

Manuscript is the post-gauntlet build: canonical `PilotPaperTex\main.tex`
(3-vote internal-anchor consensus numbers) with blockers B1–B3 + C5 fixed
2026-07-23 (see `_gauntlet\CRITIQUE.md` / `DECISIONS.md`). PDF compiled with
tectonic 2026-07-23; verified to carry n=1,072 / A2 2.53 / A4 3.02 / 40.3%
and title "Named but Not Weighted". Old-run values appear only in the
revisions appendix (intentional).

## Copied

| Export path | Source |
|---|---|
| paper/main.tex, paper/main.pdf | ...\PsychExplainedPaper\PilotPaperTex\ |
| paper/figures/ (6 files: Fig1–4 _v3, Fig3b, Fig5 _v2) | ...\PilotPaperTex\figures\ |
| pipeline/generation/ (main.py, generate_identities.py, registries/) | ...\PsychExplainedPaper\generation\ |
| pipeline/explanation/prompt_templates.py | ...\PsychExplainedPaper\explanation\ |
| pipeline/evaluation/ (judge.py, judge_prompts.py, cross-judge + self-preference + blind analyses) | ...\PsychExplainedPaper\evaluation\ |
| pipeline/analysis/ (corrected_analysis.py, run_analysis.py) | ...\PsychExplainedPaper\analysis\ |
| results/rejudge_3vote_consensus.csv — **receipt of record** for all judge-dependent numbers | ...\UpdatedExplinationGap\artifacts\ |
| results/fa_weights_4model.csv — receipt of record for FA inversion | ...\UpdatedExplinationGap\artifacts\ |
| results/cross_instrument_bias.csv, VALIDATION_LEDGER.md, rejudge_flag_reconciliation.md | ...\UpdatedExplinationGap\artifacts\ |
| results/generation_results.csv (1,458 rows), phase1_* (bias residuals, variance decomp, cohort stats) | ...\generation\data\ |
| results/explanation_results.csv (5,594 rows) | ...\explanation\data\ |

## Deliberately excluded

- `.env` (credentials), venvs, `__pycache__`
- `gitshit\` duplicate tree, `tests\Backup\`, backup/superseded tex versions
- Superseded intermediates flagged by gauntlet C1 (`rejudge_FULL_1072.csv`,
  `02a_fa_weights_overall.csv`) — wrong-join hazards; consensus CSV is the receipt of record
- Old figure generations (May-5 originals, v2 set except Fig5_v2)
- `evaluation_results.csv` single-judge primary (superseded by 3-vote consensus; obtainable on request)
