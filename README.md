# Telling More Than They Can Know
### Verbal Reports on Model Processes in Clinical Scoring (the "Explanation Gap" pilot)

**Do LLMs faithfully explain their own biased clinical judgments?** A four-model factorial
audit (GPT-4o-mini, Gemini-3-Flash, DeepSeek-V3, GLM-4.7) of whether models' self-explanations
reveal or conceal the demographic drivers of their psychiatric-instrument scoring.

📄 **Paper:** [`paper/main.pdf`](paper/main.pdf) · Patrick S. Keough · working paper / preprint in preparation

## Headline findings

- **The bias is large and real:** across 18 intersectional cohorts, low-SES vignettes score
  **5.78 PHQ-8 points higher** than high-SES (Cohen's *d* = 1.95, *p* < .001). Variance
  decomposition attributes **48.75% of score variance to SES**, versus 2.22% to which model
  generated the score.
- **The explanations conceal it:** in feature-attribution explanations, models credit SES with
  only ~5–24% of the score: a **feature-attribution inversion** against the ~49% it actually
  drives. Demographics are *named* but not *weighted*.
- **Honest instrument caveats, disclosed up front:** the fairwashing rubric (A4) is unvalidated
  and SES-confounded; judge coverage is 19.2% of explanations; judge-dependent tables use a
  3-vote consensus (n = 1,072). The judge-independent core (SES bias, variance decomposition,
  FA inversion) does not depend on these.

## Repository layout

```
paper/       manuscript source (main.tex) + compiled PDF + figures
pipeline/    generation, explanation, judging, and analysis code
results/     receipts of record: consensus judge scores, FA weights,
             cohort stats, bias residuals, validation ledger
MANIFEST.md  full provenance: what was copied from where, what was excluded and why
```

Every number in the paper traces to a file in `results/`; `results/VALIDATION_LEDGER.md`
records the claim-level verification, and the paper's revisions appendix documents all
corrections applied during self-audit (including a full 3-vote re-judge of the primary set).

## Context

Paper 3 of a four-part research line auditing LLM behavior in clinical-psychological
simulation ([PsychBench](https://arxiv.org/abs/2604.17359) → Explanation Gap → mechanistic
follow-ups). Program index: [Research_Collection_Patrick_Keough](https://github.com/pskeough/Research_Collection_Patrick_Keough).

## Authorship note

Drafts prepared with AI assistance under the author's direction; all research questions,
experimental design, analysis decisions, and claims are the author's own, and every empirical
claim is verified against the data in `results/` (see `MANIFEST.md` provenance gate).

## License

Code: MIT ([LICENSE](LICENSE)) · Paper text, figures and derived data: CC BY-NC-ND 4.0 ([LICENSE-DATA](LICENSE-DATA))
