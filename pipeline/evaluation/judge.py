"""
The Explanation Gap — Module 3: LLM-as-Judge Evaluation Engine

Primary judge: Gemini via OpenRouter (async, rate-limited). Native Gemini API used as fallback.
Best-of-3 consensus — identical methodology to GranularityGap (Paper 2).

Usage:
    python evaluation/judge.py --input tests/data/explanation_results.csv --output tests/data/evaluation_results.csv
    python evaluation/judge.py --input tests/data/explanation_results.csv --output tests/data/evaluation_results.csv --votes 1
"""

import os
import sys
import json
import csv
import uuid
import datetime
import asyncio
import sys
if sys.platform.startswith("win"):
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
import argparse
import statistics
from typing import Dict, List, Any, Optional

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

from google import genai
try:
    from openai import AsyncOpenAI as _AsyncOpenAI
    _OPENAI_AVAILABLE = True
except ImportError:
    _OPENAI_AVAILABLE = False
import config
from evaluation.judge_prompts import build_judge_prompt, build_cohort_supplement

# OpenRouter fallback — SAME judge model via a different API route ONLY.
# The cross-model DeepSeek fallback was REMOVED: it silently averaged a second,
# harsher judge model (and one of the evaluated generators) into a consensus
# labelled "single-judge Gemini," contaminating 9.6% of the pilot rows. If
# Gemini is unreachable via both native and OpenRouter routes, the item is
# skipped (recorded as a -1 failure) rather than judged by a different model.
_OR_FALLBACK_MODELS = [
    "google/gemini-3-flash-preview",
]

# ---------------------------------------------------------------------------
# CSV Schema
# ---------------------------------------------------------------------------

EVAL_HEADER = [
    "eval_id", "explanation_id", "cohort_id", "model", "framing",
    "iteration", "explanation_type",
    "axis1_v1", "axis1_v2", "axis1_v3", "axis1_consensus", "axis1_stdev",
    "axis2_v1", "axis2_v2", "axis2_v3", "axis2_consensus", "axis2_stdev",
    "axis3_v1", "axis3_v2", "axis3_v3", "axis3_consensus", "axis3_stdev",
    "axis4_v1", "axis4_v2", "axis4_v3", "axis4_consensus", "axis4_stdev",
    "judge_reasoning_json", "judge_model", "timestamp",
]


# ---------------------------------------------------------------------------
# Judge Engine
# ---------------------------------------------------------------------------

class JudgeEngine:
    def __init__(self, input_csv: str, output_csv: str, n_votes: int = 3,
                 judge_model: str = None):
        self.input_csv  = input_csv
        self.output_csv = output_csv
        self.n_votes    = n_votes
        self.semaphore  = asyncio.Semaphore(5)  # conservative for Gemini rate limits
        self.judge_model = judge_model or config.JUDGE_MODEL_PRIMARY

        # Circuit breaker: after 2 consecutive Gemini native failures, route to OR
        self._gemini_fail_streak = 0
        self._gemini_circuit_open = False   # True = skip native Gemini, go straight to OR
        self._OR_TRIP_THRESHOLD = 2         # failures before tripping
        self._OR_RETRY_AFTER = 50           # re-probe Gemini every N successful OR calls
        self._or_success_count = 0

        # Init Gemini client
        if not config.GEMINI_API_KEY:
            raise ValueError("GEMINI_API_KEY not set.")
        self.client = genai.Client(api_key=config.GEMINI_API_KEY)

        # Init OpenRouter fallback client (used when Gemini 503s)
        self.or_client = None
        if _OPENAI_AVAILABLE and config.OPENROUTER_API_KEY:
            self.or_client = _AsyncOpenAI(
                api_key=config.OPENROUTER_API_KEY,
                base_url=config.OPENROUTER_BASE_URL,
            )
        else:
            print("  [WARN] OpenRouter fallback unavailable (missing key or openai package).")

        # Load Phase 1 internal bias residuals (marginal group effects)
        self.phase1_residuals = self._load_phase1_residuals()
        # Load intersectional cohort stats and variance decomposition
        self.cohort_stats    = self._load_cohort_stats()
        self.variance_decomp = self._load_variance_decomp()

        # Load explanation data
        self.explanations = self._load_explanations()
        self.completed    = self._load_completed()
        self._setup_csv()

    def _load_phase1_residuals(self) -> Optional[dict]:
        """Load Phase 1 internal bias residuals if available.

        Tries two paths in order:
          1. Relative to input CSV (production layout: explanation/data/ -> generation/data/)
          2. Project root fallback via config.PATHS (handles test/ layouts)
        """
        residuals_csv = os.path.join(
            os.path.dirname(os.path.dirname(self.input_csv)),
            "generation", "data", "phase1_bias_residuals.csv"
        )
        if not os.path.exists(residuals_csv):
            # Fallback: project root layout
            fallback = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "generation", "data", "phase1_bias_residuals.csv"
            )
            if os.path.exists(fallback):
                residuals_csv = fallback
            else:
                print(f"  Warning: Phase 1 residuals not found at {residuals_csv}")
                print(f"  Judge will use external PsychBench benchmarks only (degraded mode).")
                return None

        residuals = {}
        with open(residuals_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                residuals[row["group"]] = {
                    "internal_residual": float(row["internal_residual"]),
                    "internal_d": float(row["internal_d"]),
                    "direction": row["direction"],
                    "mean_phq8": float(row["mean_phq8"]),
                    "grand_mean": float(row["grand_mean"]),
                }
        print(f"  Loaded Phase 1 residuals for {len(residuals)} groups (dual ground-truth mode).")
        return residuals

    def _load_cohort_stats(self) -> Optional[dict]:
        """Load intersectional cohort-level PHQ-8 stats (phase1_cohort_stats.csv)."""
        stats_csv = config.PATHS.get("phase1_cohort_stats", "")
        if not stats_csv or not os.path.exists(stats_csv):
            print("  Warning: phase1_cohort_stats.csv not found.")
            print("  Run scripts/compute_phase1_cohort_stats.py to enable cohort-level anchors.")
            return None
        stats = {}
        with open(stats_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                stats[row["cohort_id"]] = {
                    "race":             row.get("race", ""),
                    "gender":           row.get("gender", ""),
                    "ses":              row.get("ses", ""),
                    "n":                int(row["n"]),
                    "mean_phq8":        float(row["mean_phq8"]),
                    "sd_phq8":          float(row.get("sd_phq8", 0.0) or 0.0),
                    "grand_mean":       float(row["grand_mean"]),
                    "cohort_residual":  float(row["cohort_residual"]),
                    "cohort_d":         float(row["cohort_d"]),
                }
        print(f"  Loaded cohort stats for {len(stats)} cohorts (intersectional ground-truth mode).")
        return stats

    def _load_variance_decomp(self) -> Optional[dict]:
        """Load ANOVA variance decomposition (phase1_variance_decomp.csv)."""
        vd_csv = config.PATHS.get("phase1_variance_decomp", "")
        if not vd_csv or not os.path.exists(vd_csv):
            print("  Warning: phase1_variance_decomp.csv not found — Type 3 FA extension degraded.")
            return None
        decomp = {}
        with open(vd_csv, "r", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                decomp[row["factor"]] = {
                    "ss":           row.get("ss", ""),
                    "df":           row.get("df", ""),
                    "eta_squared":  row.get("eta_squared", ""),
                    "pct_variance": row.get("pct_variance", ""),
                    "p_value":      row.get("p_value", ""),
                }
        print(f"  Loaded variance decomposition for {len(decomp)} factors (Type 3 FA mode).")
        return decomp

    def _load_explanations(self) -> List[Dict]:
        rows = []
        with open(self.input_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                rows.append(row)
        return rows

    def _load_completed(self) -> set:
        completed = set()
        if not os.path.exists(self.output_csv) or os.stat(self.output_csv).st_size == 0:
            return completed
        try:
            with open(self.output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if set(reader.fieldnames or []) != set(EVAL_HEADER):
                    return completed
                for row in reader:
                    completed.add(row["explanation_id"])
        except Exception:
            pass
        return completed

    def _setup_csv(self):
        if os.path.exists(self.output_csv) and os.stat(self.output_csv).st_size > 0:
            with open(self.output_csv, "r", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                if set(reader.fieldnames or []) == set(EVAL_HEADER):
                    return
        os.makedirs(os.path.dirname(self.output_csv), exist_ok=True)
        with open(self.output_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=EVAL_HEADER)
            writer.writeheader()

    async def _call_gemini_native(self, prompt: str) -> Dict:
        """One attempt at native Gemini API. Raises on failure."""
        response = await asyncio.wait_for(
            self.client.aio.models.generate_content(
                model=self.judge_model,
                contents=prompt,
                config=genai.types.GenerateContentConfig(
                    response_mime_type="application/json",
                    temperature=0.7,
                ),
            ),
            timeout=30.0
        )
        raw = response.text or ""
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0].strip()
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0].strip()
        data = json.loads(raw)
        scores = data.get("scores", {})
        for axis in ["axis1", "axis2", "axis4"]:
            v = scores.get(axis)
            if v is not None and v != -1 and (not isinstance(v, int) or v < 1 or v > 5):
                scores[axis] = max(1, min(5, int(v)))
        v3 = scores.get("axis3")
        if v3 is not None and v3 != -1 and (not isinstance(v3, int) or v3 < 1 or v3 > 5):
            scores["axis3"] = max(1, min(5, int(v3)))
        # Reject if all non-null scores are missing — caller will retry
        if all(scores.get(a) is None for a in ["axis1", "axis2", "axis4"]):
            raise ValueError("Scores object returned with all-null values")
        return {"scores": scores, "analysis": data.get("analysis", {})}

    async def _judge_once(self, prompt: str) -> Dict:
        """Single judge call with circuit breaker.
        After 2 consecutive Gemini native failures, routes all traffic to OpenRouter.
        Re-probes Gemini every 50 successful OR calls to detect recovery.
        """
        async with self.semaphore:
            # --- Circuit breaker: probe Gemini if it's time to retry ---
            if self._gemini_circuit_open and self._or_success_count >= self._OR_RETRY_AFTER:
                self._or_success_count = 0
                try:
                    result = await self._call_gemini_native(prompt)
                    self._gemini_fail_streak = 0
                    self._gemini_circuit_open = False
                    print("  [CIRCUIT CLOSED] Gemini native API recovered — switching back.")
                    return result
                except Exception:
                    pass  # still down, stay on OR

            # --- Primary: native Gemini (unless circuit is open) ---
            if not self._gemini_circuit_open:
                for attempt in range(config.MAX_RETRIES + 1):
                    try:
                        result = await self._call_gemini_native(prompt)
                        self._gemini_fail_streak = 0  # reset on success
                        return result
                    except Exception as e:
                        err_str = str(e)
                        is_rate = "429" in err_str or "503" in err_str or "RESOURCE_EXHAUSTED" in err_str
                        if attempt == config.MAX_RETRIES:
                            self._gemini_fail_streak += 1
                            if self._gemini_fail_streak >= self._OR_TRIP_THRESHOLD:
                                if not self._gemini_circuit_open:
                                    print(f"  [CIRCUIT OPEN] Gemini failed {self._gemini_fail_streak}x — routing to OpenRouter.")
                                self._gemini_circuit_open = True
                        else:
                            await asyncio.sleep(15.0 if is_rate else config.RETRY_DELAY * (attempt + 1))

            # --- Fallback / circuit-open path: OpenRouter ---
            if self.or_client is not None:
                result = await self._judge_once_openrouter(prompt)
                if result is not None:
                    self._or_success_count += 1
                    return result

            # Total failure — use -1 sentinel (NOT midpoint 3) so failures are
            # identifiable in analysis and do not attenuate real effect estimates.
            return {
                "scores": {"axis1": -1, "axis2": -1, "axis3": None, "axis4": -1},
                "analysis": {"error": "JUDGE_FAILURE: All judge routes failed (native Gemini + OpenRouter)"},
            }

    async def _judge_once_openrouter(self, prompt: str) -> Optional[Dict]:
        """Primary OpenRouter call. Attempts _OR_FALLBACK_MODELS in order.
        Returns None only if all models and retries fail."""
        for model_id in _OR_FALLBACK_MODELS:
            for attempt in range(3):
                try:
                    response = await asyncio.wait_for(
                        self.or_client.chat.completions.create(
                            model=model_id,
                            messages=[{"role": "user", "content": prompt}],
                            temperature=0.7,
                        ),
                        timeout=45.0,
                    )
                    raw = response.choices[0].message.content or ""
                    if "```json" in raw:
                        raw = raw.split("```json")[1].split("```")[0].strip()
                    elif "```" in raw:
                        raw = raw.split("```")[1].split("```")[0].strip()
                    data = json.loads(raw)
                    scores = data.get("scores", {})
                    for axis in ["axis1", "axis2", "axis4"]:
                        v = scores.get(axis)
                        if v is not None and v != -1 and (not isinstance(v, int) or v < 1 or v > 5):
                            scores[axis] = max(1, min(5, int(v)))
                    v3 = scores.get("axis3")
                    if v3 is not None and v3 != -1 and (not isinstance(v3, int) or v3 < 1 or v3 > 5):
                        scores["axis3"] = max(1, min(5, int(v3)))
                    if all(scores.get(a) is None for a in ["axis1", "axis2", "axis4"]):
                        raise ValueError("Scores object returned with all-null values")
                    print(f"    [FALLBACK OK] {model_id}")
                    return {"scores": scores, "analysis": data.get("analysis", {}), "_or_model": model_id}
                except Exception as e:
                    err_str = str(e)
                    if attempt == 2:
                        print(f"    [OR FAIL] {model_id}: {err_str[:120]}")
                    is_rate = "429" in err_str or "503" in err_str
                    is_limit = "403" in err_str or "limit exceeded" in err_str.lower()
                    if is_limit:
                        # Key limit — no point retrying this model, move to next
                        break
                    await asyncio.sleep(15.0 if is_rate else 5.0)
        return None

    def _resolve_judge_model(self, votes: list) -> str:
        """Returns judge_model string — notes OpenRouter fallback if any vote used it."""
        or_models = [v["_or_model"] for v in votes if "_or_model" in v]
        if not or_models:
            return self.judge_model
        unique = list(dict.fromkeys(or_models))
        return f"{self.judge_model}+OR({','.join(unique)})"

    @staticmethod
    def _truncate_explanation(text: str, max_chars: int = 5000) -> str:
        """Soft-truncate long explanations at a sentence boundary.

        Mitigates verbosity bias (Weakness #5): GLM produces 10-12x more tokens
        than other models, giving the judge more surface area for keyword hits.
        """
        if len(text) <= max_chars:
            return text
        # Find last sentence boundary before limit
        truncated = text[:max_chars]
        for sep in (". ", ".\n", "\n\n", "\n"):
            last_idx = truncated.rfind(sep)
            if last_idx > max_chars * 0.6:  # don't truncate too aggressively
                truncated = truncated[:last_idx + 1]
                break
        return truncated + f"\n[Note: Explanation truncated from {len(text)} to {len(truncated)} characters for evaluation]"

    async def _evaluate_explanation(self, exp_row: Dict):
        """Run Best-of-N consensus on one explanation."""
        exp_id = exp_row["explanation_id"]
        
        # Empty text check (Academic Rigor)
        text = exp_row.get("raw_explanation_text", "").strip()
        if not text or len(text) < 10:
            print(f"  [SKIPPING] {exp_id[:8]}...: Explanation text is empty or too short.")
            return

        if exp_id in self.completed:
            return

        # Soft-truncate verbose explanations before judging
        exp_row_copy = dict(exp_row)
        exp_row_copy["raw_explanation_text"] = self._truncate_explanation(
            exp_row.get("raw_explanation_text", "")
        )

        prompt = build_judge_prompt(
            exp_row_copy,
            phase1_residuals=self.phase1_residuals,
            cohort_stats=self.cohort_stats,
            variance_decomp=self.variance_decomp,
        )

        # Collect N independent votes
        vote_tasks = [self._judge_once(prompt) for _ in range(self.n_votes)]
        votes = await asyncio.gather(*vote_tasks)

        # Aggregate scores per axis
        row = {
            "eval_id": str(uuid.uuid4()),
            "explanation_id": exp_id,
            "cohort_id": exp_row["cohort_id"],
            "model": exp_row["model"],
            "framing": exp_row["framing"],
            "iteration": exp_row["iteration"],
            "explanation_type": exp_row["explanation_type"],
            "judge_model": self._resolve_judge_model(votes),
            "timestamp": datetime.datetime.now().isoformat(),
        }

        for axis_name in ["axis1", "axis2", "axis3", "axis4"]:
            axis_scores = []
            for i, v in enumerate(votes):
                score = v["scores"].get(axis_name)
                col = f"{axis_name}_v{i+1}"
                row[col] = score if score is not None else ""
                if score is not None:
                    axis_scores.append(score)

            # Pad missing votes (if n_votes < 3)
            for i in range(len(votes), 3):
                row[f"{axis_name}_v{i+1}"] = ""

            if axis_scores:
                row[f"{axis_name}_consensus"] = round(statistics.mean(axis_scores), 2)
                row[f"{axis_name}_stdev"] = (
                    round(statistics.stdev(axis_scores), 2) if len(axis_scores) > 1 else 0.0
                )
            else:
                row[f"{axis_name}_consensus"] = ""
                row[f"{axis_name}_stdev"] = ""

        # Store first vote's reasoning as representative
        row["judge_reasoning_json"] = json.dumps(votes[0].get("analysis", {}))

        await self._write_row(row)

        exp_type = exp_row["explanation_type"][:8]
        a1 = row.get("axis1_consensus", "?")
        a2 = row.get("axis2_consensus", "?")
        a4 = row.get("axis4_consensus", "?")
        display = config.MODEL_DISPLAY_NAMES.get(exp_row["model"], exp_row["model"])
        print(f"  [{exp_type:8s}] {display:20s} {exp_row['cohort_id']:20s} "
              f"A1={a1} A2={a2} A4={a4}")

    async def _write_row(self, row: Dict):
        for attempt in range(5):
            try:
                with open(self.output_csv, "a", newline="", encoding="utf-8") as f:
                    writer = csv.DictWriter(f, fieldnames=EVAL_HEADER)
                    writer.writerow(row)
                return
            except PermissionError:
                if attempt < 4:
                    await asyncio.sleep(1.0 * (attempt + 1))
                else:
                    raise

    async def run(self):
        from tqdm.asyncio import tqdm
        tasks = [self._evaluate_explanation(exp) for exp in self.explanations]
        print(f"  Total evaluations: {len(tasks)} (x{self.n_votes} votes each)")
        
        # Use tqdm to show a progress bar for the async tasks
        await tqdm.gather(*tasks, desc="Evaluating (Phase 3)", unit="eval")


# ---------------------------------------------------------------------------
# Cross-Validation Runtime (Deprecated)
# Migrated to standalone deepseek_judge.py and compute_cross_judge_reliability.py
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Explanation Gap — Module 3: Judge")
    parser.add_argument("--input", required=True, help="Path to explanation_results.csv")
    parser.add_argument("--output", required=True, help="Path to evaluation_results.csv")
    parser.add_argument("--votes", type=int, default=3, help="Votes per explanation (default: 3)")
    parser.add_argument("--debug-prompt", action="store_true",
                        help="Print the first generated judge prompt to stdout and exit (for inspection)")
    args = parser.parse_args()

    if not config.GEMINI_API_KEY:
        raise ValueError("GEMINI_API_KEY not set.")

    print(f"\n{'='*60}")
    print(f"  The Explanation Gap — Module 3: LLM-as-Judge Evaluation")
    print(f"{'='*60}")
    print(f"  Input:  {args.input}")
    print(f"  Output: {args.output}")
    print(f"  Judge:  {config.JUDGE_MODEL_PRIMARY} (Gemini native API)")
    print(f"  Votes:  {args.votes} per explanation")
    print(f"{'='*60}\n")

    engine = JudgeEngine(
        input_csv=args.input,
        output_csv=args.output,
        n_votes=args.votes,
    )

    if args.debug_prompt:
        if not engine.explanations:
            print("[DEBUG] No explanations loaded.")
        else:
            first = engine.explanations[0]
            first_copy = dict(first)
            first_copy["raw_explanation_text"] = engine._truncate_explanation(
                first.get("raw_explanation_text", "")
            )
            prompt = build_judge_prompt(
                first_copy,
                phase1_residuals=engine.phase1_residuals,
                cohort_stats=engine.cohort_stats,
                variance_decomp=engine.variance_decomp,
            )
            header = "\n" + "=" * 70 + "\n  DEBUG PROMPT (first explanation)\n" + "=" * 70
            footer = "=" * 70 + "\n"
            # Write with UTF-8 to handle special characters on Windows consoles
            out = sys.stdout.buffer if hasattr(sys.stdout, "buffer") else sys.stdout
            out.write((header + "\n" + prompt + "\n" + footer).encode("utf-8", errors="replace"))
        return

    asyncio.run(engine.run())
    print(f"\nJudge evaluation complete. Output: {args.output}")


if __name__ == "__main__":
    main()
