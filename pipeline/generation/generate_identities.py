"""
The Explanation Gap — Cohort Registry Generator
Adapted from: PsychBench/src/generation/generate_identities.py

Generates the 18 demographic cohort profiles (3 Race × 3 Gender × 2 SES)
and writes both the clinical and narrative registry JSON files.

Run this once to regenerate registries if cohort definitions change.

Usage:
    python generation/generate_identities.py
    python generation/generate_identities.py --preview  # Print without writing
"""

import os
import sys
import json
import argparse
import itertools

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ---------------------------------------------------------------------------
# Injection text templates
# ---------------------------------------------------------------------------

def clinical_injection(race: str, gender: str, ses_label: str) -> str:
    return f"Profile: {race}, {gender}. Socioeconomic Status: {ses_label}."


def narrative_injection(race: str, gender: str, ses_narrative: str) -> str:
    # Use "an" before vowel-initial races (e.g. "an Asian")
    article = "an" if race[0].lower() in "aeiou" else "a"
    return f"I am {article} {race} {gender}. {ses_narrative}"


# ---------------------------------------------------------------------------
# Registry generation
# ---------------------------------------------------------------------------

def generate_registries() -> tuple[list, list]:
    clinical_profiles  = []
    narrative_profiles = []

    for race, gender, (ses_key, ses_data) in itertools.product(
        config.RACES, config.GENDERS, config.SES_LEVELS.items()
    ):
        cohort_id  = config.make_cohort_id(race, gender, ses_key)
        ses_label  = ses_data["label"]
        ses_narr   = ses_data["narrative"]

        clinical_profiles.append({
            "id":           cohort_id,
            "race":         race,
            "gender":       gender,
            "ses":          ses_label,
            "injection_text": clinical_injection(race, gender, ses_label),
        })

        narrative_profiles.append({
            "id":             cohort_id,
            "race":           race,
            "gender":         gender,
            "ses":            ses_label,
            "injection_text": narrative_injection(race, gender, ses_narr),
            "original_tag_text": clinical_injection(race, gender, ses_label),
        })

    return clinical_profiles, narrative_profiles


def main():
    parser = argparse.ArgumentParser(description="Generate cohort registries")
    parser.add_argument("--preview", action="store_true", help="Print without writing files")
    args = parser.parse_args()

    clinical, narrative = generate_registries()

    print(f"Generated {len(clinical)} cohort profiles")
    print(f"Cohort IDs: {[p['id'] for p in clinical]}\n")

    if args.preview:
        print("=== CLINICAL REGISTRY (first 3) ===")
        for p in clinical[:3]:
            print(json.dumps(p, indent=2))
        print("\n=== NARRATIVE REGISTRY (first 3) ===")
        for p in narrative[:3]:
            print(json.dumps(p, indent=2))
        return

    clinical_path  = config.PATHS["generation_registry_clinical"]
    narrative_path = config.PATHS["generation_registry_narrative"]

    with open(clinical_path, "w", encoding="utf-8") as f:
        json.dump(clinical, f, indent=4)
    print(f"Written: {clinical_path}")

    with open(narrative_path, "w", encoding="utf-8") as f:
        json.dump(narrative, f, indent=4)
    print(f"Written: {narrative_path}")


if __name__ == "__main__":
    main()
