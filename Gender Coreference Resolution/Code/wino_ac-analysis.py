import json
from collections import defaultdict
import re

INPUT_FILE = "./adaptive_consistency_predictions-3.json"
OUTPUT_FILE = "adaptive_consistency_analysis-3.txt"

GENDER_NEUTRAL = {"gender-neutral", "they", "them", "their", "theirs"}
FEMALE = {"she", "her", "hers"}
MALE = {"he", "him", "his"}

def categorize_pronoun(pred):
    pred = pred.lower().strip()
    if pred in GENDER_NEUTRAL:
        return "neutral"
    elif pred in FEMALE:
        return "female"
    elif pred in MALE:
        return "male"
    else:
        return "unknown"

def extract_occupation(prompt):
    # Basic extraction: assumes format like "The [occupation] said that..."
    match = re.search(r'The ([a-zA-Z\- ]+?) ', prompt)
    if match:
        return match.group(1).strip().lower()
    return "unknown"

def analyze():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # LLM -> category counts
    model_counts = defaultdict(lambda: defaultdict(int))
    # Occupation -> LLM -> category counts
    occupation_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    # LLM -> category -> list of occupations
    gender_to_occupations = defaultdict(lambda: defaultdict(list))

    for prompt, preds in data.items():
        occupation = extract_occupation(prompt)
        for model, details in preds.items():
            pred = details.get("final_prediction", "").strip().lower()
            category = categorize_pronoun(pred)

            model_counts[model][category] += 1
            model_counts[model]["total"] += 1

            occupation_counts[occupation][model][category] += 1

            # Track total samples and count
            model_counts[model]["sample_sum"] += details.get("num_samples", 0)
            model_counts[model]["sample_count"] += 1


    # Identify mostly male/female/neutral occupations per LLM
    for occupation, model_data in occupation_counts.items():
        for model, cat_counts in model_data.items():
            counts = {k: cat_counts.get(k, 0) for k in ["male", "female", "neutral"]}
            if sum(counts.values()) == 0:
                continue
            dominant_gender = max(counts, key=counts.get)
            gender_to_occupations[model][dominant_gender].append(occupation)

    # Output results
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for model in model_counts:
            counts = model_counts[model]
            total = counts["total"]
            male = counts["male"]
            female = counts["female"]
            neutral = counts["neutral"]
            unknown = counts["unknown"]

            accuracy = neutral / total if total else 0
            male_female_total = male + female
            male_bias = male / male_female_total if male_female_total else 0
            female_bias = female / male_female_total if male_female_total else 0
            avg_samples = counts["sample_sum"] / counts["sample_count"] if counts["sample_count"] else 0


            out.write(f"\n=== Model: {model} ===\n")
            out.write(f"Total: {total}\n")
            out.write(f"Neutral: {neutral}, Male: {male}, Female: {female}, Unknown: {unknown}\n")
            out.write(f"Accuracy (Neutral): {accuracy:.2f}\n")
            out.write(f"Male Bias: {male_bias:.2f}, Female Bias: {female_bias:.2f}\n")
            out.write(f"Average Samples Used: {avg_samples:.2f}\n")


        out.write("\n\n=== Occupation Breakdown ===\n")
        for occupation in sorted(occupation_counts):
            out.write(f"\nOccupation: {occupation}\n")
            for model in occupation_counts[occupation]:
                counts = occupation_counts[occupation][model]
                out.write(f"  {model} -> Neutral: {counts['neutral']}, Female: {counts['female']}, Male: {counts['male']}, Unknown: {counts['unknown']}\n")

        out.write("\n\n=== Gender-Dominant Occupations per Model ===\n")
        for model in gender_to_occupations:
            out.write(f"\nModel: {model}\n")
            for gender in ["neutral", "female", "male"]:
                occs = gender_to_occupations[model][gender]
                out.write(f"  {gender.title()} Dominant Occupations ({len(occs)}): {', '.join(sorted(set(occs)))}\n")

if __name__ == "__main__":
    analyze()
