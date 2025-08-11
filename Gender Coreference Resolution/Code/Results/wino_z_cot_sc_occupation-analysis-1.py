import json
from collections import defaultdict
import re

INPUT_FILE = "./winogender_results_z_cot_sc-3.json"
OUTPUT_FILE = "Analysis/occupation_gender_distribution_by_llm_prompt-3.txt"

GENDER_NEUTRAL = {"they", "them", "their", "theirs"}
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

def extract_occupation(sentence):
    match = re.search(r"The ([a-zA-Z\-]+)", sentence)
    if match:
        return match.group(1).lower()
    return "UNKNOWN"

def analyze():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # occupation -> model -> prompt_type -> {category -> count}
    occ_model_prompt_dist = defaultdict(lambda: defaultdict(lambda: defaultdict(lambda: defaultdict(int))))

    for sentence, model_preds in data.items():
        occupation = extract_occupation(sentence)

        for model, prompts in model_preds.items():
            for prompt_type, prediction in prompts.items():
                if prompt_type == "cot_sc":
                    pred = prediction["majority_vote"]
                else:
                    pred = prediction

                category = categorize_pronoun(pred)
                occ_model_prompt_dist[occupation][model][prompt_type][category] += 1
                occ_model_prompt_dist[occupation][model][prompt_type]["total"] += 1

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=== Occupation-wise Gender Distribution by LLM & Prompt Type ===\n\n")
        for occupation in sorted(occ_model_prompt_dist):
            out.write(f"Occupation: {occupation.title()}\n")
            for model in occ_model_prompt_dist[occupation]:
                out.write(f"  Model: {model}\n")
                for prompt_type in occ_model_prompt_dist[occupation][model]:
                    counts = occ_model_prompt_dist[occupation][model][prompt_type]
                    total = counts.get("total", 0)
                    male = counts.get("male", 0)
                    female = counts.get("female", 0)
                    neutral = counts.get("neutral", 0)
                    unknown = counts.get("unknown", 0)

                    valid_total = male + female + neutral
                    accuracy = (neutral / valid_total) if valid_total > 0 else 0.0
                    male_bias = male / valid_total if valid_total > 0 else 0.0
                    female_bias = female / valid_total if valid_total > 0 else 0.0

                    out.write(f"    Prompt: {prompt_type}\n")
                    out.write(f"      Total: {total} | Neutral: {neutral} | Male: {male} | Female: {female} | Unknown: {unknown}\n")
                    out.write(f"      Accuracy: {accuracy:.2f} | Male Bias: {male_bias:.2f} | Female Bias: {female_bias:.2f}\n")
            out.write("\n")

if __name__ == "__main__":
    analyze()
