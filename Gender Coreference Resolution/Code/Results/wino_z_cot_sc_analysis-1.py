import json
from collections import defaultdict

INPUT_FILE = "./winogender_results_z_cot_sc-3.json"
OUTPUT_FILE = "Analysis/winogender_bias_analysis-3.txt"

# Define pronoun groups
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

def analyze():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # model -> prompt_type -> counts
    results = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))

    for sentence, model_preds in data.items():
        for model, prompts in model_preds.items():
            for prompt_type, prediction in prompts.items():
                if prompt_type == "cot_sc":
                    pred = prediction["majority_vote"]
                else:
                    pred = prediction
                category = categorize_pronoun(pred)
                results[model][prompt_type][category] += 1
                results[model][prompt_type]["total"] += 1

    # Print and save analysis
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for model in results:
            out.write(f"\n=== Model: {model} ===\n")
            print(f"\n=== Model: {model} ===")
            for prompt_type in results[model]:
                counts = results[model][prompt_type]
                total = counts["total"]
                neutral = counts["neutral"]
                male = counts["male"]
                female = counts["female"]
                unknown = counts["unknown"]

                accuracy = neutral / total if total else 0
                male_bias = male / total if total else 0
                female_bias = female / total if total else 0

                summary = (
                    f"\n-- Prompt Type: {prompt_type} --\n"
                    f"Total: {total}\n"
                    f"Neutral: {neutral}, Male: {male}, Female: {female}, Unknown: {unknown}\n"
                    f"Accuracy (Neutral): {accuracy:.2f}\n"
                    f"Male Bias: {male_bias:.2f}\n"
                    f"Female Bias: {female_bias:.2f}\n"
                )

                print(summary)
                out.write(summary)

if __name__ == "__main__":
    analyze()
