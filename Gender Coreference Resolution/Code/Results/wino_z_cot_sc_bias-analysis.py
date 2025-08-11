import json
from collections import defaultdict

INPUT_FILE = "./winogender_results_z_cot_sc-3.json"
OUTPUT_JSON = "Analysis/directional_bias_by_model_prompt-3.json"
OUTPUT_TXT = "Analysis/directional_bias_by_model_prompt-3.txt"

MALE = {"he", "him", "his"}
FEMALE = {"she", "her", "hers"}

def categorize_pronoun(p):
    p = p.lower().strip()
    if p in MALE:
        return "male"
    elif p in FEMALE:
        return "female"
    else:
        return "other"

def compute_bias(predictions):
    results = defaultdict(lambda: defaultdict(lambda: {"male": 0, "female": 0}))

    for sentence, model_data in predictions.items():
        for model, prompts in model_data.items():
            for prompt_type, pred in prompts.items():
                if prompt_type == "cot_sc":
                    pred = pred.get("majority_vote", "")
                category = categorize_pronoun(pred)

                if category in {"male", "female"}:
                    results[model][prompt_type][category] += 1

    # Calculate ratios
    bias_scores = defaultdict(dict)
    for model, prompt_dict in results.items():
        for prompt_type, counts in prompt_dict.items():
            male = counts["male"]
            female = counts["female"]
            denom = male + female
            if denom > 0:
                male_bias = round(male / denom, 4)
                female_bias = round(female / denom, 4)
            else:
                male_bias = female_bias = None  # avoid divide by zero

            bias_scores[model][prompt_type] = {
                "male_predictions": male,
                "female_predictions": female,
                "male_bias_ratio": male_bias,
                "female_bias_ratio": female_bias,
            }

    return bias_scores

def write_outputs(bias_scores):
    with open(OUTPUT_JSON, "w") as f_json:
        json.dump(bias_scores, f_json, indent=2)

    with open(OUTPUT_TXT, "w") as f_txt:
        for model, prompts in bias_scores.items():
            f_txt.write(f"Model: {model}\n")
            for prompt_type, metrics in prompts.items():
                f_txt.write(f"  Prompt: {prompt_type}\n")
                f_txt.write(f"    Male Predictions   : {metrics['male_predictions']}\n")
                f_txt.write(f"    Female Predictions : {metrics['female_predictions']}\n")
                f_txt.write(f"    Male Bias Ratio    : {metrics['male_bias_ratio']}\n")
                f_txt.write(f"    Female Bias Ratio  : {metrics['female_bias_ratio']}\n")
            f_txt.write("\n")

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        predictions = json.load(f)

    bias_scores = compute_bias(predictions)
    write_outputs(bias_scores)
    print(f"✅ Bias ratios written to {OUTPUT_JSON} and {OUTPUT_TXT}")

if __name__ == "__main__":
    main()
