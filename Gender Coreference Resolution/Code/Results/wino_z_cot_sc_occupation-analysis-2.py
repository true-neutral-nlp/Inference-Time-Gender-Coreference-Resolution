import json
from collections import defaultdict
import re

INPUT_FILE = "./winogender_results_z_cot_sc-1.json"
OUTPUT_FILE = "Analysis/biased_occupations_by_llm_prompt-with-n-unk.txt"

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

def analyze_bias():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Structure: model -> prompt_type -> {'male': [occupations], 'female': [occupations]}
    bias_results = defaultdict(lambda: defaultdict(lambda: defaultdict(set)))

    # Structure: occupation -> model -> prompt_type -> category count
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

    for occupation, model_data in occ_model_prompt_dist.items():
        for model, prompt_data in model_data.items():
            for prompt_type, counts in prompt_data.items():
                male = counts.get("male", 0)
                female = counts.get("female", 0)
                neutral = counts.get("neutral", 0)
                unknown = counts.get("unknown", 0)

                if male > female and male > neutral and male > unknown:
                    bias_results[model][prompt_type]["male"].add(occupation)
                elif female > male and female > neutral and female > unknown:
                    bias_results[model][prompt_type]["female"].add(occupation)
                elif neutral > male and neutral > female and neutral > unknown:
                    bias_results[model][prompt_type]["neutral"].add(occupation)
                elif unknown > male and unknown > female and unknown > neutral:
                    bias_results[model][prompt_type]["unknown"].add(occupation)

    # Write to file
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        out.write("=== Gender-Biased Occupations by LLM & Prompt Type ===\n\n")
        for model in bias_results:
            out.write(f"Model: {model}\n")
            for prompt_type in bias_results[model]:
                out.write(f"  Prompt: {prompt_type}\n")
                male_bias = sorted(bias_results[model][prompt_type]["male"])
                female_bias = sorted(bias_results[model][prompt_type]["female"])
                neutral_bias = sorted(bias_results[model][prompt_type]["neutral"])
                unknown_bias = sorted(bias_results[model][prompt_type]["unknown"])

                out.write(f"    Male-biased occupations ({len(male_bias)}): {', '.join(male_bias)}\n")
                out.write(f"    Female-biased occupations ({len(female_bias)}): {', '.join(female_bias)}\n")
                out.write(f"    Neutral-biased occupations ({len(neutral_bias)}): {', '.join(neutral_bias)}\n")
                out.write(f"    Unknown-biased occupations ({len(unknown_bias)}): {', '.join(unknown_bias)}\n")

               
            out.write("\n")

    print(f"✅ Bias analysis complete. Output written to {OUTPUT_FILE}")

if __name__ == "__main__":
    analyze_bias()
