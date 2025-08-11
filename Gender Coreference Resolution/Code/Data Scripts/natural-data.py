import json

# File paths
input_path = "../Data/Natural Sentence Prompts/professions_prompts.json"
output_path = "../Data/Natural Sentence Prompts/prepared_profession_sentences.txt"

# Load the JSON data
with open(input_path, "r", encoding="utf-8") as infile:
    data = json.load(infile)

# Replace `{}` with the key (occupation) and write to file
with open(output_path, "w", encoding="utf-8") as outfile:
    for occupation, sentence in data.items():
        filled_sentence = sentence.replace("{}", occupation)
        outfile.write(filled_sentence + "\n")
