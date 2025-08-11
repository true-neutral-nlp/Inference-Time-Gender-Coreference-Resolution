import csv

input_path = "../Data/Winogender Schemas/data/templates.tsv"
output_path = "../Data/Winogender Schemas/data/prepared_sentences.txt"

# Define a placeholder to signal where the model should fill in the pronoun
PRONOUN_BLANK = "___"

with open(input_path, newline='', encoding='utf-8') as csvfile:
    reader = csv.DictReader(csvfile, delimiter='\t')
    processed_sentences = []

    for row in reader:
        sentence = row["sentence"]
        sentence = sentence.replace("$OCCUPATION", row["occupation(0)"])
        sentence = sentence.replace("$PARTICIPANT", row["other-participant(1)"])
        sentence = sentence.replace("$NOM_PRONOUN", PRONOUN_BLANK)
        sentence = sentence.replace("$POSS_PRONOUN", PRONOUN_BLANK)
        sentence = sentence.replace("$ACC_PRONOUN", PRONOUN_BLANK)
        processed_sentences.append(sentence)

# Write the output to a file
with open(output_path, "w", encoding="utf-8") as outfile:
    for sent in processed_sentences:
        outfile.write(sent + "\n")
