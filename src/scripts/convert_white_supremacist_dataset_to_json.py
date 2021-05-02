import json
from collections import Counter

import pandas as pd
from tqdm import tqdm

annotations = pd.read_csv("../data/hate-speech-white-supremacist/hate-speech-dataset-master/annotations_metadata.csv")

df = annotations[["file_id", "label"]]
df["label"] = [1 if el == "hate" else 0 for el in df["label"]]

distinct_comments = list(set([el.split("_")[0] for el in df["file_id"]]))

comments = []
for comment_id in tqdm(distinct_comments):
    comment_files = df[df["file_id"].str.contains(comment_id)]
    sentences = []
    for idx, row in comment_files.iterrows():
        with open(f"../data/hate-speech-white-supremacist/hate-speech-dataset-master/all_files/{row['file_id']}.txt", "r") as f:
            sentence = "".join(f.readlines()).strip()
        sentence_entry = {
            "index": int(row['file_id'].split("_")[1]) - 1,
            "hate": row['label'],
            "sentence": sentence
        }
        sentences.append(sentence_entry)

    comments.append({"id": comment_id, "comment": sentences})

with open("../data/hate-speech-white-supremacist/hate_speech_white_supremacist_forum_dataset.json", "w") as f:
    json.dump(comments, f)



