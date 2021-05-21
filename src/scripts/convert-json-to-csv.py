import pandas
import json

if __name__ == '__main__':
    with open("../data/slovenian-twitter-hatespeech/hate_speech_eval_filtered_translated_final.json", "r") as f:
        data = json.load(f)

    pandas.DataFrame(data).to_csv("../data/datasets/slo-twitter-dataset-translated.csv", header=True, index=False)