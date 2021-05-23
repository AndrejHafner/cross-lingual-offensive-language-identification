import math
import re
import time
from pathlib import Path

import numpy as np
import requests
import json

import pandas as pd

def get_tweets(ids, bearer_token):
    headers = {
        "Authorization": f"Bearer {bearer_token}"
    }
    query_path = "https://api.twitter.com/2/tweets"
    params = {
        "ids": ",".join(ids)
    }


    response = requests.get(query_path, headers=headers, params=params)
    if response.status_code not in [200, 304]:
        raise Exception(f"Request failed, error code: {response.status_code}\n {response.reason} \n {response.text}")

    return json.loads(response.text)

def filter_tweet_content(content):
    # Remove ' and "
    content = content.replace('"','').replace("'", "")
    content = " ".join([el for el in content.split(" ") if not el.startswith("@")])
    content = re.sub(r'http\S+', '', content)
    return content

def fetch_all_tweets(df, bearer_token, save_file_path):
    # Filter out wrong ids (float format from the dataframe)
    df = df[df["ID"].str.contains("^[0-9]+$")]

    tweet_ids_chunks = [list(el) for el in np.array_split(df["ID"], math.ceil(len(df) / 100))]
    i = 0
    all_tweets = []
    while i < len(tweet_ids_chunks):
        print(f"Fetching tweets - {i+1}/{len(tweet_ids_chunks)}")
        try:
            json_data = get_tweets(tweet_ids_chunks[i], bearer_token)
            tweets = json_data["data"]

            for tweet in tweets:
                tweet_metadata = df[df["ID"] == tweet["id"]]
                tweet_data = {
                    "id": tweet["id"],
                    "type": list(tweet_metadata["vrsta"]),
                    "target": list(tweet_metadata["tarča"]),
                    "annotators": list(tweet_metadata["Annotator"]),
                    "content": tweet["text"]
                }
                all_tweets.append(tweet_data)
        except Exception as e:
            print("Error occured:", e)
            time.sleep(60) # Wait 1 minute, then retry
            print("Retrying...")
            continue

        # Save each 10th iteration in case of failure
        if i % 50 == 0:
            with open(save_file_path, "w", encoding="utf-8") as f:
                json.dump(all_tweets, f, ensure_ascii=False)

        time.sleep(5) # Wait between requests in order to not reach the limit - 300 requests per 15 minutes

        i += 1



if __name__ == '__main__':
    bearer_token = "<ENTER_YOUR_TOKEN_HERE>"

    df_train = pd.read_csv("../data/slovenian-twitter-hatespeech/IMSyPP_SI_anotacije_training-clarin.csv", dtype=str)
    df_evaluation = pd.read_csv("../data/slovenian-twitter-hatespeech/IMSyPP_SI_anotacije_evaluation-clarin.csv")

    Path("../data/slovenian-twitter-hatespeech/").mkdir(parents=True, exist_ok=True)

    fetch_all_tweets(df_evaluation, bearer_token, "../data/slovenian-twitter-hatespeech/hate_speech_data_evaluation.json")
    fetch_all_tweets(df_train, bearer_token, "../data/slovenian-twitter-hatespeech/hate_speech_data_train.json")
