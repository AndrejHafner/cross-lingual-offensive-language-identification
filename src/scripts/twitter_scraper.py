import time

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
    return json.loads(response.text)

def fetch_all_tweets(df, bearer_token):

    tweet_ids_chunks = [list(el) for el in np.array_split(df["ID"], int(len(df) / 100))]

    i = 0
    while i < len(tweet_ids_chunks):
        print(f"Fetching tweets - {i+1}/{len(tweet_ids_chunks)}")
        try:
            json_data = get_tweets(tweet_ids_chunks[i], bearer_token)
        except Exception as e:
            print("Error occured:", e)
            time.sleep(60) # Wait 1 minute, then retry
            continue

        time.sleep(5) # Wait between requests in order to not reach the limit - 300 requests per 15 minutes
        i += 1



if __name__ == '__main__':
    bearer_token = "AAAAAAAAAAAAAAAAAAAAAFNzPAEAAAAAwVIc4T9ctyvwd7d6Xi5m4LqcmmA%3DhxDJIBCTEeUua2Zuarpez1rknqAglQ5AbWIQultC8hEPKDlnuh"

    df_train = pd.read_csv("../data/slovenian-twitter-hatespeech/IMSyPP_SI_anotacije_training-clarin.csv", dtype=str)
    df_evaluation = pd.read_csv("../data/slovenian-twitter-hatespeech/IMSyPP_SI_anotacije_training-clarin.csv")

    fetch_all_tweets(df_train, bearer_token)
