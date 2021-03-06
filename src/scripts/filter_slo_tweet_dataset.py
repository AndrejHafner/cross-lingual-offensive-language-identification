import json
import re
import emoji
import pandas as pd


def filter_tweet_content(content):
    # Remove ' and "
    content = content.replace('"','').replace("'", "")
    content = " ".join([el for el in content.split(" ") if not el.startswith("@")])
    content = re.sub(r'http\S+', '', content)
    content = emoji.demojize(content)
    return content

def parse_tweet_data(data):
    content = filter_tweet_content(data["content"])
    types = [int(re.findall(r'\b\d+\b', el)[0]) for el in data["type"] if isinstance(el, str)]
    if (len(types) >= 2 and types[0] == types[1]) or len(types) == 0:
        return None

    type = types[0]

    return {
        "id": data["id"],
        "type": type,
        "content": content.replace("\n", "").strip()
    }


if __name__ == '__main__':
    with open("../data/slovenian-twitter-hatespeech/hate_speech_data_train.json", "r") as f:
        train = json.load(f)


    with open("../data/slovenian-twitter-hatespeech/hate_speech_data_evaluation.json", "r") as f:
        evaluation = json.load(f)

    combined = train + evaluation

    combined_filtered = []
    filtered_cnt = 0
    for el in combined:
        tweet_data = parse_tweet_data(el)
        if tweet_data is None:
            filtered_cnt += 1
            continue
        combined_filtered.append(tweet_data)

    pd.DataFrame(combined_filtered).to_csv("../data/slovenian-twitter-hatespeech/slo-twitter-test.csv",index=False, header=True)


