import datetime
import re
import numpy as np
import pandas as pd
import json
import emoji

def read_fox_comments_dataset():
    """
    Read the FOX news comments dataset
    :return: Dataframe containing the comment and label for hate or not (1,0)
    """
    with open("../data/fox-news-comments-master/fox-news-comments.json") as f:
        df = pd.DataFrame([json.loads(line) for line in f.readlines()])

    # Remove irrelevant columns
    for col_name in ["title", "succ", "meta", "user", "mentions", "prev"]:
        del df[col_name]

    return df

def remove_emojies(text):
    return emoji.get_emoji_regexp().sub(r'', text)

def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Function to calculate the accuracy of our predictions vs labels
def flat_accuracy(preds, labels):
    pred_flat = np.argmax(preds, axis=1).flatten()
    labels_flat = labels.flatten()
    return np.sum(pred_flat == labels_flat) / len(labels_flat)

def read_white_supremacist_dataset():
    with open("../data/hate-speech-white-supremacist/hate_speech_white_supremacist_forum_dataset.json", "r") as f:
        data = json.load(f)
        sentences = []
        for comment in data:
            for sentence in comment["comment"]:
                sentences.append((comment["id"], sentence["sentence"], sentence["hate"]))
        df = pd.DataFrame()
        df["comment_id"] = [el[0] for el in sentences]
        df["sentence"] = [el[1] for el in sentences]
        df["hateful"] = [el[2] for el in sentences]
        return df

def filter_gab_reddit_comment(comment):
    comment = re.sub(r'http\S+', '', comment)
    comment = re.sub(r'\t', '', comment)
    comment = ".".join(comment.split(".")[1:])
    comment = remove_emojies(comment)
    return comment.strip()

def parse_gab_reddit_dataset(filename):
    df = pd.read_csv(filename, dtype=str)

    all_comments = []
    for idx, row in df.iterrows():
        try:
            comments = row["text"]
            hate_speech_idx = json.loads(row["hate_speech_idx"]) if not pd.isnull(row["hate_speech_idx"]) and "n/a" not in row["hate_speech_idx"] else []

            comments_row = [(int(el.split(".")[0]), filter_gab_reddit_comment(el)) for el in comments.split("\n") if len(el.strip()) > 0]
            comments_row_labeled = [(el[0] in hate_speech_idx, el[1]) for el in comments_row if len(el[1]) > 0]
            all_comments += comments_row_labeled
        except:
            print("Error occured, continuing..")

    df_cleaned = pd.DataFrame()
    df_cleaned["comment"] = [el[1] for el in all_comments]
    df_cleaned["label"] = [el[0] for el in all_comments]

    return df_cleaned

