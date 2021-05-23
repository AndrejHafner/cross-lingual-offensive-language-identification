import json
from multiprocessing import Pool

import pandas as pd
import six
from google.cloud import translate_v2 as translate
from tqdm import tqdm

translate_client = translate.Client()

def translate_text(text, source_lang="sl", target_lang="en"):

    text = text.replace("\n", " ").strip()
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")
    # Translate strings from english to slovenian, in which case this method
    # will return a sequence of results for each text.
    result = translate_client.translate(text, target_language=target_lang, source_language=source_lang, format_='text')

    return result["translatedText"]

def pool_func(comment):
    translated = translate_text(comment["content"])
    comment["content"] = translated
    return comment


def save_json(filename, obj):
    with open(filename, "w") as f:
        json.dump(obj, f, ensure_ascii=False)

if __name__ == '__main__':


    df = pd.read_csv("../../data/slovenian-twitter-hatespeech/slo-twitter-test.csv")
    data = df.to_dict("records")

    data_translated = []

    pool = Pool(processes=8)
    for idx, translated_comment in tqdm(enumerate(pool.imap(pool_func, data)), total=len(data)):
        data_translated.append(translated_comment)

    pd.DataFrame(data_translated).to_csv("../../data/slovenian-twitter-hatespeech/slo-twitter-translated.csv", index=False, header=True)
