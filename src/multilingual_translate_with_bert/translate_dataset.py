import json
from multiprocessing import Pool

import six
from google.cloud import translate_v2 as translate
from tqdm import tqdm

translate_client = translate.Client()


def translate_text(text, source_lang="sl", target_lang="en"):

    text = text.replace("\n", " ").strip()
    if isinstance(text, six.binary_type):
        text = text.decode("utf-8")

    # Text can also be a sequence of strings, in which case this method
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


    with open("../data/slovenian-twitter-hatespeech/hate_speech_slo_eval_filtered.json", "r") as f:
        data = json.load(f)

    data_translated = []

    pool = Pool(processes=8)
    for idx, translated_comment in tqdm(enumerate(pool.imap(pool_func, data)), total=len(data)):
        data_translated.append(translated_comment)
        if idx % 100 == 0:
            save_json("../data/slovenian-twitter-hatespeech/hate_speech_eval_filtered_translated.json", data_translated)

    save_json("../data/slovenian-twitter-hatespeech/hate_speech_eval_filtered_translated_final.json", data_translated)
