import json
import re
import torch
import numpy as np
import pandas as pd
import torch
import nltk
from sklearn.metrics import precision_score, recall_score, f1_score
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup

from src.multilingual_model_classification.train import get_torch_device, convert_to_input


def filter_tweet_content(content):
    # Remove ' and "
    content = content.replace('"','').replace("'", "")
    content = " ".join([el for el in content.split(" ") if not el.startswith("@")])
    content = re.sub(r'http\S+', '', content)
    return content

def parse_slovenian_hate_speech_dataset(filename):
    with open(filename, "r") as f:
        data = json.load(f)

    tweets = []
    labels = []
    for element in data:
        content = filter_tweet_content(element["content"])
        type = [int(re.findall(r'\b\d+\b', el)[0]) for el in element["type"]]
        # FIXME: Change how we consider annotator opinions?
        hate_speech = 0 not in type
        tweets.append(content)
        labels.append(hate_speech)

    df = pd.DataFrame()
    df["text"] = tweets
    df["label"] = labels

    return df

if __name__ == '__main__':
    max_length = 64

    df_slo = parse_slovenian_hate_speech_dataset(
        "../../data/slovenian-twitter-hatespeech/hate_speech_data_evaluation.json")
    x_test = df_slo["text"].values
    y_test = df_slo["label"].values

    device = get_torch_device()
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    test_input_ids, test_attention_masks = convert_to_input(x_test, tokenizer, max_length=max_length)

    y_test = torch.tensor(y_test).to(torch.int64)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, y_test)
    test_dataloader = DataLoader(
        test_dataset,  # The validation samples.
        sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
        batch_size=32  # Evaluate with this batch size.
    )

    model = BertForSequenceClassification.from_pretrained("../../data/models/crosloen_bert/")
    # TODO: Load trained model
    # model.bert.load_state_dict(torch.load("../data/models/crosloen_bert/pytorch_model.bin"))
    model.cuda()

    predictions = []
    true_labels = []

    for batch in tqdm(test_dataloader):
        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        with torch.no_grad():
            eval_outputs = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)
        logits = eval_outputs.logits

        # Move logits and labels to CPU
        logits = np.argmax(logits.detach().cpu().numpy(), axis=1)
        label_ids = b_labels.to('cpu').numpy()

        # Store predictions and true labels
        predictions.append(logits)
        true_labels.append(label_ids)

    predictions = np.hstack(predictions)
    true_labels = np.hstack(true_labels)

    precision = precision_score(true_labels, predictions)
    recall = recall_score(true_labels, predictions)
    f1score = f1_score(true_labels, predictions)

    print(f"f1-score: {f1score}, precision: {precision}, recall: {recall}")


# Without pretraining, pure cro slo en bert: f1-score: 0.24440102209529535, precision: 0.21913746630727762, recall: 0.2762487257900102

# Slovenian dataset: tested pretrained on Gab:
# f1-score: 0.35710349388294943, precision: 0.2642740619902121, recall: 0.5504587155963303

# Slovenian dataset, tested pretrained on Reddit:
# f1-score: 0.2865440464666021, precision: 0.2728110599078341, recall: 0.30173292558613657