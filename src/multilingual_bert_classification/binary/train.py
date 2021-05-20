
import numpy as np
import torch

from transformers import BertTokenizer, BertForSequenceClassification, AdamW, \
    get_linear_schedule_with_warmup

from src.bert.methods import convert_to_input, get_dataloaders, train
from src.bert.utils import get_torch_device
from src.utils import read_fox_comments_dataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, SequentialSampler, TensorDataset



if __name__ == '__main__':

    batch_size = 16
    learning_rate = 5e-5
    epochs = 1
    max_length = 64

    # df_gab = parse_gab_reddit_dataset("../data/gab-reddit-hate-speech/gab.csv")# (17870 false, 146001 true)
    # # df_reddit = parse_gab_reddit_dataset("../data/gab-reddit-hate-speech/reddit.csv")# (16959 false, 5251 true)
    # comments = df_gab["comment"].values
    # labels = df_gab["label"].values

    # df = read_white_supremacist_dataset()
    # comments = df["sentence"].values
    # labels = df["hateful"].values


    df = read_fox_comments_dataset()
    comments = df["text"].values
    labels = df["label"].values

    x_train, x_test, y_train, y_test = train_test_split(comments, labels, test_size=0.1, random_state=42)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.2, random_state=42)
    # Get Pytorch device
    device = get_torch_device()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Convert comments to inputs for BERT
    print("Tokenizing the inputs...")
    train_input_ids, train_attention_masks = convert_to_input(x_train, tokenizer, max_length=max_length)
    val_input_ids, val_attention_masks = convert_to_input(x_val, tokenizer, max_length=max_length)
    test_input_ids, test_attention_masks = convert_to_input(x_test, tokenizer, max_length=max_length)

    y_train = torch.tensor(y_train).to(torch.int64)
    y_val = torch.tensor(y_val).to(torch.int64)
    y_test = torch.tensor(y_test).to(torch.int64)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, y_train)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, y_val)
    test_dataset = TensorDataset(test_input_ids, test_attention_masks, y_test)

    test_dataloader = DataLoader(
        test_dataset,  # The validation samples.
        sampler=SequentialSampler(test_dataset),  # Pull out batches sequentially.
        batch_size=32  # Evaluate with this batch size.
    )
    train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, batch_size)

    # Create the model
    # model = BertForSequenceClassification.from_pretrained(
    #     "bert-base-uncased",
    #     num_labels=2,
    #     output_attentions=False,
    #     output_hidden_states=False,
    # )

    model = BertForSequenceClassification.from_pretrained("../../data/models/crosloen_bert/")
    model.cuda()

    # Create the optimizer
    optimizer = AdamW(model.parameters(),
                      lr=learning_rate,  # args.learning_rate - default is 5e-5, our notebook had 2e-5
                      eps=1e-8  # args.adam_epsilon  - default is 1e-8.
                      )

    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=len(train_dataloader) * epochs)

    train(model, optimizer, scheduler, train_dataloader, val_dataloader, epochs, device)


    predictions = []
    true_labels = []

    for batch in test_dataloader:
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


    # Gab dataset:
    # Running Validation...
    #   Accuracy: 0.91
    # f1-score: 0.9, precision: 0.9, recall: 0.9
    #   Validation Loss: 0.25
    #   Validation took: 0:00:44
    #
    # Training complete!
    # Total training took 0:26:04 (h:mm:ss)
    # f1-score: 0.8998982015609094, precision: 0.8953409858203917, recall: 0.9045020463847203

    # Reddit: f1-score: 0.8062157221206582, precision: 0.8136531365313653, recall: 0.7989130434782609

    # Slovenian dataset: tested pretrained on Gab:
    # f1-score: 0.35710349388294943, precision: 0.2642740619902121, recall: 0.5504587155963303