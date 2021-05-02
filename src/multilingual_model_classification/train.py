import json
import random
import time
from collections import Counter

import numpy as np
import pandas as pd
import torch
import nltk

from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification, AdamW, BertConfig, \
    get_linear_schedule_with_warmup
from utils import read_fox_comments_dataset, flat_accuracy, format_time
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from torch.utils.data import DataLoader, SequentialSampler, RandomSampler, TensorDataset

nltk.download('punkt')


def get_torch_device():
    # Check for GPU...
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        return torch.device("cuda")

    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")


def convert_to_input(contents, tokenizer, max_length=128, pad_token=0, pad_token_segment_id=0):
    input_ids, attention_masks, token_type_ids = [], [], []

    for sentence in tqdm(contents, position=0, leave=True):
        # inputs = tokenizer.encode_plus(sentence,
        #                                add_special_tokens=True,
        #                                max_length=max_length,
        #                                padding="max_length",
        #                                return_tensors="pt",
        #                                return_attention_mask=True)
        #
        #
        # input_ids.append(inputs["input_ids"])
        # attention_masks.append(inputs["attention_mask"])
        inputs = tokenizer.encode_plus(sentence, add_special_tokens=True, max_length=max_length)

        i, t = inputs["input_ids"], inputs["token_type_ids"]
        m = [1] * len(i)

        padding_length = max_length - len(i)

        i = i + ([pad_token] * padding_length)
        m = m + ([0] * padding_length)
        t = t + ([pad_token_segment_id] * padding_length)

        input_ids.append(torch.Tensor([i]))
        attention_masks.append(torch.Tensor([m]))
        token_type_ids.append(torch.Tensor([t]))

    return torch.cat(input_ids, dim=0).to(torch.int64), torch.cat(attention_masks, dim=0).to(torch.int64)

def get_dataloaders(train_dataset, val_dataset, batch_size):
    train_dataloader = DataLoader(
        train_dataset,  # The training samples.
        sampler=RandomSampler(train_dataset),  # Select batches randomly
        batch_size=batch_size  # Trains with this batch size.
    )

    val_dataloader = DataLoader(
        val_dataset,  # The validation samples.
        sampler=SequentialSampler(val_dataset),  # Pull out batches sequentially.
        batch_size=batch_size  # Evaluate with this batch size.
    )

    return train_dataloader, val_dataloader

def train(model, optimizer, scheduler, train_dataloader, validation_dataloader):
    # Set the seed value all over the place to make this reproducible.
    seed_val = 42

    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)

    # We'll store a number of quantities such as training and validation loss,
    # validation accuracy, and timings.
    training_stats = []

    # Measure the total training time for the whole run.
    total_t0 = time.time()

    # For each epoch...
    for epoch_i in range(0, epochs):

        # ========================================
        #               Training
        # ========================================

        # Perform one full pass over the training set.

        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        print('Training...')

        # Measure how long the training epoch takes.
        t0 = time.time()

        # Reset the total loss for this epoch.
        total_train_loss = 0

        # Put the model into training mode. Don't be mislead--the call to
        # `train` just changes the *mode*, it doesn't *perform* the training.
        # `dropout` and `batchnorm` layers behave differently during training
        # vs. test (source: https://stackoverflow.com/questions/51433378/what-does-model-train-do-in-pytorch)
        model.train()

        # For each batch of training data...
        for step, batch in enumerate(train_dataloader):

            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)

                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))

            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using the
            # `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Always clear any previously calculated gradients before performing a
            # backward pass. PyTorch doesn't do this automatically because
            # accumulating the gradients is "convenient while training RNNs".
            # (source: https://stackoverflow.com/questions/48001598/why-do-we-need-to-call-zero-grad-in-pytorch)
            model.zero_grad()

            # Perform a forward pass (evaluate the model on this training batch).
            # The documentation for this `model` function is here:
            # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
            # It returns different numbers of parameters depending on what arguments
            # arge given and what flags are set. For our useage here, it returns
            # the loss (because we provided labels) and the "logits"--the model
            # outputs prior to activation.
            model_output = model(b_input_ids,
                                 token_type_ids=None,
                                 attention_mask=b_input_mask,
                                 labels=b_labels)

            loss = model_output.loss

            # Accumulate the training loss over all of the batches so that we can
            # calculate the average loss at the end. `loss` is a Tensor containing a
            # single value; the `.item()` function just returns the Python value
            # from the tensor.
            total_train_loss += loss.item()

            # Perform a backward pass to calculate the gradients.
            loss.backward()

            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)

            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()

            # Update the learning rate.
            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(training_time))

        # ========================================
        #               Validation
        # ========================================
        # After the completion of each training epoch, measure our performance on
        # our validation set.

        print("")
        print("Running Validation...")

        t0 = time.time()

        # Put the model in evaluation mode--the dropout layers behave differently
        # during evaluation.
        model.eval()

        # Tracking variables
        total_eval_accuracy = 0
        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch

        val_predictions = []
        val_labels = []
        for batch in validation_dataloader:
            # Unpack this training batch from our dataloader.
            #
            # As we unpack the batch, we'll also copy each tensor to the GPU using
            # the `to` method.
            #
            # `batch` contains three pytorch tensors:
            #   [0]: input ids
            #   [1]: attention masks
            #   [2]: labels
            b_input_ids = batch[0].to(device)
            b_input_mask = batch[1].to(device)
            b_labels = batch[2].to(device)

            # Tell pytorch not to bother with constructing the compute graph during
            # the forward pass, since this is only needed for backprop (training).
            with torch.no_grad():
                # Forward pass, calculate logit predictions.
                # token_type_ids is the same as the "segment ids", which
                # differentiates sentence 1 and 2 in 2-sentence tasks.
                # The documentation for this `model` function is here:
                # https://huggingface.co/transformers/v2.2.0/model_doc/bert.html#transformers.BertForSequenceClassification
                # Get the "logits" output by the model. The "logits" are the output
                # values prior to applying an activation function like the softmax.
                eval_outputs = model(b_input_ids,
                                       token_type_ids=None,
                                       attention_mask=b_input_mask,
                                       labels=b_labels)
                loss = eval_outputs.loss
                logits = eval_outputs.logits

            # Accumulate the validation loss.
            total_eval_loss += loss.item()

            # Move logits and labels to CPU
            logits = logits.detach().cpu().numpy()
            label_ids = b_labels.to('cpu').numpy()

            # Calculate the accuracy for this batch of test sentences, and
            # accumulate it over all batches.
            total_eval_accuracy += flat_accuracy(logits, label_ids)

            pred_flat = np.argmax(logits, axis=1).flatten()
            labels_flat = label_ids.flatten()

            val_predictions.append(pred_flat)
            val_labels.append(labels_flat)

        val_predictions = np.hstack(np.array(val_predictions))
        val_labels = np.hstack(np.array(val_labels))

        precision = precision_score(val_labels, val_predictions)
        recall = recall_score(val_labels, val_predictions)
        f1score = f1_score(val_labels, val_predictions)

        # Report the final accuracy for this validation run.
        avg_val_accuracy = total_eval_accuracy / len(validation_dataloader)
        print("  Accuracy: {0:.2f}".format(avg_val_accuracy))
        print(f"f1-score: {round(f1score, 2)}, precision: {round(precision,2)}, recall: {round(recall, 2)}")
        # Calculate the average loss over all of the batches.
        avg_val_loss = total_eval_loss / len(validation_dataloader)

        # Measure how long the validation run took.
        validation_time = format_time(time.time() - t0)

        print("  Validation Loss: {0:.2f}".format(avg_val_loss))
        print("  Validation took: {:}".format(validation_time))

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Valid. Accur.': avg_val_accuracy,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print("")
    print("Training complete!")

    print("Total training took {:} (h:mm:ss)".format(format_time(time.time() - total_t0)))

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

if __name__ == '__main__':

    batch_size = 24
    learning_rate = 5e-5
    epochs = 10
    max_length = 64
    #
    # df = read_white_supremacist_dataset()
    # comments = df["sentence"].values
    # labels = df["hateful"].values

    df = read_fox_comments_dataset()
    comments = df["text"].values
    labels = df["label"].values

    cnt = Counter(labels)
    a = 0

    x_train, x_val, y_train, y_val = train_test_split(comments, labels, test_size=0.2, random_state=42)

    # Get Pytorch device
    device = get_torch_device()

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)

    # Convert comments to inputs for BERT
    print("Tokenizing the inputs...")
    train_input_ids, train_attention_masks = convert_to_input(x_train, tokenizer, max_length=max_length)
    val_input_ids, val_attention_masks = convert_to_input(x_val, tokenizer, max_length=max_length)

    y_train = torch.tensor(y_train).to(torch.int64)
    y_val = torch.tensor(y_val).to(torch.int64)

    train_dataset = TensorDataset(train_input_ids, train_attention_masks, y_train)
    val_dataset = TensorDataset(val_input_ids, val_attention_masks, y_val)

    train_dataloader, val_dataloader = get_dataloaders(train_dataset, val_dataset, batch_size)

    # Create the model
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased",
        num_labels=2,
        output_attentions=False,
        output_hidden_states=False,
    )
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

    train(model, optimizer, scheduler, train_dataloader, val_dataloader)

    a = 0
