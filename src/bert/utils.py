import datetime

import numpy as np
import torch

def get_torch_device():
    # Check for GPU...
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        return torch.device("cuda")

    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")

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