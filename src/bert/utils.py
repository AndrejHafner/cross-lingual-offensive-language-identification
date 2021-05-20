import torch

def get_torch_device():
    # Check for GPU...
    if torch.cuda.is_available():
        print('GPU:', torch.cuda.get_device_name(0))
        return torch.device("cuda")

    else:
        print('No GPU available, using the CPU instead.')
        return torch.device("cpu")
