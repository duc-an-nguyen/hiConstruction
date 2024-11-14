import torch

# Variables to configure the model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')