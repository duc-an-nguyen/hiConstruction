import torch
from RulModel         import ModelHI
from config           import device

def load_model(model_path, model_params):
        modelo = ModelHI(**model_params)
        modelo = modelo.to(device)
        modelo.load_state_dict(torch.load(model_path, weights_only=True))
        modelo.eval()
        return modelo