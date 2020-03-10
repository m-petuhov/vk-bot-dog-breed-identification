import torch
import copy


def load_weights(model, path, device):
    model.load_state_dict(torch.load(path, map_location=torch.device(device)))
    model.eval()
