import numpy as np
import torch

def score_naturalness(model, waveform, sample_rate):

    device = next(model.parameters()).device

    audio_tensor = torch.from_numpy(waveform).float()
    audio_tensor = audio_tensor.unsqueeze(0).to(device)

    with torch.inference_mode():
        prediction = model(audio_tensor, sample_rate)

    return prediction.item()