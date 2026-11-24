import numpy as np
import torch
import torch.nn.functional as F

def extract_speaker_embedding(

        model,
        processor,
        waveform, sample_rate
):
    if waveform.ndim != 1:
        raise ValueError("Wavf must be mono")
    if sample_rate!= 16000:
        raise ValueError(f"Resample to 16000. currently {sample_rate} ")

    device = next(model.parameters()).device

    inputs = processor(waveform, sampling_rate = sample_rate, return_tensors = "pt")

    inputs  = {

        name: tensor.to(device) for name, tensor in inputs.items()

    }

    with torch.inference_mode():
        embedding = model(**inputs).embeddings[0]

    embedding = F.normalize(embedding, dim = 0)
    return embedding.cpu()

def cosine_compare_speakers(emb_a, emb_b):
    return torch.dot(emb_a, emb_b).item()