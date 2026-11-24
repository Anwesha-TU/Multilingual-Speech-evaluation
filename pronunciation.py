
def transcribe_audio(model, wavf, sample_rate, lang):
    if sample_rate!= 16000:
        raise ValueError("Resample to 16000")

    result = model.transcribe(
        wavf, language = lang.strip().lower(),
        task = "transcribe",
        temperature = 0.0,
        fp16 = (model.device.type == "cuda"),
    )

    return result["text"].strip()


