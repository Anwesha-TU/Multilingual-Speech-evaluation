from pathlib import Path
import soundfile as sf
import librosa
from schema import Keys

def load_audio(path):
    path = Path(path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    waveform, sample_rate = sf.read(
        path,
        dtype="float32",
    )

    if waveform.shape[0] == 0:
        raise ValueError(f"Audo file is emprt: {path}")
    

    return waveform, sample_rate

def to_mono(wavf):

    if wavf.ndim == 1:
        return wavf

    if wavf.ndim == 2:
        return wavf.mean(axis = 1)

def resample_audio(wavf, sample_rate, target_rate = 16000):
    if sample_rate == target_rate:
        return wavf, sample_rate
    resampled = librosa.resample(wavf, orig_sr=sample_rate, target_sr=target_rate, res_type="soxr_hq")
    return resampled, target_rate

def prepare_audio(path):
    waveform, sample_rate = load_audio(path)
    waveform = to_mono(waveform)

    waveform, sample_rate = resample_audio(
        waveform,
        sample_rate,
        target_rate=Keys.SAMPLE_RATE

    )

    return waveform, sample_rate