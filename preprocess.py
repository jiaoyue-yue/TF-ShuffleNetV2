"""WAV to 1 x 48 x 157 log-Mel features, following manuscript section 1.1.2."""
import numpy as np
import soundfile as sf
import librosa

def fix_length_center(y: np.ndarray, target_len: int):
    """
    长于 target_len：截取中间段
    短于 target_len：补零
    """
    n = len(y)
    if n == target_len:
        return y

    if n > target_len:
        start = (n - target_len) // 2
        return y[start:start + target_len]

    out = np.zeros((target_len,), dtype=y.dtype)
    out[:n] = y
    return out

def wav_to_log_mel(path):
    y, sr = sf.read(str(path), dtype="float32", always_2d=True)
    if sr != 8000:
        raise ValueError("Expected the manuscript sampling rate of 8000 Hz")
    y = fix_length_center(y[:, 0], 160000)
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_fft=2048, win_length=2048, hop_length=1024,
        window="hann", center=True, power=2.0, n_mels=48, fmin=0, fmax=4000)
    feature = np.log(mel + 1e-10).astype(np.float32)[None]
    if feature.shape != (1, 48, 157) or not np.isfinite(feature).all():
        raise ValueError("Invalid log-Mel feature")
    return feature


def normalize_record(feature):
    feature = np.asarray(feature, dtype=np.float32)
    if feature.shape != (1, 48, 157):
        raise ValueError("Expected 1 x 48 x 157")
    return (feature - feature.mean()) / (feature.std(ddof=1) + 1e-6)
