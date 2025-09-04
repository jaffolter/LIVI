"""
Audio toolbox utilities.

Reusable helpers for loading, resampling, and preprocessing audio...
"""

import torch
import torchaudio
import torchaudio.transforms as T


def load_audio(
    path: str,
    target_sample_rate: int = 16_000,
    mono: bool = True,
) -> torch.Tensor:
    """
    Load an audio file, convert to mono, and resample.

    Args:
        path (str): Path to the audio file (e.g. .mp3, .wav).
        target_sample_rate (int): Desired sample rate (Hz). Default = 16k.
        mono (bool): If True, convert multi-channel audio to mono.

    Returns:
        torch.Tensor: Waveform tensor of shape (T,), resampled and optionally mono.
    """
    # Load waveform
    waveform, sr = torchaudio.load_with_torchcodec(path)  # shape (C, T)

    # Convert to mono if multiple channels
    if mono and waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0)

    # Resample if needed
    if sr != target_sample_rate:
        resampler = T.Resample(orig_freq=sr, new_freq=target_sample_rate)
        waveform = resampler(waveform)

    return waveform
