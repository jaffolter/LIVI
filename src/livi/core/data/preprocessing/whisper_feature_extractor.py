from typing import List, Optional
import numpy as np
import torch
from transformers import WhisperFeatureExtractor
from functools import lru_cache

from livi.core.data.utils.audio_toolbox import load_audio


class WhisperExtractor:
    """
    Wrapper around HuggingFace WhisperFeatureExtractor for log-Mel feature extraction.

    Usage
    -----
    - Input: audio waveforms already chunked into ~30s segments.
    - Output: log-Mel spectrogram features (float32 tensors), ready for training.

    Notes
    -----
    - This class does *not* perform segmentation itself; you must provide
        fixed-length waveforms externally.
    - All waveforms must be at `self.sample_rate` (e.g., 16 kHz for Whisper).
    """

    def __init__(self, sample_rate: int, model_name: str = "openai/whisper-large-v3-turbo"):
        self.sample_rate = sample_rate
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

        self.processor = WhisperFeatureExtractor.from_pretrained(model_name)

    def pipeline(self, waveforms: List[np.ndarray]) -> torch.Tensor:
        """
        Extract log-Mel features from a list of audio waveforms.

        Args:
            waveforms: List of audio waveforms as numpy arrays.

        Returns:
            np.ndarray: Extracted log-Mel features.
        """
        with torch.no_grad():
            audio_features = self.processor(
                waveforms, sampling_rate=self.sample_rate, padding=True, return_tensors="pt", device="cuda"
            )
            mel = audio_features.input_features.to(self.device)

        return mel


# ------------------------------- RUNNERS -------------------------------


@lru_cache(maxsize=8)
def get_cached_feature_extractor(
    sample_rate: int = 16_000,
    model_name: str = "openai/whisper-large-v3-turbo",
) -> WhisperExtractor:
    """Cache detectors keyed by config to avoid reloading the model repeatedly."""
    return WhisperExtractor(
        sample_rate=sample_rate,
        model_name=model_name,
    )


def extract_mel(
    waveforms: List[np.ndarray],
    feature_extractor: Optional[WhisperExtractor],
    sample_rate: Optional[int] = 16_000,
    model_name: Optional[str] = "openai/whisper-large-v3-turbo",
) -> torch.Tensor:
    """
    Extract vocal components from a waveform.

    Args:
        waveforms: List of audio waveforms as numpy arrays.
        feature_extractor: Pre-initialized Whisper feature extractor.
        sample_rate: Sampling rate of Whisper (16000).
        model_name: Model name for feature extraction.

    Returns:
        torch.Tensor: Extracted log-Mel features.
    """
    feature_extractor = feature_extractor or get_cached_feature_extractor(
        sample_rate=sample_rate,
        model_name=model_name,
    )
    return feature_extractor.pipeline(waveforms=waveforms)
