from __future__ import annotations

from typing import List, Tuple, Optional
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F


from livi.core.data.utils.audio_toolbox import load_audio


class VocalDetector:
    """
    Vocal activity detection and fixed-length audio chunks extraction.

    Since the model is proprietary, we only extract 30s non-overlapping chunks
    from raw audio.
    We let the user implement their own vocal detection model if needed.
    """

    def __init__(
        self,
        sample_rate: int = 16_000,
        chunk_sec: float = 30.0,
    ) -> None:
        """Initialize the VocalDetector.

        Args:
            sample_rate (int, optional): Sample rate for audio processing. Defaults to 16_000.
            chunk_sec (float, optional): Duration of each audio chunk in seconds. Defaults to 30.0.
        """
        self.sample_rate = int(sample_rate)
        self.chunk_sec = float(chunk_sec)
        self.chunk_size = int(self.sample_rate * self.chunk_sec)

    def pipeline(
        self,
        waveform: np.array,
    ) -> Tuple[
        np.array,  # chunks_audio (N, T_chunk)
    ]:
        """
        Load audio and extract non-overlapping chunks of fixed length.
        Args:
            waveform (np.array): Input waveform (1D numpy array).
        Returns:
            chunks_audio (np.array): Extracted audio chunks (N, T_chunk).
        """

        T_total = waveform.shape[1]

        chunks: List[np.ndarray] = []
        for start in range(0, T_total, self.chunk_size):
            end = min(start + self.chunk_size, T_total)
            chunk = waveform[:, start:end]

            # pad the last chunk if it's too short
            if chunk.shape[1] < self.chunk_size:
                chunk = F.pad(chunk, (0, self.chunk_size - chunk.shape[1]))

            chunks.append(chunk.squeeze(0).numpy())

        return chunks


# ------------------------------- Runners -------------------------------
@lru_cache(maxsize=8)
def get_cached_vocal_detector(
    sample_rate: int = 16_000,
    chunk_sec: float = 30.0,
) -> VocalDetector:
    """
    Cache detectors keyed by config to avoid reloading the model repeatedly.
    """
    return VocalDetector(
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
    )


def extract_vocals(
    waveform: torch.Tensor,
    vocal_detector: Optional[VocalDetector],
    sample_rate: Optional[int] = 16_000,
    chunk_sec: Optional[float] = 30.0,
) -> Tuple[
    torch.Tensor,  # chunks_audio (N, T_chunk)
]:
    """
    Extract vocal components from a waveform, on a single audio file.
    """

    vocal_detector = vocal_detector or get_cached_vocal_detector(
        sample_rate=sample_rate,
        chunk_sec=chunk_sec,
    )

    return vocal_detector.pipeline(waveform=waveform)
