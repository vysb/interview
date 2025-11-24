"""Audio processing utilities."""

import numpy as np


def normalize_audio(audio: np.ndarray) -> np.ndarray:
    """Normalize audio to [-1, 1] range.

    Args:
        audio: Audio array

    Returns:
        Normalized audio array
    """
    if audio.size == 0:
        return audio

    max_val = np.max(np.abs(audio))
    if max_val > 0:
        return audio / max_val
    return audio


def convert_to_int16(audio: np.ndarray) -> np.ndarray:
    """Convert float32 audio to int16.

    Args:
        audio: Float32 audio array in [-1, 1] range

    Returns:
        Int16 audio array
    """
    audio_normalized = normalize_audio(audio)
    return (audio_normalized * 32767).astype(np.int16)


def concatenate_audio_chunks(chunks: list[np.ndarray]) -> np.ndarray:
    """Concatenate multiple audio chunks into single array.

    Args:
        chunks: List of audio chunks

    Returns:
        Concatenated audio array
    """
    if not chunks:
        return np.array([], dtype=np.float32)

    return np.concatenate(chunks, axis=0)

