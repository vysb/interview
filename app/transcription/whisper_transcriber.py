"""Whisper transcription module for real-time audio transcription."""

from typing import Optional

import numpy as np
from faster_whisper import WhisperModel


class WhisperTranscriber:
    """Transcribes audio using local Whisper model."""

    def __init__(
        self,
        model_size: str = "base",
        device: str = "cpu",
        compute_type: str = "int8",
        language: Optional[str] = None,
    ):
        """Initialize Whisper transcriber.

        Args:
            model_size: Whisper model size (tiny, base, small, medium, large)
            device: Device to use (cpu, cuda)
            compute_type: Compute type (int8, int8_float16, float16, float32)
            language: Language code (None for auto-detection)
        """
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.language = language
        self.model: Optional[WhisperModel] = None

    def load_model(self) -> None:
        """Load Whisper model."""
        if self.model is None:
            self.model = WhisperModel(
                self.model_size,
                device=self.device,
                compute_type=self.compute_type,
            )

    def transcribe(
        self,
        audio: np.ndarray,
        sample_rate: int = 16000,
        beam_size: int = 5,
        best_of: int = 5,
        temperature: float = 0.0,
        vad_filter: bool = True,
    ) -> str:
        """Transcribe audio to text.

        Args:
            audio: Audio array (float32 in [-1, 1] range or int16)
            sample_rate: Sample rate of audio
            beam_size: Beam size for decoding
            best_of: Number of candidates to consider
            temperature: Temperature for sampling
            vad_filter: Use VAD filter (we already have VAD, so can disable)

        Returns:
            Transcribed text
        """
        if self.model is None:
            self.load_model()

        if audio.size == 0:
            return ""

        # Convert to float32 if needed
        if audio.dtype == np.int16:
            audio_float = audio.astype(np.float32) / 32768.0
        else:
            audio_float = audio.astype(np.float32)

        # Transcribe
        # faster-whisper automatically handles sample_rate (assumes 16kHz)
        # If audio is not 16kHz, it should be resampled before calling transcribe
        segments, info = self.model.transcribe(
            audio_float,
            language=self.language,
            beam_size=beam_size,
            best_of=best_of,
            temperature=temperature,
            vad_filter=vad_filter,
        )

        # Combine all segments
        text_parts = []
        for segment in segments:
            text_parts.append(segment.text.strip())

        return " ".join(text_parts).strip()

    def __enter__(self):
        """Context manager entry."""
        self.load_model()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        pass

