"""Voice Activity Detection with buffering to prevent word cutoffs."""

import collections
from typing import Callable, Optional

import numpy as np
import torch


class VoiceActivityDetector:
    """Advanced VAD with buffering to prevent word cutoffs."""

    def __init__(
        self,
        sample_rate: int = 16000,
        pre_speech_buffer_ms: int = 300,
        post_speech_buffer_ms: int = 500,
        min_speech_duration_ms: int = 250,
        min_silence_duration_ms: int = 800,
        speech_threshold: float = 0.5,
    ):
        """Initialize VAD detector.

        Args:
            sample_rate: Audio sample rate in Hz
            pre_speech_buffer_ms: Buffer before speech starts (ms)
            post_speech_buffer_ms: Buffer after speech ends (ms) - prevents word cutoffs
            min_speech_duration_ms: Minimum speech duration to trigger (ms)
            min_silence_duration_ms: Minimum silence duration to end segment (ms)
            speech_threshold: Probability threshold for speech detection
        """
        self.sample_rate = sample_rate
        self.pre_speech_buffer_ms = pre_speech_buffer_ms
        self.post_speech_buffer_ms = post_speech_buffer_ms
        self.min_speech_duration_ms = min_speech_duration_ms
        self.min_silence_duration_ms = min_silence_duration_ms
        self.speech_threshold = speech_threshold

        # Convert ms to samples
        self.pre_speech_buffer_samples = int(
            self.pre_speech_buffer_ms * self.sample_rate / 1000
        )
        self.post_speech_buffer_samples = int(
            self.post_speech_buffer_ms * self.sample_rate / 1000
        )
        self.min_speech_samples = int(
            self.min_speech_duration_ms * self.sample_rate / 1000
        )
        self.min_silence_samples = int(
            self.min_silence_duration_ms * self.sample_rate / 1000
        )

        # Silero VAD requires specific chunk sizes
        # 512 samples for 16kHz, 256 for 8kHz
        self.vad_chunk_size = 512 if self.sample_rate == 16000 else 256
        self.vad_buffer = np.array([], dtype=np.float32)

        # Load Silero VAD model
        self.model, self.utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.eval()

        # State tracking
        self.pre_speech_buffer: collections.deque = collections.deque(
            maxlen=self.pre_speech_buffer_samples
        )
        self.current_segment: list[np.ndarray] = []
        self.post_speech_buffer: list[np.ndarray] = []
        self.silence_counter = 0
        self.speech_counter = 0
        self.is_speaking = False
        self.on_speech_segment: Optional[Callable[[np.ndarray], None]] = None

    def _is_speech(self, audio_chunk: np.ndarray) -> bool:
        """Detect if audio chunk contains speech.

        Args:
            audio_chunk: Audio chunk as numpy array (must be exactly vad_chunk_size)

        Returns:
            True if speech detected, False otherwise
        """
        if audio_chunk.size != self.vad_chunk_size:
            return False

        # Convert to float32 if needed (Silero VAD expects float32 in [-1, 1])
        if audio_chunk.dtype == np.int16:
            audio_float = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype != np.float32:
            audio_float = audio_chunk.astype(np.float32)
        else:
            audio_float = audio_chunk

        # Ensure audio is in [-1, 1] range
        if audio_float.max() > 1.0 or audio_float.min() < -1.0:
            audio_float = np.clip(audio_float, -1.0, 1.0)

        # Silero VAD expects float32 audio tensor with shape [1, vad_chunk_size]
        audio_tensor = torch.from_numpy(audio_float).unsqueeze(0)
        speech_prob = self.model(audio_tensor, self.sample_rate).item()

        return speech_prob >= self.speech_threshold

    def process_audio_chunk(self, audio_chunk: np.ndarray) -> None:
        """Process audio chunk through VAD.

        Args:
            audio_chunk: Audio chunk to process
        """
        if audio_chunk.size == 0:
            return

        # Flatten if needed
        if len(audio_chunk.shape) > 1:
            audio_chunk = audio_chunk.flatten()

        # Convert to float32 if needed
        if audio_chunk.dtype == np.int16:
            audio_chunk = audio_chunk.astype(np.float32) / 32768.0
        elif audio_chunk.dtype != np.float32:
            audio_chunk = audio_chunk.astype(np.float32)

        # Add to buffer
        self.vad_buffer = np.concatenate([self.vad_buffer, audio_chunk])

        # Process chunks of exact size required by Silero VAD
        while len(self.vad_buffer) >= self.vad_chunk_size:
            vad_chunk = self.vad_buffer[: self.vad_chunk_size]
            self.vad_buffer = self.vad_buffer[self.vad_chunk_size :]

            is_speech = self._is_speech(vad_chunk)
            self._process_vad_result(is_speech, vad_chunk)

    def _process_vad_result(self, is_speech: bool, vad_chunk: np.ndarray) -> None:
        """Process VAD result for a single chunk.

        Args:
            is_speech: Whether speech was detected
            vad_chunk: The processed audio chunk
        """
        chunk_size = len(vad_chunk)

        if is_speech:
            self.speech_counter += chunk_size
            self.silence_counter = 0

            # Add to pre-speech buffer if not speaking yet
            if not self.is_speaking:
                self.pre_speech_buffer.append(vad_chunk.copy())

                # Check if we have enough speech to start
                if self.speech_counter >= self.min_speech_samples:
                    self._start_speech_segment()
            else:
                # Add to current segment
                self.current_segment.append(vad_chunk.copy())
        else:
            self.silence_counter += chunk_size
            self.speech_counter = 0

            if self.is_speaking:
                # Add silence to post-speech buffer (limit size)
                current_buffer_size = sum(len(chunk) for chunk in self.post_speech_buffer)
                if current_buffer_size < self.post_speech_buffer_samples:
                    self.post_speech_buffer.append(vad_chunk.copy())

                # Check if we have enough silence to end
                if self.silence_counter >= self.min_silence_samples:
                    self._end_speech_segment()
            else:
                # Clear pre-speech buffer if too much silence
                if self.silence_counter >= self.min_silence_samples:
                    self.pre_speech_buffer.clear()

    def _start_speech_segment(self) -> None:
        """Start a new speech segment."""
        if self.is_speaking:
            return

        self.is_speaking = True
        self.current_segment = []
        self.post_speech_buffer = []

        # Add pre-speech buffer to segment
        if self.pre_speech_buffer:
            for chunk in self.pre_speech_buffer:
                self.current_segment.append(chunk)
            self.pre_speech_buffer.clear()

    def _end_speech_segment(self) -> None:
        """End current speech segment and trigger callback."""
        if not self.is_speaking:
            return

        if not self.current_segment:
            self.is_speaking = False
            return

        # Concatenate speech chunks
        speech_segment = np.concatenate(self.current_segment, axis=0) if self.current_segment else np.array([], dtype=np.float32)

        # Add post-speech buffer to prevent word cutoffs
        if self.post_speech_buffer:
            post_buffer = np.concatenate(self.post_speech_buffer, axis=0)
            full_segment = np.concatenate([speech_segment, post_buffer], axis=0)
        else:
            full_segment = speech_segment

        # Trigger callback if set
        if self.on_speech_segment and len(full_segment) > 0:
            self.on_speech_segment(full_segment)

        # Reset state
        self.current_segment = []
        self.post_speech_buffer = []
        self.is_speaking = False
        self.silence_counter = 0

    def set_speech_callback(self, callback: Callable[[np.ndarray], None]) -> None:
        """Set callback function for completed speech segments.

        Args:
            callback: Function to call with complete speech segment
        """
        self.on_speech_segment = callback

    def flush(self) -> None:
        """Flush any pending speech segment and process remaining buffer."""
        # Process any remaining audio in vad_buffer (pad with zeros if needed)
        if len(self.vad_buffer) > 0:
            # Pad to vad_chunk_size if needed
            if len(self.vad_buffer) < self.vad_chunk_size:
                padding = np.zeros(self.vad_chunk_size - len(self.vad_buffer), dtype=np.float32)
                self.vad_buffer = np.concatenate([self.vad_buffer, padding])
            
            if len(self.vad_buffer) >= self.vad_chunk_size:
                vad_chunk = self.vad_buffer[: self.vad_chunk_size]
                is_speech = self._is_speech(vad_chunk)
                self._process_vad_result(is_speech, vad_chunk)
                self.vad_buffer = self.vad_buffer[self.vad_chunk_size :]

        if self.is_speaking and self.current_segment:
            # Force end current segment
            full_segment = np.concatenate(self.current_segment, axis=0)
            if self.on_speech_segment and len(full_segment) > 0:
                self.on_speech_segment(full_segment)
            self.current_segment = []
            self.post_speech_buffer = []
            self.is_speaking = False

