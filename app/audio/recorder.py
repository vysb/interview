"""Audio recording module for real-time microphone capture."""

import queue
import threading
from typing import Callable, Optional

import numpy as np
import sounddevice as sd


class AudioRecorder:
    """Records audio from microphone in real-time."""

    def __init__(
        self,
        sample_rate: int = 16000,
        channels: int = 1,
        chunk_size: int = 1024,
        dtype: str = "float32",
    ):
        """Initialize audio recorder.

        Args:
            sample_rate: Sample rate in Hz (default: 16000)
            channels: Number of audio channels (default: 1 for mono)
            chunk_size: Size of audio chunks in frames
            dtype: Data type for audio samples
        """
        self.sample_rate = sample_rate
        self.channels = channels
        self.chunk_size = chunk_size
        self.dtype = dtype
        self.is_recording = False
        self.audio_queue: queue.Queue = queue.Queue()
        self.stream: Optional[sd.InputStream] = None
        self.callback: Optional[Callable[[np.ndarray], None]] = None

    def _audio_callback(
        self, indata: np.ndarray, frames: int, time_info: dict, status: sd.CallbackFlags
    ) -> None:
        """Callback function for audio stream."""
        if status:
            print(f"Audio callback status: {status}")
        if self.is_recording:
            audio_chunk = indata.copy()
            if self.callback:
                self.callback(audio_chunk)
            else:
                self.audio_queue.put(audio_chunk)

    def start_recording(self, callback: Optional[Callable[[np.ndarray], None]] = None) -> None:
        """Start recording audio.

        Args:
            callback: Optional callback function to process audio chunks in real-time
        """
        if self.is_recording:
            return

        self.callback = callback
        self.is_recording = True

        self.stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype=self.dtype,
            blocksize=self.chunk_size,
            callback=self._audio_callback,
        )
        self.stream.start()

    def stop_recording(self) -> None:
        """Stop recording audio."""
        if not self.is_recording:
            return

        self.is_recording = False
        if self.stream:
            self.stream.stop()
            self.stream.close()
            self.stream = None

    def get_audio_chunk(self, timeout: Optional[float] = None) -> Optional[np.ndarray]:
        """Get next audio chunk from queue.

        Args:
            timeout: Timeout in seconds (None for blocking)

        Returns:
            Audio chunk as numpy array or None if timeout
        """
        try:
            return self.audio_queue.get(timeout=timeout)
        except queue.Empty:
            return None

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.stop_recording()

