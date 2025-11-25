"""Main CLI application for real-time audio transcription."""

import argparse
import signal
import sys
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
from typing import Optional

import numpy as np

from app.audio.recorder import AudioRecorder
from app.transcription.whisper_transcriber import WhisperTranscriber
from app.vad.voice_activity_detector import VoiceActivityDetector


class TranscriptionApp:
    """Main application for real-time transcription."""

    def __init__(
        self,
        sample_rate: int = 16000,
        model_size: str = "small",
        device: str = "cpu",
        language: Optional[str] = None,
        pre_speech_buffer_ms: int = 300,
        post_speech_buffer_ms: int = 500,
        initial_prompt: Optional[str] = None,
    ):
        """Initialize transcription application.

        Args:
            sample_rate: Audio sample rate
            model_size: Whisper model size
            device: Device for Whisper (cpu/cuda)
            language: Language code (None for auto)
            pre_speech_buffer_ms: Pre-speech buffer in ms
            post_speech_buffer_ms: Post-speech buffer in ms
            initial_prompt: Initial prompt text to guide transcription
        """
        self.sample_rate = sample_rate
        self.language = language
        self.initial_prompt = initial_prompt
        self.recorder = AudioRecorder(sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            pre_speech_buffer_ms=pre_speech_buffer_ms,
            post_speech_buffer_ms=post_speech_buffer_ms,
        )
        self.transcriber = WhisperTranscriber(
            model_size=model_size,
            device=device,
            language=language,
            initial_prompt=initial_prompt,
        )
        self.is_running = False
        self.executor = ThreadPoolExecutor(max_workers=2)
        self.transcription_queue: Queue[tuple[np.ndarray, str]] = Queue()

    def _on_speech_segment(self, audio_segment: np.ndarray) -> None:
        """Handle completed speech segment.

        Args:
            audio_segment: Complete audio segment
        """
        # Submit transcription to thread pool
        self.executor.submit(self._transcribe_segment, audio_segment)

    def _transcribe_segment(self, audio_segment: np.ndarray) -> None:
        """Transcribe audio segment.

        Args:
            audio_segment: Audio segment to transcribe
        """
        try:
            text = self.transcriber.transcribe(
                audio_segment,
                sample_rate=self.sample_rate,
            )
            if text:
                print(f"\n[Transcription]: {text}", flush=True)
        except Exception as e:
            print(f"\n[Error during transcription]: {e}", flush=True, file=sys.stderr)

    def _audio_callback(self, audio_chunk: np.ndarray) -> None:
        """Process audio chunk from recorder.

        Args:
            audio_chunk: Audio chunk
        """
        if self.is_running:
            self.vad.process_audio_chunk(audio_chunk)

    def start(self) -> None:
        """Start transcription."""
        if self.is_running:
            return

        # Log configuration
        language_str = self.language if self.language else "auto-detection"
        print(f"Language: {language_str}", flush=True)
        
        if self.initial_prompt:
            print(f"Initial prompt: {self.initial_prompt}", flush=True)
        else:
            print("Initial prompt: not used", flush=True)

        print("Loading Whisper model...", flush=True)
        self.transcriber.load_model()
        print("Model loaded. Starting recording...", flush=True)

        self.is_running = True
        self.vad.set_speech_callback(self._on_speech_segment)
        self.recorder.start_recording(callback=self._audio_callback)

        print("Recording started. Speak into your microphone.", flush=True)
        print("Press Ctrl+C to stop.\n", flush=True)

    def stop(self) -> None:
        """Stop transcription."""
        if not self.is_running:
            return

        self.is_running = False
        self.recorder.stop_recording()
        self.vad.flush()
        self.executor.shutdown(wait=True)
        print("\nRecording stopped.", flush=True)

    def run(self) -> None:
        """Run transcription until interrupted."""
        # Setup signal handlers
        def signal_handler(sig, frame):
            self.stop()
            sys.exit(0)

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        try:
            self.start()
            # Keep running until interrupted
            while self.is_running:
                import time

                time.sleep(0.1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop()


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Real-time audio transcription with advanced VAD"
    )
    parser.add_argument(
        "--sample-rate",
        type=int,
        default=16000,
        help="Audio sample rate (default: 16000)",
    )
    parser.add_argument(
        "--model-size",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper (default: cpu)",
    )
    parser.add_argument(
        "--language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'ru'). None for auto-detection",
    )
    parser.add_argument(
        "--pre-speech-buffer-ms",
        type=int,
        default=300,
        help="Pre-speech buffer in milliseconds (default: 300)",
    )
    parser.add_argument(
        "--post-speech-buffer-ms",
        type=int,
        default=500,
        help="Post-speech buffer in milliseconds to prevent word cutoffs (default: 500)",
    )
    parser.add_argument(
        "--initial-prompt",
        type=str,
        default=None,
        help="Initial prompt text to guide transcription. Default: programming context prompt",
    )

    args = parser.parse_args()

    # Set default initial prompt if not provided
    default_prompt = (
        "Это разговор о программировании."
    )
    # Use default if not provided, or None if empty string is explicitly provided
    if args.initial_prompt is None:
        initial_prompt = default_prompt
    elif args.initial_prompt.strip() == "":
        initial_prompt = None
    else:
        initial_prompt = args.initial_prompt

    app = TranscriptionApp(
        sample_rate=args.sample_rate,
        model_size=args.model_size,
        device=args.device,
        language=args.language,
        pre_speech_buffer_ms=args.pre_speech_buffer_ms,
        post_speech_buffer_ms=args.post_speech_buffer_ms,
        initial_prompt=initial_prompt,
    )

    app.run()


if __name__ == "__main__":
    main()

