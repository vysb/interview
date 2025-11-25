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
from app.llm.llm_client import LLMClient


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
        max_speech_duration_ms: int = 15000,
        initial_prompt: Optional[str] = None,
        llm_client: Optional[LLMClient] = None,
    ):
        """Initialize transcription application.

        Args:
            sample_rate: Audio sample rate
            model_size: Whisper model size
            device: Device for Whisper (cpu/cuda)
            language: Language code (None for auto)
            pre_speech_buffer_ms: Pre-speech buffer in ms
            post_speech_buffer_ms: Post-speech buffer in ms
            max_speech_duration_ms: Maximum speech duration before forced interruption (ms)
            initial_prompt: Initial prompt text to guide transcription
            llm_client: Optional LLM client for processing transcriptions
        """
        self.sample_rate = sample_rate
        self.language = language
        self.initial_prompt = initial_prompt
        self.llm_client = llm_client
        self.recorder = AudioRecorder(sample_rate=sample_rate)
        self.vad = VoiceActivityDetector(
            sample_rate=sample_rate,
            pre_speech_buffer_ms=pre_speech_buffer_ms,
            post_speech_buffer_ms=post_speech_buffer_ms,
            max_speech_duration_ms=max_speech_duration_ms,
        )
        self.transcriber = WhisperTranscriber(
            model_size=model_size,
            device=device,
            language=language,
            initial_prompt=initial_prompt,
        )
        self.is_running = False
        # Increase workers if LLM is enabled to handle both transcription and LLM calls
        self.executor = ThreadPoolExecutor(max_workers=3 if llm_client else 2)
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
                # Process with LLM if enabled (non-blocking)
                if self.llm_client:
                    self.executor.submit(self._process_with_llm, text)
        except Exception as e:
            print(f"\n[Error during transcription]: {e}", flush=True, file=sys.stderr)

    def _process_with_llm(self, transcription: str) -> None:
        """Process transcription with LLM (non-blocking).

        Args:
            transcription: Transcribed text
        """
        if not self.llm_client:
            return

        try:
            response = self.llm_client.process_transcription(transcription)
            if response:
                # Green color for LLM responses
                green_color = "\033[32m"
                reset_color = "\033[0m"
                print(f"{green_color}[LLM Response]: {response}{reset_color}", flush=True)
            elif response is None:
                # Response is None means there was an error (already logged in LLM client)
                pass
        except Exception as e:
            # Log unexpected errors
            print(f"[LLM Error]: Unexpected error: {e}", flush=True, file=sys.stderr)

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
        print(f"Transcription Language: {language_str}", flush=True)
        
        if self.initial_prompt:
            print(f"Transcription Initial prompt: {self.initial_prompt}", flush=True)


        if self.llm_client:
            print(f"LLM enabled: {self.llm_client.base_url}", flush=True)
            if self.llm_client.initial_prompt:
                print(f"LLM initial prompt: {self.llm_client.initial_prompt}", flush=True)

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
        "--trsc-model-size",
        type=str,
        default="small",
        choices=["tiny", "base", "small", "medium", "large"],
        help="Whisper model size (default: medium)",
    )
    parser.add_argument(
        "--trsc-device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device for Whisper (default: cpu)",
    )
    parser.add_argument(
        "--trsc-language",
        type=str,
        default=None,
        help="Language code (e.g., 'en', 'ru'). None for auto-detection",
    )
    parser.add_argument(
        "--trsc-pre-speech-buffer-ms",
        type=int,
        default=300,
        help="Pre-speech buffer in milliseconds (default: 300)",
    )
    parser.add_argument(
        "--trsc-post-speech-buffer-ms",
        type=int,
        default=500,
        help="Post-speech buffer in milliseconds to prevent word cutoffs (default: 500)",
    )
    parser.add_argument(
        "--trsc-max-speech-duration-ms",
        type=int,
        default=15000,
        help="Maximum speech duration in milliseconds before forced interruption (default: 15000 = 15 seconds)",
    )
    parser.add_argument(
        "--trsc-initial-prompt",
        type=str,
        default="Это разговор о программировании.",
        help="Initial prompt text to guide transcription. Default: \"Это разговор о программировании.\"",
    )
    parser.add_argument(
        "--llm",
        action="store_true",
        help="Enable LLM processing of transcriptions (default: disabled)",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default="http://localhost:11434/v1",
        help="LLM API base URL (default: http://localhost:11434/v1 for Ollama)",
    )
    parser.add_argument(
        "--llm-key",
        type=str,
        default=None,
        help="LLM API key (required for OpenAI, optional for local Ollama)",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default="llama3",
        help="LLM model name (default: llama3 for Ollama)",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)",
    )
    parser.add_argument(
        "--llm-max-tokens",
        type=int,
        default=500,
        help="LLM max tokens in response (default: 500 for flexible compact responses)",
    )
    parser.add_argument(
        "--llm-init-prompt",
        type=str,
        default="",
        help="LLM initial system prompt (default: empty)",
    )

    args = parser.parse_args()

    # Initialize LLM client if enabled
    llm_client = None
    if args.llm:
        try:
            llm_client = LLMClient(
                base_url=args.llm_base_url,
                api_key=args.llm_key,
                model=args.llm_model,
                temperature=args.llm_temperature,
                max_tokens=args.llm_max_tokens,
                initial_prompt=args.llm_init_prompt if args.llm_init_prompt.strip() else None,
            )
        except ImportError as e:
            print(f"Error: {e}", file=sys.stderr)
            print("Install openai package: pip install openai", file=sys.stderr)
            sys.exit(1)
        except Exception as e:
            print(f"Warning: Failed to initialize LLM client: {e}", file=sys.stderr)
            print("Continuing without LLM...", file=sys.stderr)

    app = TranscriptionApp(
        sample_rate=args.sample_rate,
        model_size=args.trsc_model_size,
        device=args.trsc_device,
        language=args.trsc_language,
        pre_speech_buffer_ms=args.trsc_pre_speech_buffer_ms,
        post_speech_buffer_ms=args.trsc_post_speech_buffer_ms,
        max_speech_duration_ms=args.trsc_max_speech_duration_ms,
        initial_prompt=args.trsc_initial_prompt,
        llm_client=llm_client,
    )

    app.run()


if __name__ == "__main__":
    main()

