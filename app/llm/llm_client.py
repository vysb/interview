"""LLM client for processing transcriptions."""

import os
from typing import Optional

try:
    from openai import OpenAI
except ImportError:
    OpenAI = None


class LLMClient:
    """Client for LLM API (supports OpenAI and local llama servers)."""

    def __init__(
        self,
        base_url: str = "http://localhost:11434/v1",
        api_key: Optional[str] = None,
        model: str = "llama3",
        temperature: float = 0.7,
        max_tokens: int = 500,
        initial_prompt: Optional[str] = None,
    ):
        """Initialize LLM client.

        Args:
            base_url: Base URL for LLM API (default: http://localhost:11434/v1 for Ollama)
            api_key: API key (required for OpenAI, optional for local Ollama)
            model: Model name (default: llama3 for Ollama)
            temperature: Temperature for generation (default: 0.7)
            max_tokens: Maximum tokens in response (default: 500 for flexible responses)
            initial_prompt: Initial system prompt
        """
        if OpenAI is None:
            raise ImportError(
                "openai package is required. Install it with: pip install openai"
            )

        self.base_url = base_url
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        # Add compact response instruction if no custom prompt provided
        compact_instruction = (
            "Отвечай компактно и по делу по русски. "
            "Если можно ответить одним предложением - используй короткий ответ. "
            "Если нужно больше - используй столько предложений, сколько необходимо, но будь лаконичным."
        )
        
        if initial_prompt:
            self.initial_prompt = f"{initial_prompt}\n\n{compact_instruction}"
        else:
            self.initial_prompt = compact_instruction

        # Initialize OpenAI client (works with both OpenAI and local servers)
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key or os.getenv("OPENAI_API_KEY") or "not-needed",
        )

    def process_transcription(self, transcription: str) -> Optional[str]:
        """Process transcription through LLM.

        Args:
            transcription: Transcribed text

        Returns:
            LLM response or None if error
        """
        if not transcription.strip():
            return None

        try:
            messages = []
            if self.initial_prompt:
                messages.append(
                    {"role": "system", "content": self.initial_prompt}
                )
            messages.append({"role": "user", "content": transcription})

            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
            )

            return response.choices[0].message.content.strip()
        except Exception as e:
            # Log error for debugging but don't crash the app
            import sys
            print(f"[LLM Error]: {type(e).__name__}: {e}", flush=True, file=sys.stderr)
            return None

