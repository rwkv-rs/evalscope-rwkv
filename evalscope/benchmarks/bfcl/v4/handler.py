from typing import Any, Optional, Tuple

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler


class ConfiguredOpenAICompletionsHandler(OpenAICompletionsHandler):
    """Keep EvalScope generation limits local to this BFCL handler instance."""

    max_tokens: Optional[int] = None

    def generate_with_backoff(self, **kwargs: Any) -> Tuple[Any, float]:
        if self.max_tokens is not None:
            kwargs.setdefault('max_tokens', self.max_tokens)
        return super().generate_with_backoff(**kwargs)
