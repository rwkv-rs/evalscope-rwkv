from typing import Any, Dict, Optional, Tuple

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from openai.resources.chat.completions import Completions

from evalscope.api.model import GenerateConfig
from evalscope.models.utils.openai import openai_completion_params
from evalscope.utils.argument_utils import get_supported_params

# The BFCL handler sets these itself and parses whole (non-streamed) responses.
_HANDLER_OWNED = {'model', 'messages', 'tools', 'stream', 'stream_options'}


class ConfiguredOpenAICompletionsHandler(OpenAICompletionsHandler):
    """Apply the task's EvalScope generation config to every BFCL request of this handler instance."""

    max_tokens: Optional[int] = None
    generation_config: Optional[GenerateConfig] = None

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._request_params = self._build_request_params()

    def _build_request_params(self) -> Dict[str, Any]:
        config = self.generation_config or GenerateConfig()
        if self.max_tokens is not None and config.max_tokens is None:
            config = config.model_copy(update={'max_tokens': self.max_tokens})
        params = {
            key: value
            for key, value in openai_completion_params(model=self.model_name, config=config, tools=True).items()
            if key not in _HANDLER_OWNED
        }
        # Same rule as the generic OpenAI path: non-standard fields such as top_k or
        # chat_template_kwargs travel in extra_body instead of being rejected by the SDK.
        supported = get_supported_params(Completions.create)
        extra_body = dict(params.pop('extra_body', None) or {})
        for key in list(params):
            if key not in supported:
                extra_body[key] = params.pop(key)
        if extra_body:
            params['extra_body'] = extra_body
        return params

    def generate_with_backoff(self, **kwargs: Any) -> Tuple[Any, float]:
        for key, value in self._request_params.items():
            if key == 'extra_body':
                kwargs['extra_body'] = {**value, **(kwargs.get('extra_body') or {})}
            else:
                # Values set by BFCL for a specific request (e.g. temperature) take precedence.
                kwargs.setdefault(key, value)
        return super().generate_with_backoff(**kwargs)
