import threading
from typing import Any, Dict, List, Optional, Tuple

from bfcl_eval.model_handler.api_inference.openai_completion import OpenAICompletionsHandler
from openai.resources.chat.completions import Completions

from evalscope.api.model import GenerateConfig
from evalscope.models.utils.openai import move_unsupported_params_to_extra_body, openai_completion_params
from evalscope.utils.argument_utils import get_supported_params

# The BFCL handler sets these itself and parses whole (non-streamed) responses.
_HANDLER_OWNED = {'model', 'messages', 'tools', 'stream', 'stream_options'}
# Never persist credentials that may travel with a request.
_UNRECORDED_REQUEST_KEYS = {'extra_headers'}
_trace = threading.local()


def _dump(value: Any) -> Any:
    return value.model_dump(mode='json') if hasattr(value, 'model_dump') else value


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
        move_unsupported_params_to_extra_body(params, get_supported_params(Completions.create))
        return params

    @staticmethod
    def start_trace() -> None:
        """Record every request/response exchanged by this thread until :meth:`collect_trace`."""
        _trace.records = []

    @staticmethod
    def collect_trace() -> List[Dict[str, Any]]:
        records = getattr(_trace, 'records', None) or []
        _trace.records = None
        return records

    def generate_with_backoff(self, **kwargs: Any) -> Tuple[Any, float]:
        for key, value in self._request_params.items():
            if key == 'extra_body':
                kwargs['extra_body'] = {**value, **(kwargs.get('extra_body') or {})}
            else:
                # Values set by BFCL for a specific request (e.g. temperature) take precedence.
                kwargs.setdefault(key, value)
        records = getattr(_trace, 'records', None)
        request = {k: _dump(v) for k, v in kwargs.items() if k not in _UNRECORDED_REQUEST_KEYS}
        try:
            response, latency = super().generate_with_backoff(**kwargs)
        except Exception as exc:
            if records is not None:
                records.append({'request': request, 'error': f'{type(exc).__name__}: {exc}'})
            raise
        if records is not None:
            # The full response keeps server fields such as tool-call ids, reasoning_content
            # and a rendered prompt_text when the server returns one.
            records.append({'request': request, 'response': _dump(response), 'latency': latency})
        return response, latency
