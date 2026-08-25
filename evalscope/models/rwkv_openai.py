import uuid
from typing import Any, List, Optional

from openai.types.chat import ChatCompletion

from evalscope.api.model import ChatCompletionChoice, GenerateConfig
from evalscope.api.tool import ToolInfo, parse_tool_call
from evalscope.models.openai_compatible import OpenAICompatibleAPI
from evalscope.models.utils.rwkv import parse_rwkv_text_tool_calls


class RWKVOpenAIAPI(OpenAICompatibleAPI):
    """OpenAI-compatible RWKV API with text tool-call normalization.

    Native OpenAI ``message.tool_calls`` always take precedence. When an RWKV
    endpoint instead emits a BFCL-style JSON call in ``message.content``, this
    backend converts only the transport shape and leaves schema validation and
    benchmark scoring to EvalScope.
    """

    def __init__(
        self,
        model_name: str,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        config: GenerateConfig = GenerateConfig(),
        text_tool_call_compat: bool = True,
        **model_args: Any,
    ) -> None:
        if not isinstance(text_tool_call_compat, bool):
            raise TypeError('text_tool_call_compat must be a boolean')
        self.text_tool_call_compat = text_tool_call_compat
        super().__init__(
            model_name=model_name,
            base_url=base_url,
            api_key=api_key,
            config=config,
            **model_args,
        )

    def chat_choices_from_completion(
        self,
        completion: ChatCompletion,
        tools: List[ToolInfo],
    ) -> List[ChatCompletionChoice]:
        """Normalize opt-in RWKV text calls after standard OpenAI parsing."""

        choices = super().chat_choices_from_completion(completion, tools)
        if not self.text_tool_call_compat or not tools:
            return choices

        raw_choices = sorted(completion.choices or [], key=lambda choice: choice.index)
        for raw_choice, choice in zip(raw_choices, choices):
            message = choice.message
            if message.tool_calls:
                continue
            content = raw_choice.message.content
            if not isinstance(content, str) or not content.strip():
                continue
            try:
                candidates = parse_rwkv_text_tool_calls(content, tools)
            except ValueError as error:
                message.metadata = {
                    **(message.metadata or {}),
                    'rwkv_text_tool_call_compat': {
                        'status': 'unchanged',
                        'error': str(error),
                        'raw_content': content,
                    },
                }
                continue

            tool_calls = []
            for candidate in candidates:
                tool_call = parse_tool_call(
                    id=f'call_rwkv_{uuid.uuid4().hex}',
                    function=candidate.name,
                    arguments=candidate.arguments,
                    tools=tools,
                )
                tool_call.internal = {
                    'provider': 'rwkv',
                    'transport': 'text_tool_call',
                    'raw_arguments': candidate.arguments,
                }
                tool_calls.append(tool_call)

            message.content = ''
            message.tool_calls = tool_calls
            message.metadata = {
                **(message.metadata or {}),
                'rwkv_text_tool_call_compat': {
                    'status': 'converted',
                    'raw_content': content,
                    'tool_call_count': len(tool_calls),
                },
            }
            choice.stop_reason = 'tool_calls'
        return choices


__all__ = ['RWKVOpenAIAPI']
