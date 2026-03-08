from typing import Dict, Any, List, Union, AsyncGenerator
import os
import openai
from core.component.llm.llm_adapter.completion import (
    ChatCompletionRequest,
    ChatCompletionResponse,
)
from core.component.llm.llm_adapter.llm_backend_adapter import LLMBackendAdapter
from core.constants.errors import ErrorMessage


class OpenAIAdapter(LLMBackendAdapter):
    """OpenAI API adapter (implemented based on the official openai package)"""

    def __init__(self, config: Dict[str, Any]):
        # Save configuration
        self.config = config
        self.api_key = config.get("api_key") or os.getenv("OPENAI_API_KEY")
        self.base_url = config.get("base_url") or os.getenv("OPENAI_BASE_URL")
        self.timeout = config.get("timeout", 600)

        if not self.api_key:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        # Instantiate openai async client
        self.client = openai.AsyncOpenAI(
            api_key=self.api_key, base_url=self.base_url, timeout=self.timeout
        )

    @staticmethod
    def _should_use_max_completion_tokens(model: str | None) -> bool:
        """GPT-5 series uses max_completion_tokens instead of max_tokens."""
        if not model:
            return False
        normalized_model = model.lower().split("/")[-1]
        return normalized_model.startswith("gpt-5")

    @staticmethod
    def _normalize_temperature_for_model(
        model: str | None, temperature: float | None
    ) -> float | None:
        """
        GPT-5 series currently only supports default temperature behavior.
        To avoid 400 errors, omit temperature for gpt-5* models.
        """
        if temperature is None:
            return None
        if OpenAIAdapter._should_use_max_completion_tokens(model):
            return None
        return temperature

    async def chat_completion(
        self, request: ChatCompletionRequest
    ) -> Union[ChatCompletionResponse, AsyncGenerator[str, None]]:
        """
        Perform chat completion, supporting both streaming and non-streaming modes.
        """
        if not request.model:
            raise ValueError(ErrorMessage.INVALID_PARAMETER.value)

        params = request.to_dict()
        # The request `to_dict` method already filters for None values, but we can be explicit here for clarity
        # for what the openai client expects.
        client_params = {
            "model": params.get("model"),
            "messages": params.get("messages"),
            "temperature": self._normalize_temperature_for_model(
                params.get("model"), params.get("temperature")
            ),
            "top_p": params.get("top_p"),
            "frequency_penalty": params.get("frequency_penalty"),
            "presence_penalty": params.get("presence_penalty"),
            "stream": params.get("stream", False),
        }
        if self._should_use_max_completion_tokens(params.get("model")):
            client_params["max_completion_tokens"] = params.get("max_tokens")
        else:
            client_params["max_tokens"] = params.get("max_tokens")
        # Remove None values to avoid openai errors
        final_params = {k: v for k, v in client_params.items() if v is not None}

        try:
            if final_params.get("stream"):
                # Streaming response, return async generator
                async def stream_gen():
                    response_stream = await self.client.chat.completions.create(
                        **final_params
                    )
                    async for chunk in response_stream:
                        content = getattr(chunk.choices[0].delta, "content", None)
                        if content:
                            yield content

                return stream_gen()
            else:
                # Non-streaming response
                response = await self.client.chat.completions.create(**final_params)
                return ChatCompletionResponse.from_dict(response.model_dump())
        except Exception as e:
            raise RuntimeError(f"OpenAI chat completion request failed: {e}")

    def get_available_models(self) -> List[str]:
        """Get available model list (can be extended to call openai model list API)"""
        return self.config.get("models", [])
