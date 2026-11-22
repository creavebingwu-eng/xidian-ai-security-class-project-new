import json
import logging
import os
from pathlib import Path
import time

from .utils import FunctionSpec, OutputType, opt_messages_to_list, backoff_create
from funcy import notnone, once, select_values
import openai
import httpx2
from rich import print
from dotenv import load_dotenv

load_dotenv(
    dotenv_path=Path(__file__).resolve().parents[3] / ".env",
    override=True,
    encoding="utf-8-sig",
)
os.environ.setdefault("HTTP_PROXY", "http://127.0.0.1:7892")
os.environ.setdefault("HTTPS_PROXY", "http://127.0.0.1:7892")

logger = logging.getLogger("ai-scientist")


OPENAI_TIMEOUT_EXCEPTIONS = (
    openai.RateLimitError,
    openai.APIConnectionError,
    openai.APITimeoutError,
    openai.InternalServerError,
)

def get_ai_client(model: str, max_retries=2) -> openai.OpenAI:
    if model.startswith("ollama/"):
        client = openai.OpenAI(
            base_url="http://localhost:11434/v1", 
            max_retries=max_retries
        )
    elif model.startswith("qwen3.8-"):
        proxy_url = os.getenv("HTTPS_PROXY") or os.getenv("HTTP_PROXY")
        http_client = None
        if proxy_url:
            api_timeout = float(os.getenv("AI_SCIENTIST_API_TIMEOUT", "300"))
            http_client = httpx2.Client(
                proxy=proxy_url,
                trust_env=False,
                timeout=api_timeout,
            )
        client = openai.OpenAI(
            api_key=os.environ["OPENAI_API_KEY"],
            base_url=os.environ["OPENAI_BASE_URL"],
            http_client=http_client,
            max_retries=max_retries,
        )
    else:
        client = openai.OpenAI(max_retries=max_retries)
    return client


def query(
    system_message: str | None,
    user_message: str | None,
    func_spec: FunctionSpec | None = None,
    **model_kwargs,
) -> tuple[OutputType, float, int, int, dict]:
    client = get_ai_client(model_kwargs.get("model"), max_retries=0)
    filtered_kwargs: dict = select_values(notnone, model_kwargs)  # type: ignore

    messages = opt_messages_to_list(system_message, user_message)

    if func_spec is not None:
        filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
        # force the model to use the function
        filtered_kwargs["tool_choice"] = func_spec.openai_tool_choice_dict

    # Qwen's reasoning mode can make long code-generation requests slow enough
    # for the proxy/API connection to be closed. Keep it configurable, but use
    # the stable non-thinking mode by default for experiment code generation.
    if filtered_kwargs.get("model", "").startswith("qwen3.8-"):
        thinking = os.getenv("QWEN_ENABLE_THINKING", "false").strip().lower()
        filtered_kwargs["extra_body"] = {
            "enable_thinking": thinking in {"1", "true", "yes", "on"}
        }

    if filtered_kwargs.get("model", "").startswith("ollama/"):
       filtered_kwargs["model"] = filtered_kwargs["model"].replace("ollama/", "")

    t0 = time.time()
    completion = backoff_create(
        client.chat.completions.create,
        OPENAI_TIMEOUT_EXCEPTIONS,
        messages=messages,
        **filtered_kwargs,
    )
    req_time = time.time() - t0

    choice = completion.choices[0]

    if func_spec is None:
        output = choice.message.content
    else:
        assert (
            choice.message.tool_calls
        ), f"function_call is empty, it is not a function call: {choice.message}"
        assert (
            choice.message.tool_calls[0].function.name == func_spec.name
        ), "Function name mismatch"
        try:
            print(f"[cyan]Raw func call response: {choice}[/cyan]")
            output = json.loads(choice.message.tool_calls[0].function.arguments)
        except json.JSONDecodeError as e:
            logger.error(
                f"Error decoding the function arguments: {choice.message.tool_calls[0].function.arguments}"
            )
            raise e

    in_tokens = completion.usage.prompt_tokens
    out_tokens = completion.usage.completion_tokens

    info = {
        "system_fingerprint": completion.system_fingerprint,
        "model": completion.model,
        "created": completion.created,
    }

    return output, req_time, in_tokens, out_tokens, info
