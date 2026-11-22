"""Translation by Claude over the internet, for interpreter-quality wording; needs the user's Anthropic API key."""

from __future__ import annotations

import os
from typing import Callable

from .translator import clean_output

KEY = "claude"
LABEL = "Claude · online, interpreter quality (needs an API key)"
MODEL = "claude-opus-5-5"
FALLBACK_BETA = "server-side-fallback-2026-07-01"

INTERPRETER = (
    "You work as a professional interpreter and subtitler for games, films, series, anime, streams and videos. "
    "Translate like a top dubbing script writer: natural spoken {target} that matches each character's age, "
    "personality, relationship and mood, never stiff or literal.\n\n"
)

MISSING_KEY = "Paste your Anthropic API key to use Claude, or pick a model that runs on this PC."


class OnlineError(RuntimeError):
    pass


class OnlineHiccup(OnlineError):
    """A failure that passes by itself (no connection, rate limit, overloaded); the line is skipped, not the run."""


def api_key(settings) -> str:
    return (getattr(settings, "online_key", "") or os.environ.get("ANTHROPIC_API_KEY", "")).strip()


def is_online(settings) -> bool:
    return getattr(settings, "model", "") == KEY and not getattr(settings, "custom_model_path", "")


def _request(messages: list[dict]) -> tuple[str, list[dict]]:
    """Our chat messages as the API takes them: the system prompt apart, and only role and content per message."""
    system = messages[0]["content"] if messages and messages[0]["role"] == "system" else ""
    target = messages[0].get("target", "the target language") if system else "the target language"
    rest = messages[1:] if system else messages
    return INTERPRETER.format(target=target) + system, [{"role": m["role"], "content": m["content"]} for m in rest]


class ClaudeTranslator:
    device = "Online (Claude)"

    def __init__(self, key: str, model: str = MODEL, client=None) -> None:
        if client is None:
            if not key:
                raise OnlineError(MISSING_KEY)
            import anthropic

            client = anthropic.Anthropic(api_key=key, timeout=30.0, max_retries=2)
        self._client = client
        self.model = model

    def _stream(self, system: str, messages: list[dict], max_tokens: int, effort: str):
        extra = {"system": system} if system else {}
        return self._client.beta.messages.stream(
            model=self.model,
            max_tokens=max_tokens,
            messages=messages,
            **extra,
            output_config={"effort": effort},
            betas=[FALLBACK_BETA],
            fallbacks="default",
        )

    def _run(self, system: str, messages: list[dict], max_tokens: int, effort: str,
             on_partial: Callable[[str], None] | None = None) -> str:
        import anthropic

        raw, shown = "", ""
        try:
            with self._stream(system, messages, max_tokens, effort) as stream:
                for text in stream.text_stream:
                    raw += text
                    visible = clean_output(raw)
                    if on_partial and visible != shown:
                        shown = visible
                        on_partial(visible)
                final = stream.get_final_message()
        except anthropic.AuthenticationError as exc:
            raise OnlineError("Claude didn't accept the API key. Check it at console.anthropic.com.") from exc
        except anthropic.PermissionDeniedError as exc:
            raise OnlineError(f"Claude refused the API key: {exc.message}") from exc
        except anthropic.RateLimitError as exc:
            raise OnlineHiccup("Claude's rate limit was reached; lines will resume shortly.") from exc
        except anthropic.APIConnectionError as exc:
            raise OnlineHiccup("Can't reach Claude. Check the internet connection.") from exc
        except anthropic.APIStatusError as exc:
            error = OnlineHiccup if exc.status_code >= 500 else OnlineError
            raise error(f"Claude returned an error ({exc.status_code}): {exc.message}") from exc
        if final.stop_reason == "refusal":
            return ""
        return clean_output(raw)

    def translate(self, messages: list[dict], on_partial: Callable[[str], None] | None = None) -> str:
        system, turns = _request(messages)
        budget = min(4096, 1024 + 4 * len(messages[-1].get("item", messages[-1]["content"])))
        return self._run(system, turns, budget, "low", on_partial)

    def answer(self, question: str, max_tokens: int = 120) -> str:
        return self._run("", [{"role": "user", "content": question}], max(max_tokens, 1024), "low")

    def close(self) -> None:
        close = getattr(self._client, "close", None)
        if close:
            close()
