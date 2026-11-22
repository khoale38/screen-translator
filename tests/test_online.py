from types import SimpleNamespace

import anthropic
import httpx2
import pytest

from screen_translator import online
from screen_translator.address import Address
from screen_translator.config import Settings, SettingsStore
from screen_translator.context import Turn
from screen_translator.pipeline import Pipeline, default_judge, default_translator
from screen_translator.translator import build_messages


class FakeStream:
    def __init__(self, pieces, stop_reason="end_turn", error=None):
        self.pieces, self.stop_reason, self.error = pieces, stop_reason, error

    def __enter__(self):
        if self.error:
            raise self.error
        return self

    def __exit__(self, *_):
        return False

    @property
    def text_stream(self):
        return iter(self.pieces)

    def get_final_message(self):
        return SimpleNamespace(stop_reason=self.stop_reason)


class FakeClient:
    def __init__(self, *streams):
        self.streams = list(streams)
        self.requests = []
        self.beta = SimpleNamespace(messages=SimpleNamespace(stream=self.stream))

    def stream(self, **request):
        self.requests.append(request)
        return self.streams.pop(0)


def test_a_line_goes_to_claude_with_its_history_and_who_is_talking_to_whom():
    client = FakeClient(FakeStream(["Mày nợ ", "tao năm mươi đô."]))
    translator = online.ClaudeTranslator("", client=client)
    address = Address("Lamar", "Franklin", "tao", "mày")
    messages = build_messages("Lamar: You owe me fifty bucks.", [Turn("Franklin: Yo homie", "Franklin: Ê mày", 0.0)],
                              "en", "vi", {"Los Santos": "Los Santos"}, "a crime game", address=address)
    shown = []

    assert translator.translate(messages, shown.append) == "Mày nợ tao năm mươi đô."

    request = client.requests[0]
    assert request["model"] == "claude-opus-5-5"
    assert request["fallbacks"] == "default" and request["betas"] == [online.FALLBACK_BETA]
    assert request["system"].startswith("You work as a professional interpreter") and "Los Santos" in request["system"]
    assert [m["role"] for m in request["messages"]] == ["user", "assistant", "user"]
    assert all(set(m) == {"role", "content"} for m in request["messages"])
    assert 'Lamar says "tao"' in request["messages"][-1]["content"]
    assert shown == ["Mày nợ", "Mày nợ tao năm mươi đô."]


def test_a_refused_line_is_left_blank():
    translator = online.ClaudeTranslator("", client=FakeClient(FakeStream(["Sorry"], stop_reason="refusal")))
    assert translator.translate([{"role": "user", "content": "x"}]) == ""


def test_the_judge_asks_without_a_system_prompt():
    client = FakeClient(FakeStream(['{"relation": "friends"}']))
    assert online.ClaudeTranslator("", client=client).answer("Who?") == '{"relation": "friends"}'
    assert "system" not in client.requests[0]


def _failure(status):
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx2.Response(status, request=request)
    kind = {401: anthropic.AuthenticationError, 529: anthropic.InternalServerError}[status]
    return kind("nope", response=response, body=None)


def test_a_wrong_key_stops_but_a_busy_server_only_skips_the_line():
    with pytest.raises(online.OnlineError, match="didn't accept the API key") as stopped:
        online.ClaudeTranslator("", client=FakeClient(FakeStream([], error=_failure(401)))).translate(
            [{"role": "user", "content": "x"}])
    assert not isinstance(stopped.value, online.OnlineHiccup)
    with pytest.raises(online.OnlineHiccup):
        online.ClaudeTranslator("", client=FakeClient(FakeStream([], error=_failure(529)))).translate(
            [{"role": "user", "content": "x"}])


def test_choosing_claude_uses_it_for_lines_and_for_judging(tmp_path):
    settings = Settings(model=online.KEY, online_key="sk-ant-test")
    assert isinstance(default_translator(settings), online.ClaudeTranslator)
    assert isinstance(default_judge(settings), online.ClaudeTranslator)


def test_starting_claude_without_a_key_says_to_paste_one(tmp_path, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    store = SettingsStore(tmp_path / "settings.json")
    store.update({"model": online.KEY})
    pipeline = Pipeline(store, lambda _event: None)
    with pytest.raises(online.OnlineError, match="Paste your Anthropic API key"):
        pipeline.start()
