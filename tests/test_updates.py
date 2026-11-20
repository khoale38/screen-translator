import io
import json

from screen_translator import updates


def reply(monkeypatch, body):
    def fake_urlopen(request, timeout):
        assert request.full_url == updates.LATEST
        if isinstance(body, Exception):
            raise body
        return io.BytesIO(json.dumps(body).encode())

    monkeypatch.setattr(updates.urllib.request, "urlopen", fake_urlopen)


def test_versions_compare_by_number():
    assert updates.is_newer("v0.1.10", "0.1.9")
    assert updates.is_newer("v0.2", "0.1.9")
    assert not updates.is_newer("v0.1.0", "0.1")
    assert not updates.is_newer("v0.2.0-rc1", "0.1.0")


def test_a_newer_release_is_reported_with_its_page(monkeypatch):
    reply(monkeypatch, {"tag_name": "v0.2.0", "html_url": "https://github.com/khoale38/screen-translator/releases/tag/v0.2.0"})
    release = updates.check("0.1.0")
    assert release == updates.Release("0.2.0", "https://github.com/khoale38/screen-translator/releases/tag/v0.2.0")


def test_same_version_prereleases_and_network_errors_stay_quiet(monkeypatch):
    reply(monkeypatch, {"tag_name": "v0.1.0"})
    assert updates.check("0.1.0") is None
    reply(monkeypatch, {"tag_name": "v0.3.0", "prerelease": True})
    assert updates.check("0.1.0") is None
    reply(monkeypatch, OSError("offline"))
    assert updates.check("0.1.0") is None
