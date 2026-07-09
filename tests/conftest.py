"""Pytest configuration and fixtures for claude-code-transcripts tests."""

import pytest


@pytest.fixture(autouse=True)
def isolated_watch_state_dir(tmp_path, monkeypatch):
    """Keep every test's watch-daemon state out of the real home directory."""
    d = tmp_path / "cct-state"
    monkeypatch.setenv("CLAUDE_CODE_TRANSCRIPTS_STATE_DIR", str(d))
    return d


@pytest.fixture(autouse=True)
def mock_webbrowser_open(monkeypatch):
    """Automatically mock webbrowser.open to prevent browsers opening during tests."""
    opened_urls = []

    def mock_open(url):
        opened_urls.append(url)
        return True

    monkeypatch.setattr("claude_code_transcripts.webbrowser.open", mock_open)
    return opened_urls
