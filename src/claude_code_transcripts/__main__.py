"""Enable `python -m claude_code_transcripts` — the watch daemon re-execs
itself this way, which works identically for uv tool installs and dev venvs."""

from claude_code_transcripts import main

if __name__ == "__main__":
    main()
