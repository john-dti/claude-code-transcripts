# claude-code-transcripts

[![PyPI](https://img.shields.io/pypi/v/claude-code-transcripts.svg)](https://pypi.org/project/claude-code-transcripts/)
[![Changelog](https://img.shields.io/github/v/release/simonw/claude-code-transcripts?include_prereleases&label=changelog)](https://github.com/simonw/claude-code-transcripts/releases)
[![Tests](https://github.com/simonw/claude-code-transcripts/workflows/Test/badge.svg)](https://github.com/simonw/claude-code-transcripts/actions?query=workflow%3ATest)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](https://github.com/simonw/claude-code-transcripts/blob/main/LICENSE)

Convert Claude Code session files (JSON or JSONL) to clean, mobile-friendly HTML pages with pagination.

[Example transcript](https://static.simonwillison.net/static/2025/claude-code-microjs/index.html) produced using this tool.

Read [A new way to extract detailed transcripts from Claude Code](https://simonwillison.net/2025/Dec/25/claude-code-transcripts/) for background on this project.

> [!WARNING]
>
> The `web` commands for both listing Claude Code for web sessions and converting those to a transcript are both broken right now due to changes to the unofficial and undocumented APIs that these commands were using. See [issue #77](https://github.com/simonw/claude-code-transcripts/issues/77) for details.

## Installation

Install this tool using `uv`:
```bash
uv tool install claude-code-transcripts
```
Or run it without installing:
```bash
uvx claude-code-transcripts --help
```

## Usage

This tool converts Claude Code session files into browseable multi-page HTML transcripts.

There are five commands available:

- `local` (default) - select from local Claude Code sessions stored in `~/.claude/projects`
- `watch` - live-tail an active session into the browser as Claude writes it
- `web` - select from web sessions via the Claude API
- `json` - convert a specific JSON or JSONL session file
- `all` - convert all local sessions to a browsable HTML archive

The quickest way to view a recent local session:

```bash
claude-code-transcripts
```

This shows an interactive picker to select a session, generates HTML, and opens it in your default browser.

### Output options

The `local`, `web`, `json`, and `all` commands write HTML files and support these options (the `watch` command is a live server instead — see [Live tailing](#live-tailing)):

- `-o, --output DIRECTORY` - output directory (default: writes to temp dir and opens browser)
- `-a, --output-auto` - auto-name output subdirectory based on session ID or filename
- `--repo OWNER/NAME` - GitHub repo for commit links (auto-detected if not specified). For `web` command, also filters the session list.
- `--open` - open the generated `index.html` in your default browser (default if no `-o` specified)
- `--gist` - upload the generated HTML files to a GitHub Gist and output a preview URL
- `--json` - include the original session file in the output directory

The generated output includes:
- `index.html` - an index page with a timeline of prompts and commits
- `page-001.html`, `page-002.html`, etc. - paginated transcript pages

Pages are titled with the session's name — Claude Code's auto-generated title when available (the same name `claude --resume` shows), otherwise the first real prompt — so browser tabs stay identifiable with several transcripts open.

### Session info card

Every generated page (and the live `watch` view) carries a floating session info card, collapsed to a small pill in the bottom-right corner showing prompt count and current context size. Click it to expand the full card:

- session name (Claude Code's auto-title)
- counts: prompts, messages, tool calls, commits
- usage: current context size and cumulative output tokens, read from the session's recorded `usage` data (hidden when the source has none, e.g. web JSON exports)
- recap: Claude Code's latest "※ recap" away-summary, falling back to the last assistant reply
- prompt navigation: every prompt links straight to its message, across pages
- artifact links: each prompt row expands (▸) to deep links into that turn's notable moments — ★ insights, 💭 substantial thinking, 📋 plans, ✓ the final reply — each targeting the exact content block. The same links appear under each prompt on the index timeline.
- chapter dividers: when Claude Code's auto-title changes mid-session, a `── new title ──` divider marks the spot in the prompt list (and the index timeline); sessions with one stable title look unchanged
- jump to top / jump to the latest message

The expanded/collapsed choice persists per browser (localStorage).

### Local sessions

Local Claude Code sessions are stored as JSONL files in `~/.claude/projects`. Run with no arguments to select from recent sessions:

```bash
claude-code-transcripts
# or explicitly:
claude-code-transcripts local
```

Use `--limit` to control how many sessions are shown (default: 10):

```bash
claude-code-transcripts local --limit 20
```

### Live tailing

The `watch` command streams an **in-progress** session into your browser in real time — a `tail -f` with the same rich rendering as the static pages. It starts a small local server, opens your browser, and pushes each new message (via [Server-Sent Events](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events)) as Claude appends it to the session file. A running stats bar (prompts, messages, tool calls, commits) and a clickable list of prompts update live as the session grows.

```bash
# Tail the most-recently-modified session (the one you're actively working in)
claude-code-transcripts watch
```

By default it picks the newest session under `~/.claude/projects`. Other ways to choose:

```bash
# Choose from a list instead of auto-selecting the newest
claude-code-transcripts watch --pick

# Tail a specific session file
claude-code-transcripts watch --session ~/.claude/projects/my-project/abc123.jsonl
```

Options:

- `--session PATH` - tail a specific session file instead of the newest
- `--pick` - choose the session from a list instead of auto-selecting the newest. The picker shows the same columns as `local` (date, size, git branch, project, slash command, summary).
- `--limit N` - maximum sessions to show with `--pick` (default: 10)
- `-s, --source DIRECTORY` - projects folder to search (default: `~/.claude/projects`)
- `--port N` - port to serve on (default: an OS-assigned free port)
- `--repo OWNER/NAME` - GitHub repo for commit links (auto-detected if not specified)
- `--open` / `--no-open` - open the live view in your browser (default: open)
- `--poll-interval SECONDS` - how often to check the file for new lines (default: 0.3)

Press `Ctrl-C` to stop the server. The page reconnects automatically and re-syncs if the session file is rewritten (for example when Claude compacts it).

The tab title follows the session's auto-generated name (the same name `claude --resume` shows) and renames itself live as the session evolves, so multiple watch tabs stay identifiable.

The [session info card](#session-info-card) is live too: counts and context size update as Claude works, the recap follows Claude Code's away-summaries (with the latest assistant reply standing in until one exists), every prompt is jumpable from the card, and "latest" drops you back to the streaming tail.

### Web sessions

Import sessions directly from the Claude API:

```bash
# Interactive session picker
claude-code-transcripts web

# Import a specific session by ID
claude-code-transcripts web SESSION_ID

# Import and publish to gist
claude-code-transcripts web SESSION_ID --gist
```

The session picker displays sessions grouped by their associated GitHub repository:

```
simonw/datasette              2025-01-15T10:30:00  Fix the bug in query parser
simonw/llm                    2025-01-14T09:00:00  Add streaming support
(no repo)                     2025-01-13T14:22:00  General coding session
```

Use `--repo` to filter the session list to a specific repository:

```bash
claude-code-transcripts web --repo simonw/datasette
```

On macOS, API credentials are automatically retrieved from your keychain (requires being logged into Claude Code). On other platforms, provide `--token` and `--org-uuid` manually.

### Publishing to GitHub Gist

Use the `--gist` option to automatically upload your transcript to a GitHub Gist and get a shareable preview URL:

```bash
claude-code-transcripts --gist
claude-code-transcripts web --gist
claude-code-transcripts json session.json --gist
```

This will output something like:
```
Gist: https://gist.github.com/username/abc123def456
Preview: https://gisthost.github.io/?abc123def456/index.html
Files: /var/folders/.../session-id
```

The preview URL uses [gisthost.github.io](https://gisthost.github.io/) to render your HTML gist. The tool automatically injects JavaScript to fix relative links when served through gisthost.

Combine with `-o` to keep a local copy:

```bash
claude-code-transcripts json session.json -o ./my-transcript --gist
```

**Requirements:** The `--gist` option requires the [GitHub CLI](https://cli.github.com/) (`gh`) to be installed and authenticated (`gh auth login`).

### Auto-naming output directories

Use `-a/--output-auto` to automatically create a subdirectory named after the session:

```bash
# Creates ./session_ABC123/ subdirectory
claude-code-transcripts web SESSION_ABC123 -a

# Creates ./transcripts/session_ABC123/ subdirectory
claude-code-transcripts web SESSION_ABC123 -o ./transcripts -a
```

### Including the source file

Use the `--json` option to include the original session file in the output directory:

```bash
claude-code-transcripts json session.json -o ./my-transcript --json
```

This will output:
```
JSON: ./my-transcript/session_ABC.json (245.3 KB)
```

This is useful for archiving the source data alongside the HTML output.

### Converting from JSON/JSONL files

Convert a specific session file directly:

```bash
claude-code-transcripts json session.json -o output-directory/
claude-code-transcripts json session.jsonl --open
```
This works with both JSONL files in the `~/.claude/projects/` folder and JSON session files extracted from Claude Code for web.

The `json` command can take a URL to a JSON or JSONL file as an alternative to a path on disk.

### Converting all sessions

Convert all your local Claude Code sessions to a browsable HTML archive:

```bash
claude-code-transcripts all
```

This creates a directory structure with:
- A master index listing all projects
- Per-project pages listing sessions
- Individual session transcripts

Options:

- `-s, --source DIRECTORY` - source directory (default: `~/.claude/projects`)
- `-o, --output DIRECTORY` - output directory (default: `./claude-archive`)
- `--include-agents` - include agent session files (excluded by default)
- `--dry-run` - show what would be converted without creating files
- `--open` - open the generated archive in your default browser
- `-q, --quiet` - suppress all output except errors

Examples:

```bash
# Preview what would be converted
claude-code-transcripts all --dry-run

# Convert all sessions and open in browser
claude-code-transcripts all --open

# Convert to a specific directory
claude-code-transcripts all -o ./my-archive

# Include agent sessions
claude-code-transcripts all --include-agents
```

## Development

To contribute to this tool, first checkout the code. You can run the tests using `uv run`:
```bash
cd claude-code-transcripts
uv run pytest
```
And run your local development copy of the tool like this:
```bash
uv run claude-code-transcripts --help
```
