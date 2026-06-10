"""Convert Claude Code session JSON to a clean mobile-friendly HTML page with pagination."""

import json
import html
import os
import platform
import re
import shutil
import subprocess
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import click
from click_default_group import DefaultGroup
import httpx
from jinja2 import Environment, PackageLoader
from markdown_it import MarkdownIt
import questionary

# Set up Jinja2 environment
_jinja_env = Environment(
    loader=PackageLoader("claude_code_transcripts", "templates"),
    autoescape=True,
)

# Load macros template and expose macros
_macros_template = _jinja_env.get_template("macros.html")
_macros = _macros_template.module


def get_template(name):
    """Get a Jinja2 template by name."""
    return _jinja_env.get_template(name)


# Regex to match git commit output: [branch hash] message
COMMIT_PATTERN = re.compile(r"\[[\w\-/]+ ([a-f0-9]{7,})\] (.+?)(?:\n|$)")

# Regex to detect GitHub repo from git push output (e.g., github.com/owner/repo/pull/new/branch)
GITHUB_REPO_PATTERN = re.compile(
    r"github\.com/([a-zA-Z0-9_-]+/[a-zA-Z0-9_-]+)/pull/new/"
)

PROMPTS_PER_PAGE = 5
LONG_TEXT_THRESHOLD = (
    300  # Characters - text blocks longer than this are shown in index
)


def extract_text_from_content(content):
    """Extract plain text from message content.

    Handles both string content (older format) and array content (newer format).

    Args:
        content: Either a string or a list of content blocks like
                 [{"type": "text", "text": "..."}, {"type": "image", ...}]

    Returns:
        The extracted text as a string, or empty string if no text found.
    """
    if isinstance(content, str):
        return content.strip()
    elif isinstance(content, list):
        # Extract text from content blocks of type "text"
        texts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                text = block.get("text", "")
                if text:
                    texts.append(text)
        return " ".join(texts).strip()
    return ""


_COMMAND_NAME_RE = re.compile(r"<command-name>([^<]*)</command-name>")
_COMMAND_ARGS_RE = re.compile(r"<command-args>(.*?)</command-args>", re.DOTALL)


# Slash commands that toggle UI/config or manage the session shell rather than
# describing the work. When one is the first user message we keep scanning for
# the real task command (e.g. /plan), but fall back to it if it is the ONLY
# content — so a /clear-only session still gets a title instead of being dropped
# as "(no summary)". Tune freely; task/skill commands (/plan, /security-review,
# /deep-research, /morningly, /init, /review) are deliberately absent because
# their args carry the session's intent.
_CONTROL_COMMANDS = frozenset(
    {
        "/effort",
        "/model",
        "/clear",
        "/config",
        "/cost",
        "/status",
        "/compact",
        "/output-style",
        "/fast",
        "/resume",
        "/doctor",
        "/login",
        "/logout",
        "/help",
        "/permissions",
        "/memory",
        "/hooks",
        "/mcp",
        "/agents",
        "/ide",
        "/vim",
        "/terminal-setup",
        "/bug",
    }
)


def _parse_command(text):
    """Return (command_name, args_body) for a slash-command wrapper, else (None, "").

    Claude Code records user-typed slash commands as text content shaped like:
        <command-message>plan</command-message>
        <command-name>/plan</command-name>
        <command-args>...actual prompt body...</command-args>
    """
    if "<command-name>" not in text and "<command-message>" not in text:
        return None, ""
    name_match = _COMMAND_NAME_RE.search(text)
    args_match = _COMMAND_ARGS_RE.search(text)
    name = name_match.group(1).strip() if name_match else None
    args = args_match.group(1).strip() if args_match else ""
    return name, args


def _is_control_command(name):
    """True if `name` is a non-demarcating UI/config slash command."""
    if not name:
        return False
    normalized = name.strip().lower()
    if not normalized.startswith("/"):
        normalized = "/" + normalized
    return normalized in _CONTROL_COMMANDS


def _truncate(text, max_length):
    """Trim `text` to max_length, appending an ellipsis when shortened."""
    if len(text) > max_length:
        return text[: max_length - 3] + "..."
    return text


# Some away_summary contents end with a UI hint Claude Code appends to its
# "※ recap:" display; strip it only as a trailer so mid-text mentions survive.
_RECAP_SUFFIX_RE = re.compile(r"\s*\(disable recaps in /config\)\s*$", re.IGNORECASE)


def strip_recap_suffix(text):
    """Remove the trailing "(disable recaps in /config)" UI hint, if present."""
    return _RECAP_SUFFIX_RE.sub("", text)


def extract_command_summary(text):
    """Return a readable "name: args" summary for slash-command messages, else None.

    Retained for backward compatibility; the picker/HTML paths now use
    scan_session_metadata, which keeps the command name and args body separate.
    """
    name, args = _parse_command(text)
    if name is None and not args:
        return None
    if args and name:
        return f"{name}: {args}"
    return args or name or None


@dataclass
class SessionMetadata:
    """What a session picker / archive row needs to identify a session.

    summary: best human title (args body, prose, or command name); "(no summary)"
        if nothing usable was found.
    branch: gitBranch from the first entry that carries one, or None.
    command: originating slash command (e.g. "/plan"), or None for prose sessions.
    from_control_fallback: True when `summary` is only a skipped control command
        (e.g. a /clear-only session) — lets callers avoid double-printing it.
    ai_title: Claude Code's evolving auto-generated session name (the LAST
        ``type=="ai-title"`` line), or None. This is the name `claude --resume`
        shows; independent of `summary` so picker rows keep the prompt text.
    recap: the latest away-summary recap (Claude Code's "※ recap:" text), or
        None. Untruncated — it feeds the session info card, not picker rows.
    """

    summary: str
    branch: str | None
    command: str | None
    from_control_fallback: bool
    ai_title: str | None = None
    recap: str | None = None

    @property
    def title(self):
        """Best display name: the AI title when present, else the summary."""
        return self.ai_title or self.summary


def scan_session_metadata(filepath, max_length=200):
    """Extract a session's title, git branch, and originating slash command.

    Single pass over the JSONL. Summary precedence:
      1. an explicit ``type=="summary"`` line (legacy Claude Code title format)
      2. the first non-meta user message that carries intent — skipping
         control/UI slash commands (``_CONTROL_COMMANDS``) so the real task
         command wins over a leading ``/effort``/``/clear`` toggle
      3. a skipped control command, if it was the only content (fallback)

    Independently, the LAST ``type=="ai-title"`` line (Claude Code's evolving
    auto-generated session name) is captured as ``ai_title``; the ``title``
    property prefers it over the summary heuristic.

    The summary holds the command *args body* (the real prose); the command name
    is returned separately so callers can surface it in its own column.
    """
    filepath = Path(filepath)
    explicit_summary = None
    chosen_summary = None
    chosen_command = None
    control_fallback = None  # (name, body) of the first skipped control command
    branch = None
    ai_title = None
    recap = None

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue

                if branch is None and obj.get("gitBranch"):
                    branch = obj["gitBranch"]

                # Last one wins: Claude Code rewrites the auto-title as the
                # session evolves, so keep overwriting until EOF.
                if obj.get("type") == "ai-title" and obj.get("aiTitle"):
                    ai_title = obj["aiTitle"]
                    continue

                # Latest away_summary = the session's current recap.
                if (
                    obj.get("type") == "system"
                    and obj.get("subtype") == "away_summary"
                    and obj.get("content")
                ):
                    recap = strip_recap_suffix(obj["content"]).strip()
                    continue

                if (
                    explicit_summary is None
                    and obj.get("type") == "summary"
                    and obj.get("summary")
                ):
                    explicit_summary = obj["summary"]
                    continue

                if (
                    chosen_summary is None
                    and obj.get("type") == "user"
                    and not obj.get("isMeta")
                    and obj.get("message", {}).get("content")
                ):
                    text = extract_text_from_content(obj["message"]["content"])
                    if not text:
                        continue
                    if text.startswith("<"):
                        name, body = _parse_command(text)
                        if name is None and not body:
                            # non-command wrapper (<system-reminder>, stdout) — skip
                            continue
                        if _is_control_command(name):
                            if control_fallback is None:
                                control_fallback = (name, body)
                            continue  # keep scanning for the real task command
                        chosen_summary = body or name
                        chosen_command = name
                    else:
                        chosen_summary = text
                        chosen_command = None
    except Exception:
        pass

    from_control_fallback = False
    if explicit_summary is not None:
        summary = explicit_summary
    elif chosen_summary is not None:
        summary = chosen_summary
    elif control_fallback is not None:
        chosen_command = control_fallback[0]
        summary = control_fallback[1] or control_fallback[0]
        from_control_fallback = True
    else:
        summary = "(no summary)"
        chosen_command = None

    return SessionMetadata(
        summary=_truncate(summary, max_length),
        branch=branch,
        command=chosen_command,
        from_control_fallback=from_control_fallback,
        ai_title=_truncate(ai_title, max_length) if ai_title else None,
        recap=recap or None,
    )


def format_session_choice(meta, mtime, size_bytes, project, summary_width=44):
    """Build one aligned picker row: date · size · [branch] · project · command · summary.

    The `local` and `watch --pick` pickers glob across ALL projects, so each row
    needs branch + project + command to be distinguishable (e.g. eight identical
    `/security-review` runs separated only by branch). When the summary would just
    repeat the command column (a bare `/clear` with no args) it is blanked so the
    command isn't printed twice; a command that carries a body keeps its body.
    """
    date_str = datetime.fromtimestamp(mtime).strftime("%Y-%m-%d %H:%M")
    size_kb = size_bytes / 1024
    branch = f"[{_truncate(meta.branch, 18)}]" if meta.branch else ""
    cmd = _truncate(meta.command, 16) if meta.command else ""
    if meta.command and meta.summary == meta.command:
        summary = ""  # would duplicate the command column
    else:
        summary = _truncate(meta.summary, summary_width)
    proj = _truncate(project or "", 28)
    return f"{date_str}  {size_kb:5.0f} KB  {branch:<20} {proj:<28} {cmd:<16} {summary}".rstrip()


def build_session_choices(folder, limit=10):
    """Scan recent sessions once and build aligned questionary Choices.

    Shared by the `local` and `watch --pick` pickers so both render identical
    rows (see format_session_choice). find_local_sessions already extracted
    everything a row needs — one metadata scan per file. Each Choice.value is
    the session Path.
    """
    choices = []
    for filepath, meta in find_local_sessions(folder, limit=limit):
        stat = filepath.stat()
        project = get_project_display_name(filepath.parent.name)
        display = format_session_choice(meta, stat.st_mtime, stat.st_size, project)
        choices.append(questionary.Choice(title=display, value=filepath))
    return choices


# Module-level variable for GitHub repo (set by generate_html)
_github_repo = None

# API constants
API_BASE_URL = "https://api.anthropic.com/v1"
ANTHROPIC_VERSION = "2023-06-01"


def get_session_summary(filepath, max_length=200):
    """Extract a human-readable summary from a session file.

    Supports both JSON and JSONL formats.
    Returns a summary string or "(no summary)" if none found.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".jsonl":
        return scan_session_metadata(filepath, max_length).summary
    try:
        # For JSON files, try to get first user message
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
        loglines = data.get("loglines", [])
        for entry in loglines:
            if entry.get("type") == "user":
                msg = entry.get("message", {})
                content = msg.get("content", "")
                text = extract_text_from_content(content)
                if text:
                    return _truncate(text, max_length)
        return "(no summary)"
    except Exception:
        return "(no summary)"


def get_session_title(filepath, max_length=80):
    """Best display name for a session file, or None when nothing usable.

    JSONL: the last aiTitle, else the summary heuristic. JSON: the first user
    message. "(no summary)" maps to None so callers fall back to the generic
    page title. Shorter default truncation than the picker — this feeds
    browser-tab titles.
    """
    filepath = Path(filepath)
    if filepath.suffix == ".jsonl":
        title = scan_session_metadata(filepath, max_length=max_length).title
    else:
        title = get_session_summary(filepath, max_length=max_length)
    if not title or title == "(no summary)":
        return None
    return title


def find_local_sessions(folder, limit=10):
    """Find recent JSONL session files in the given folder.

    Returns a list of (Path, SessionMetadata) tuples sorted by modification
    time. Excludes agent files and warmup/empty sessions. Carrying the full
    metadata (not just the summary string) lets pickers render branch/project/
    command columns without re-scanning every file.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    results = []
    for f in folder.glob("**/*.jsonl"):
        if f.name.startswith("agent-"):
            continue
        meta = scan_session_metadata(f)
        # Skip boring/empty sessions
        if meta.summary.lower() == "warmup" or meta.summary == "(no summary)":
            continue
        results.append((f, meta))

    # Sort by modification time, most recent first
    results.sort(key=lambda x: x[0].stat().st_mtime, reverse=True)
    return results[:limit]


def resolve_active_session(folder, session=None):
    """Resolve which session file the live `watch` view should tail.

    If `session` is given, use it directly. Otherwise return the most-recently-
    modified `.jsonl` under `folder` (the session Claude is actively writing),
    excluding agent sidechains and warmup sessions. Unlike find_local_sessions
    we do NOT skip "(no summary)" sessions — a just-started session that has no
    summary yet must still be tailable. Returns a Path, or None if none found.
    """
    if session:
        return Path(session)

    folder = Path(folder)
    if not folder.exists():
        return None

    files = sorted(
        (f for f in folder.glob("**/*.jsonl") if not f.name.startswith("agent-")),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for f in files:
        if get_session_summary(f).lower() == "warmup":
            continue
        return f
    return None


def get_project_display_name(folder_name):
    """Convert encoded folder name to readable project name.

    Claude Code stores projects in folders like:
    - -home-user-projects-myproject -> myproject
    - -mnt-c-Users-name-Projects-app -> app

    For nested paths under common roots (home, projects, code, Users, etc.),
    extracts the meaningful project portion.
    """
    # Common path prefixes to strip
    prefixes_to_strip = [
        "-home-",
        "-mnt-c-Users-",
        "-mnt-c-users-",
        "-Users-",
    ]

    name = folder_name
    for prefix in prefixes_to_strip:
        if name.lower().startswith(prefix.lower()):
            name = name[len(prefix) :]
            break

    # Split on dashes and find meaningful parts
    parts = name.split("-")

    # Common intermediate directories to skip
    skip_dirs = {"projects", "code", "repos", "src", "dev", "work", "documents"}

    # Find the first meaningful part (after skipping username and common dirs)
    meaningful_parts = []
    found_project = False

    for i, part in enumerate(parts):
        if not part:
            continue
        # Skip the first part if it looks like a username (before common dirs)
        if i == 0 and not found_project:
            # Check if next parts contain common dirs
            remaining = [p.lower() for p in parts[i + 1 :]]
            if any(d in remaining for d in skip_dirs):
                continue
        if part.lower() in skip_dirs:
            found_project = True
            continue
        meaningful_parts.append(part)
        found_project = True

    if meaningful_parts:
        return "-".join(meaningful_parts)

    # Fallback: return last non-empty part or original
    for part in reversed(parts):
        if part:
            return part
    return folder_name


def find_all_sessions(folder, include_agents=False):
    """Find all sessions in a Claude projects folder, grouped by project.

    Returns a list of project dicts, each containing:
    - name: display name for the project
    - path: Path to the project folder
    - sessions: list of session dicts with path, summary, mtime, size

    Sessions are sorted by modification time (most recent first) within each project.
    Projects are sorted by their most recent session.
    """
    folder = Path(folder)
    if not folder.exists():
        return []

    projects = {}

    for session_file in folder.glob("**/*.jsonl"):
        # Skip agent files unless requested
        if not include_agents and session_file.name.startswith("agent-"):
            continue

        # Get metadata and skip boring sessions
        meta = scan_session_metadata(session_file)
        if meta.summary.lower() == "warmup" or meta.summary == "(no summary)":
            continue

        # Group by display name so folders that resolve to the same project
        # (e.g., C--projects-devjig and d--projects-devjig after a drive move)
        # merge into a single entry instead of silently overwriting each other
        # when the display name is later used as an output directory.
        display_name = get_project_display_name(session_file.parent.name)

        if display_name not in projects:
            projects[display_name] = {
                "name": display_name,
                "sessions": [],
            }

        stat = session_file.stat()
        projects[display_name]["sessions"].append(
            {
                "path": session_file,
                "summary": meta.summary,
                "title": meta.title,
                "recap": meta.recap,
                "branch": meta.branch,
                "command": meta.command,
                "mtime": stat.st_mtime,
                "size": stat.st_size,
            }
        )

    # Sort sessions within each project by mtime (most recent first)
    for project in projects.values():
        project["sessions"].sort(key=lambda s: s["mtime"], reverse=True)

    # Convert to list and sort projects by most recent session
    result = list(projects.values())
    result.sort(
        key=lambda p: p["sessions"][0]["mtime"] if p["sessions"] else 0, reverse=True
    )

    return result


def _get_tool_version():
    """Return the installed package version, or 'unknown' for source-tree runs.

    Used by the freshness-check feature to detect that the tool was upgraded
    since a transcript was last rendered, so the user can re-render with the
    new templates/CSS/JS.
    """
    try:
        from importlib.metadata import version, PackageNotFoundError

        try:
            return version("claude-code-transcripts")
        except PackageNotFoundError:
            return "unknown"
    except ImportError:
        return "unknown"


def _session_state_path(session_dir):
    return Path(session_dir) / ".cct-state.json"


def _read_session_state(session_dir):
    """Return parsed session state dict, or None on any read/parse failure.

    A None return is the signal to treat the session as stale (the sidecar
    is missing or malformed, so we can't trust the existing output).
    """
    path = _session_state_path(session_dir)
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


def _write_session_state(session_dir, source_path, source_mtime):
    """Atomically write the per-session state sidecar.

    Writes to a temp file then os.replace so a crash mid-write can't leave a
    half-written sidecar that would mislead the next freshness check.
    """
    state = {
        "tool_version": _get_tool_version(),
        "source_mtime": source_mtime,
        "source_path": str(source_path),
    }
    target = _session_state_path(session_dir)
    tmp = target.with_name(target.name + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(state, f, indent=2)
    os.replace(tmp, target)


def _session_is_stale(session_info, session_dir, current_version):
    """Return (is_stale, reason). reason is empty when not stale.

    Order of checks matters: a missing output dominates everything else, then
    sidecar integrity, then version mismatch, then source mtime.
    """
    session_dir = Path(session_dir)
    if not (session_dir / "index.html").exists():
        return True, "no output yet"
    state = _read_session_state(session_dir)
    if state is None:
        return True, "sidecar missing or malformed"
    stored_version = state.get("tool_version", "unknown")
    if stored_version != current_version:
        return True, f"tool version changed: {stored_version} -> {current_version}"
    stored_mtime = state.get("source_mtime")
    if not isinstance(stored_mtime, (int, float)):
        return True, "sidecar missing source mtime"
    if session_info["mtime"] > stored_mtime:
        return True, "source modified since render"
    return False, ""


def generate_batch_html(
    source_folder,
    output_dir,
    include_agents=False,
    progress_callback=None,
    only_stale=False,
):
    """Generate HTML archive for all sessions in a Claude projects folder.

    Creates:
    - Master index.html listing all projects
    - Per-project directories with index.html listing sessions
    - Per-session directories with transcript pages

    Args:
        source_folder: Path to the Claude projects folder
        output_dir: Path for output archive
        include_agents: Whether to include agent-* session files
        progress_callback: Optional callback(project_name, session_name, current, total)
            called after each session is processed
        only_stale: When True, skip sessions whose output is already up to date
            (per `.cct-state.json` sidecar). The per-project index regenerates
            only for projects that had at least one stale session; the master
            index regenerates if any project had changes.

    Returns statistics dict. Always includes total_projects, total_sessions
    (== regenerated count), failed_sessions, output_dir. When only_stale is
    True, also includes skipped_sessions and regenerated_sessions.
    """
    source_folder = Path(source_folder)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    projects = find_all_sessions(source_folder, include_agents=include_agents)

    current_version = _get_tool_version() if only_stale else None

    # When only_stale, the progress total reflects the stale count, not the
    # full session count — otherwise the progress bar lies.
    if only_stale:
        total_session_count = 0
        for project in projects:
            for session in project["sessions"]:
                session_dir = output_dir / project["name"] / session["path"].stem
                is_stale, _ = _session_is_stale(session, session_dir, current_version)
                if is_stale:
                    total_session_count += 1
    else:
        total_session_count = sum(len(p["sessions"]) for p in projects)

    processed_count = 0
    successful_sessions = 0
    skipped_sessions = 0
    failed_sessions = []
    any_project_changed = False

    for project in projects:
        project_dir = output_dir / project["name"]
        project_dir.mkdir(exist_ok=True)
        project_changed = False

        for session in project["sessions"]:
            session_name = session["path"].stem
            session_dir = project_dir / session_name

            if only_stale:
                is_stale, _ = _session_is_stale(session, session_dir, current_version)
                if not is_stale:
                    skipped_sessions += 1
                    continue

            # Generate transcript HTML with error handling
            try:
                # Title/recap from the scan find_all_sessions already did —
                # avoids a second metadata pass per archived session.
                generate_html(
                    session["path"],
                    session_dir,
                    title=session.get("title"),
                    recap=session.get("recap"),
                )
                successful_sessions += 1
                # Record the mtime we actually rendered against, not a re-stat
                # — if the source kept growing during conversion, the next run
                # should still detect it as stale.
                _write_session_state(session_dir, session["path"], session["mtime"])
                project_changed = True
            except Exception as e:
                failed_sessions.append(
                    {
                        "project": project["name"],
                        "session": session_name,
                        "error": str(e),
                    }
                )

            processed_count += 1

            if progress_callback:
                progress_callback(
                    project["name"], session_name, processed_count, total_session_count
                )

        # In full-rebuild mode, always regenerate the per-project index. In
        # incremental mode, only when this project actually had a regeneration
        # — otherwise an unchanged project's index would be needlessly rewritten.
        if not only_stale or project_changed:
            _generate_project_index(project, project_dir)

        if project_changed:
            any_project_changed = True

    # Master index re-renders on full rebuild always, or in incremental mode
    # whenever any project changed (session counts/dates feed it).
    if not only_stale or any_project_changed:
        _generate_master_index(projects, output_dir)

    stats = {
        "total_projects": len(projects),
        "total_sessions": successful_sessions,
        "failed_sessions": failed_sessions,
        "output_dir": output_dir,
    }
    if only_stale:
        stats["regenerated_sessions"] = successful_sessions
        stats["skipped_sessions"] = skipped_sessions
    return stats


def _generate_project_index(project, output_dir):
    """Generate index.html for a single project."""
    template = get_template("project_index.html")

    # Format sessions for template
    sessions_data = []
    for session in project["sessions"]:
        mod_time = datetime.fromtimestamp(session["mtime"])
        sessions_data.append(
            {
                "name": session["path"].stem,
                "summary": session["summary"],
                "branch": session.get("branch"),
                "command": session.get("command"),
                "date": mod_time.strftime("%Y-%m-%d %H:%M"),
                "size_kb": session["size"] / 1024,
            }
        )

    html_content = template.render(
        project_name=project["name"],
        sessions=sessions_data,
        session_count=len(sessions_data),
        css=CSS,
        js=JS,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_content, encoding="utf-8")


def _generate_master_index(projects, output_dir):
    """Generate master index.html listing all projects."""
    template = get_template("master_index.html")

    # Format projects for template
    projects_data = []
    total_sessions = 0

    for project in projects:
        session_count = len(project["sessions"])
        total_sessions += session_count

        # Get most recent session date
        if project["sessions"]:
            most_recent = datetime.fromtimestamp(project["sessions"][0]["mtime"])
            recent_date = most_recent.strftime("%Y-%m-%d")
        else:
            recent_date = "N/A"

        projects_data.append(
            {
                "name": project["name"],
                "session_count": session_count,
                "recent_date": recent_date,
            }
        )

    html_content = template.render(
        projects=projects_data,
        total_projects=len(projects),
        total_sessions=total_sessions,
        css=CSS,
        js=JS,
    )

    output_path = output_dir / "index.html"
    output_path.write_text(html_content, encoding="utf-8")


def parse_session_file(filepath):
    """Parse a session file and return normalized data.

    Supports both JSON and JSONL formats.
    Returns a dict with 'loglines' key containing the normalized entries.
    """
    filepath = Path(filepath)

    if filepath.suffix == ".jsonl":
        return _parse_jsonl_file(filepath)
    else:
        # Standard JSON format
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)


def _normalize_jsonl_obj(obj, include_meta=False):
    """Normalize one parsed JSONL object to a standard logline entry.

    Returns the entry dict for user/assistant messages, or None for any other
    entry type (summary, file-history-snapshot, etc.). Shared by the batch
    parser (_parse_jsonl_file) and the live tail reader (read_new_loglines).

    With ``include_meta=True`` (the tail path only), additionally returns
    typed meta entries the live view reacts to but the static parser must
    never see: ``{"type": "ai-title", "title": ...}`` for session-name
    changes and ``{"type": "away-summary", "text": ..., "timestamp": ...}``
    for recap updates. The default keeps the static contract byte-identical.
    """
    entry_type = obj.get("type")

    if include_meta and entry_type == "ai-title":
        if obj.get("aiTitle"):
            return {"type": "ai-title", "title": obj["aiTitle"]}
        return None

    if include_meta and entry_type == "system":
        if obj.get("subtype") == "away_summary" and obj.get("content"):
            return {
                "type": "away-summary",
                "text": strip_recap_suffix(obj["content"]).strip(),
                "timestamp": obj.get("timestamp", ""),
            }
        return None

    # Skip non-message entries
    if entry_type not in ("user", "assistant"):
        return None

    # Convert to standard format
    entry = {
        "type": entry_type,
        "timestamp": obj.get("timestamp", ""),
        "message": obj.get("message", {}),
    }

    # Preserve isCompactSummary if present
    if obj.get("isCompactSummary"):
        entry["isCompactSummary"] = True

    return entry


def _parse_jsonl_file(filepath):
    """Parse JSONL file and convert to standard format."""
    loglines = []

    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            entry = _normalize_jsonl_obj(obj)
            if entry is not None:
                loglines.append(entry)

    return {"loglines": loglines}


def read_new_loglines(path, offset):
    """Read complete JSONL lines appended to `path` after byte `offset`.

    Returns ``(loglines, new_offset)``. Only consumes through the last newline,
    so a partially-written trailing line is left for a later call. ``new_offset``
    advances past every complete line, including ones that are blank, malformed,
    or non-message (so the tail never re-reads them). Splitting on ``b"\\n"`` is
    UTF-8-safe: 0x0A never appears inside a multibyte sequence.
    """
    path = Path(path)
    with open(path, "rb") as f:
        f.seek(offset)
        data = f.read()

    last_nl = data.rfind(b"\n")
    if last_nl == -1:
        return [], offset  # no complete line yet

    consumed = data[: last_nl + 1]
    new_offset = offset + len(consumed)

    loglines = []
    for raw in consumed.split(b"\n"):
        if not raw.strip():
            continue
        try:
            line = raw.decode("utf-8")
        except UnicodeDecodeError:
            continue
        try:
            obj = json.loads(line)
        except json.JSONDecodeError:
            continue
        # Tail-only: surface meta entries (title changes) alongside messages.
        entry = _normalize_jsonl_obj(obj, include_meta=True)
        if entry is not None:
            loglines.append(entry)

    return loglines, new_offset


def format_sse_event(event_name, data):
    """Frame one Server-Sent Event. JSON-encoding `data` keeps the payload on a
    single `data:` line (embedded newlines are escaped), so multi-line HTML is
    SSE-safe by construction."""
    return f"event: {event_name}\ndata: {json.dumps(data)}\n\n"


class CredentialsError(Exception):
    """Raised when credentials cannot be obtained."""

    pass


def get_access_token_from_keychain():
    """Get access token from macOS keychain.

    Returns the access token or None if not found.
    Raises CredentialsError with helpful message on failure.
    """
    if platform.system() != "Darwin":
        return None

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-a",
                os.environ.get("USER", ""),
                "-s",
                "Claude Code-credentials",
                "-w",
            ],
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return None

        # Parse the JSON to get the access token
        creds = json.loads(result.stdout.strip())
        return creds.get("claudeAiOauth", {}).get("accessToken")
    except (json.JSONDecodeError, subprocess.SubprocessError):
        return None


def get_org_uuid_from_config():
    """Get organization UUID from ~/.claude.json.

    Returns the organization UUID or None if not found.
    """
    config_path = Path.home() / ".claude.json"
    if not config_path.exists():
        return None

    try:
        with open(config_path) as f:
            config = json.load(f)
        return config.get("oauthAccount", {}).get("organizationUuid")
    except (json.JSONDecodeError, IOError):
        return None


def get_api_headers(token, org_uuid):
    """Build API request headers."""
    return {
        "Authorization": f"Bearer {token}",
        "anthropic-version": ANTHROPIC_VERSION,
        "Content-Type": "application/json",
        "x-organization-uuid": org_uuid,
    }


def fetch_sessions(token, org_uuid):
    """Fetch list of sessions from the API.

    Returns the sessions data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(f"{API_BASE_URL}/sessions", headers=headers, timeout=30.0)
    response.raise_for_status()
    return response.json()


def fetch_session(token, org_uuid, session_id):
    """Fetch a specific session from the API.

    Returns the session data as a dict.
    Raises httpx.HTTPError on network/API errors.
    """
    headers = get_api_headers(token, org_uuid)
    response = httpx.get(
        f"{API_BASE_URL}/session_ingress/session/{session_id}",
        headers=headers,
        timeout=60.0,
    )
    response.raise_for_status()
    return response.json()


def detect_github_repo(loglines):
    """
    Detect GitHub repo from git push output in tool results.

    Looks for patterns like:
    - github.com/owner/repo/pull/new/branch (from git push messages)

    Returns the first detected repo (owner/name) or None.
    """
    for entry in loglines:
        message = entry.get("message", {})
        content = message.get("content", [])
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "tool_result":
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    match = GITHUB_REPO_PATTERN.search(result_content)
                    if match:
                        return match.group(1)
    return None


def extract_repo_from_session(session):
    """Extract GitHub repo from session metadata.

    Looks in session_context.outcomes for git_info.repo,
    or parses from session_context.sources URL.

    Returns repo as "owner/name" or None.
    """
    context = session.get("session_context", {})

    # Try outcomes first (has clean repo format)
    outcomes = context.get("outcomes", [])
    for outcome in outcomes:
        if outcome.get("type") == "git_repository":
            git_info = outcome.get("git_info", {})
            repo = git_info.get("repo")
            if repo:
                return repo

    # Fall back to sources URL
    sources = context.get("sources", [])
    for source in sources:
        if source.get("type") == "git_repository":
            url = source.get("url", "")
            # Parse github.com/owner/repo from URL
            if "github.com/" in url:
                # Extract owner/repo from https://github.com/owner/repo
                match = re.search(r"github\.com/([^/]+/[^/]+?)(?:\.git)?$", url)
                if match:
                    return match.group(1)

    return None


def enrich_sessions_with_repos(sessions, token=None, org_uuid=None, fetch_fn=None):
    """Enrich sessions with repo information from session metadata.

    Args:
        sessions: List of session dicts from the API
        token: Unused (kept for backward compatibility)
        org_uuid: Unused (kept for backward compatibility)
        fetch_fn: Unused (kept for backward compatibility)

    Returns:
        List of session dicts with 'repo' key added
    """
    enriched = []
    for session in sessions:
        session_copy = dict(session)
        session_copy["repo"] = extract_repo_from_session(session)
        enriched.append(session_copy)
    return enriched


def filter_sessions_by_repo(sessions, repo):
    """Filter sessions by repo.

    Args:
        sessions: List of session dicts with 'repo' key
        repo: Repo to filter by (owner/name), or None to return all

    Returns:
        Filtered list of sessions
    """
    if repo is None:
        return sessions
    return [s for s in sessions if s.get("repo") == repo]


def format_json(obj):
    try:
        if isinstance(obj, str):
            obj = json.loads(obj)
        formatted = json.dumps(obj, indent=2, ensure_ascii=False)
        return f'<pre class="json">{html.escape(formatted)}</pre>'
    except (json.JSONDecodeError, TypeError):
        return f"<pre>{html.escape(str(obj))}</pre>"


# CommonMark-compliant renderer. Claude emits CommonMark/GFM, so its
# compaction summaries indent nested bullets 3 spaces (aligned past the "1. "
# marker) and start lists on the line right after a header. The legacy
# python-markdown engine mis-parsed both (4-space nesting rule + no
# list-interrupts-paragraph), flattening sub-bullets into the parent <ol> and
# absorbing the first item into a <p>. The "commonmark" preset keeps raw-HTML
# passthrough (html=True) to match the old behavior; ``table`` restores the GFM
# tables that the old ``tables`` extension provided (fenced code is built in).
_md = MarkdownIt("commonmark").enable("table")

# A GFM table delimiter row, e.g. "| --- | :--: |" or "---|---". Used only to
# detect which lines are table rows so we can scope the pipe-escaping fix below.
_TABLE_DELIMITER_RE = re.compile(r"^\s*\|?\s*:?-+:?\s*(\|\s*:?-+:?\s*)+\|?\s*$")
# An unescaped "|" (not already preceded by a backslash).
_UNESCAPED_PIPE_RE = re.compile(r"(?<!\\)\|")


def _escape_pipes_in_code_spans(line):
    """Escape unescaped ``|`` that fall inside inline code spans on a single
    table-row line, so the GFM table parser doesn't split a cell mid-code-span.
    markdown-it unescapes ``\\|`` back to ``|`` inside code spans *in table
    cells*, so this is a no-op on the rendered code text — it only prevents the
    cell-shattering. Backtick runs of any length are matched to their closing
    run of equal length, mirroring CommonMark code-span rules."""
    out = []
    i, n = 0, len(line)
    while i < n:
        ch = line[i]
        if ch == "\\" and i + 1 < n:  # keep existing backslash escapes intact
            out.append(line[i : i + 2])
            i += 2
            continue
        if ch == "`":
            j = i
            while j < n and line[j] == "`":
                j += 1
            run = j - i  # opening backtick-run length
            k = j
            while k < n:
                if line[k] == "`":
                    m = k
                    while m < n and line[m] == "`":
                        m += 1
                    if m - k == run:  # matching closing run -> span is line[j:k]
                        content = _UNESCAPED_PIPE_RE.sub(r"\\|", line[j:k])
                        out.append(line[i:j] + content + line[k:m])
                        i = m
                        break
                    k = m
                else:
                    k += 1
            else:  # no closing run: not a code span, emit the rest verbatim
                out.append(line[i:])
                i = n
            continue
        out.append(ch)
        i += 1
    return "".join(out)


def _protect_table_code_span_pipes(text):
    """Pre-escape bare ``|`` inside inline code spans, but only on lines that
    belong to a GFM table. GFM technically requires ``\\|`` there, yet Claude
    transcripts routinely contain bare pipes (shell commands in comparison
    tables); the strict parser would shatter those cells. Scoped to table rows
    so prose code spans like ``ps aux | grep`` are never touched."""
    if "|" not in text or "`" not in text:
        return text
    lines = text.split("\n")
    in_table = [False] * len(lines)
    for idx, ln in enumerate(lines):
        if "|" in ln and _TABLE_DELIMITER_RE.match(ln):
            if idx > 0 and "|" in lines[idx - 1]:
                in_table[idx - 1] = True  # header row
            in_table[idx] = True  # delimiter row (no code spans; harmless)
            j = idx + 1
            while j < len(lines) and lines[j].strip() and "|" in lines[j]:
                in_table[j] = True  # body rows until blank/non-row line
                j += 1
    if not any(in_table):
        return text
    return "\n".join(
        _escape_pipes_in_code_spans(ln) if in_table[idx] and "`" in ln else ln
        for idx, ln in enumerate(lines)
    )


def render_markdown_text(text):
    if not text:
        return ""
    # markdown-it appends a trailing newline after block elements; strip it so
    # output stays byte-for-byte identical to the legacy renderer everywhere
    # except the previously-broken nested-list case.
    return _md.render(_protect_table_code_span_pipes(text)).rstrip("\n")


def is_json_like(text):
    if not text or not isinstance(text, str):
        return False
    text = text.strip()
    return (text.startswith("{") and text.endswith("}")) or (
        text.startswith("[") and text.endswith("]")
    )


def render_todo_write(tool_input, tool_id):
    todos = tool_input.get("todos", [])
    if not todos:
        return ""
    return _macros.todo_list(todos, tool_id)


def render_write_tool(tool_input, tool_id):
    """Render Write tool calls with file path header and content preview."""
    file_path = tool_input.get("file_path", "Unknown file")
    content = tool_input.get("content", "")
    return _macros.write_tool(file_path, content, tool_id)


def render_edit_tool(tool_input, tool_id):
    """Render Edit tool calls with diff-like old/new display."""
    file_path = tool_input.get("file_path", "Unknown file")
    old_string = tool_input.get("old_string", "")
    new_string = tool_input.get("new_string", "")
    replace_all = tool_input.get("replace_all", False)
    return _macros.edit_tool(file_path, old_string, new_string, replace_all, tool_id)


def render_bash_tool(tool_input, tool_id):
    """Render Bash tool calls with command as plain text."""
    command = tool_input.get("command", "")
    description = tool_input.get("description", "")
    return _macros.bash_tool(command, description, tool_id)


def render_content_block(block, block_id=None):
    if not isinstance(block, dict):
        return f"<p>{html.escape(str(block))}</p>"
    block_type = block.get("type", "")
    if block_type == "image":
        source = block.get("source", {})
        media_type = source.get("media_type", "image/png")
        data = source.get("data", "")
        return _macros.image_block(media_type, data)
    elif block_type == "thinking":
        thinking_text = block.get("thinking", "")
        content_html = render_markdown_text(thinking_text)
        return _macros.thinking(content_html, thinking_text, block_id or "")
    elif block_type == "text":
        text = block.get("text", "")
        content_html = render_markdown_text(text)
        return _macros.assistant_text(content_html, text, block_id or "")
    elif block_type == "tool_use":
        tool_name = block.get("name", "Unknown tool")
        tool_input = block.get("input", {})
        tool_id = block.get("id", "")
        if tool_name == "TodoWrite":
            return render_todo_write(tool_input, tool_id)
        if tool_name == "Write":
            return render_write_tool(tool_input, tool_id)
        if tool_name == "Edit":
            return render_edit_tool(tool_input, tool_id)
        if tool_name == "Bash":
            return render_bash_tool(tool_input, tool_id)
        description = tool_input.get("description", "")
        display_input = {k: v for k, v in tool_input.items() if k != "description"}
        input_json = json.dumps(display_input, indent=2, ensure_ascii=False)
        return _macros.tool_use(
            tool_name, description, input_json, tool_id, block_id or ""
        )
    elif block_type == "tool_result":
        content = block.get("content", "")
        is_error = block.get("is_error", False)
        has_images = False

        # Check for git commits and render with styled cards
        if isinstance(content, str):
            commits_found = list(COMMIT_PATTERN.finditer(content))
            if commits_found:
                # Build commit cards + remaining content
                parts = []
                last_end = 0
                for match in commits_found:
                    # Add any content before this commit
                    before = content[last_end : match.start()].strip()
                    if before:
                        parts.append(f"<pre>{html.escape(before)}</pre>")

                    commit_hash = match.group(1)
                    commit_msg = match.group(2)
                    parts.append(
                        _macros.commit_card(commit_hash, commit_msg, _github_repo)
                    )
                    last_end = match.end()

                # Add any remaining content after last commit
                after = content[last_end:].strip()
                if after:
                    parts.append(f"<pre>{html.escape(after)}</pre>")

                content_html = "".join(parts)
            else:
                content_html = f"<pre>{html.escape(content)}</pre>"
        elif isinstance(content, list):
            # Handle tool result content that contains multiple blocks (text, images, etc.)
            parts = []
            for item in content:
                if isinstance(item, dict):
                    item_type = item.get("type", "")
                    if item_type == "text":
                        text = item.get("text", "")
                        if text:
                            parts.append(f"<pre>{html.escape(text)}</pre>")
                    elif item_type == "image":
                        source = item.get("source", {})
                        media_type = source.get("media_type", "image/png")
                        data = source.get("data", "")
                        if data:
                            parts.append(_macros.image_block(media_type, data))
                            has_images = True
                    else:
                        # Unknown type, render as JSON
                        parts.append(format_json(item))
                else:
                    # Non-dict item, escape as text
                    parts.append(f"<pre>{html.escape(str(item))}</pre>")
            content_html = "".join(parts) if parts else format_json(content)
        elif is_json_like(content):
            content_html = format_json(content)
        else:
            content_html = format_json(content)
        return _macros.tool_result(content_html, is_error, has_images)
    else:
        return format_json(block)


def render_user_message_content(message_data):
    content = message_data.get("content", "")
    if isinstance(content, str):
        if is_json_like(content):
            return _macros.user_content(format_json(content))
        return _macros.user_content(render_markdown_text(content), content)
    elif isinstance(content, list):
        return "".join(render_content_block(block) for block in content)
    return f"<p>{html.escape(str(content))}</p>"


def render_assistant_message(message_data, msg_id=None):
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return f"<p>{html.escape(str(content))}</p>"
    if msg_id is None:
        return "".join(render_content_block(block) for block in content)
    return "".join(
        render_content_block(block, block_id=bid)
        for bid, block in iter_assistant_blocks(message_data, msg_id)
    )


def make_msg_id(timestamp):
    return f"msg-{timestamp.replace(':', '-').replace('.', '-')}"


def block_anchor(msg_id, index):
    """Stable element id for content block `index` of message `msg_id`."""
    return f"{msg_id}-b{index}"


def iter_assistant_blocks(message_data, msg_id):
    """Yield (block_id, block) for every content block of an assistant message.

    The single enumeration shared by rendering and artifact extraction, so
    deep-link anchors can never drift from the rendered ids. The index is the
    content-array position — every block counts, whether or not its renderer
    emits an id.
    """
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return
    for i, block in enumerate(content):
        yield block_anchor(msg_id, i), block


# Artifact extraction: the deep-linkable moments of a session (insights,
# substantial thinking, plans, per-prompt completion replies) that feed the
# session card's prompt tree and the index timeline.
INSIGHT_MARKER = "★ Insight"
ARTIFACT_LABEL_LENGTH = 60
ARTIFACT_ICONS = {"insight": "★", "thinking": "💭", "plan": "📋", "completion": "✓"}

_MD_HEADING_RE = re.compile(r"^#{1,6}\s+(.+)$", re.MULTILINE)


def artifact_label(text, max_length=ARTIFACT_LABEL_LENGTH):
    """Whitespace-collapsed, capped label for an artifact link."""
    return _truncate(" ".join(text.split()), max_length)


def is_insight_text(text):
    """True when a text block is an Insight callout (★ Insight marker)."""
    return INSIGHT_MARKER in text


def insight_label(text):
    """First meaningful line after the ★ Insight marker, as the link label.

    Separator lines (box-drawing dashes/backticks) and bullet prefixes are
    skipped/stripped; falls back to the generic label when nothing usable
    follows the marker.
    """
    lines = text.split("\n")
    for i, line in enumerate(lines):
        if INSIGHT_MARKER not in line:
            continue
        for follow in lines[i + 1 :]:
            cleaned = follow.strip().strip("`").strip()
            cleaned = cleaned.lstrip("-*• ").strip()
            if cleaned and set(cleaned) - {"─", "-", "`"}:
                return artifact_label(cleaned)
        break
    return artifact_label(text)


def plan_label(tool_input):
    """Label for an ExitPlanMode artifact.

    Legacy sessions carried the plan markdown in input.plan — use its first
    heading. Modern input is {"allowedPrompts": [...]} with the plan text in
    the tool result, so a generic label is all the input offers.
    """
    plan_md = tool_input.get("plan") if isinstance(tool_input, dict) else None
    if plan_md:
        match = _MD_HEADING_RE.search(plan_md)
        if match:
            return artifact_label("Plan: " + match.group(1))
    return "Plan presented"


def iter_block_artifacts(message_data, msg_id):
    """Yield (kind, anchor, label) for an assistant message's notable blocks.

    kind: "thinking" (only blocks >= LONG_TEXT_THRESHOLD chars), "insight",
    "plan" (ExitPlanMode), or "text" — a plain text block, which is not an
    artifact itself but the running completion candidate. Anchors come from
    iter_assistant_blocks, the same enumeration rendering uses.
    """
    for anchor, block in iter_assistant_blocks(message_data, msg_id):
        if not isinstance(block, dict):
            continue
        btype = block.get("type")
        if btype == "thinking":
            text = block.get("thinking", "")
            if len(text) >= LONG_TEXT_THRESHOLD:
                yield "thinking", anchor, artifact_label(text)
        elif btype == "text":
            text = block.get("text", "")
            if not text:
                continue
            if is_insight_text(text):
                yield "insight", anchor, insight_label(text)
            else:
                yield "text", anchor, artifact_label(text)
        elif btype == "tool_use" and block.get("name") == "ExitPlanMode":
            yield "plan", anchor, plan_label(block.get("input", {}))


def extract_conversation_artifacts(message_groups):
    """Collect deep-linkable artifacts for one prompt's conversation.

    message_groups: [(page_num, messages)] — the prompt's conversation first,
    then each continuation conversation with ITS page (continuations can
    cross page boundaries); messages are (log_type, message_json, timestamp)
    tuples as built by _build_conversations.

    Returns [{"type","label","anchor","page"}] in block order. The last
    text-or-insight block becomes the prompt's completion: a plain text block
    is appended as type "completion"; an insight is retyped in place (its
    richer label kept) so it isn't listed twice.
    """
    artifacts = []
    # (artifact_index | None, anchor, label, page) of the last text/insight
    last_text = None
    for page, messages in message_groups:
        for log_type, message_json, timestamp in messages:
            if log_type != "assistant" or not message_json:
                continue
            try:
                message_data = json.loads(message_json)
            except json.JSONDecodeError:
                continue
            msg_id = make_msg_id(timestamp)
            for kind, anchor, label in iter_block_artifacts(message_data, msg_id):
                if kind == "text":
                    last_text = (None, anchor, label, page)
                    continue
                artifacts.append(
                    {"type": kind, "label": label, "anchor": anchor, "page": page}
                )
                if kind == "insight":
                    last_text = (len(artifacts) - 1, anchor, label, page)
    if last_text is not None:
        idx, anchor, label, page = last_text
        if idx is None:
            artifacts.append(
                {"type": "completion", "label": label, "anchor": anchor, "page": page}
            )
        else:
            artifacts[idx]["type"] = "completion"
    return artifacts


def extract_entry_artifacts(entry):
    """Live per-logline artifact split: (immediate, last_text).

    immediate = [{"type","label","id"}] to emit as the entry renders
    (insight/thinking/plan); last_text = {"id","label"} for the entry's final
    text-or-insight block — the running completion candidate the server holds
    until the next prompt arrives (insight keeps its richer label) — or None.
    """
    if entry.get("type") != "assistant":
        return [], None
    message_data = entry.get("message", {})
    msg_id = make_msg_id(entry.get("timestamp", ""))
    immediate = []
    last_text = None
    for kind, anchor, label in iter_block_artifacts(message_data, msg_id):
        if kind == "text":
            last_text = {"id": anchor, "label": label}
            continue
        immediate.append({"type": kind, "label": label, "id": anchor})
        if kind == "insight":
            last_text = {"id": anchor, "label": label}
    return immediate, last_text


def analyze_conversation(messages):
    """Analyze messages in a conversation to extract stats and long texts."""
    tool_counts = {}  # tool_name -> count
    long_texts = []
    commits = []  # list of (hash, message, timestamp)

    for log_type, message_json, timestamp in messages:
        if not message_json:
            continue
        try:
            message_data = json.loads(message_json)
        except json.JSONDecodeError:
            continue

        content = message_data.get("content", [])
        if not isinstance(content, list):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue
            block_type = block.get("type", "")

            if block_type == "tool_use":
                tool_name = block.get("name", "Unknown")
                tool_counts[tool_name] = tool_counts.get(tool_name, 0) + 1
            elif block_type == "tool_result":
                # Check for git commit output
                result_content = block.get("content", "")
                if isinstance(result_content, str):
                    for match in COMMIT_PATTERN.finditer(result_content):
                        commits.append((match.group(1), match.group(2), timestamp))
            elif block_type == "text":
                text = block.get("text", "")
                if len(text) >= LONG_TEXT_THRESHOLD:
                    long_texts.append(text)

    return {
        "tool_counts": tool_counts,
        "long_texts": long_texts,
        "commits": commits,
    }


def format_tool_stats(tool_counts):
    """Format tool counts into a concise summary string."""
    if not tool_counts:
        return ""

    # Abbreviate common tool names
    abbrev = {
        "Bash": "bash",
        "Read": "read",
        "Write": "write",
        "Edit": "edit",
        "Glob": "glob",
        "Grep": "grep",
        "Task": "task",
        "TodoWrite": "todo",
        "WebFetch": "fetch",
        "WebSearch": "search",
    }

    parts = []
    for name, count in sorted(tool_counts.items(), key=lambda x: -x[1]):
        short_name = abbrev.get(name, name.lower())
        parts.append(f"{count} {short_name}")

    return " · ".join(parts)


def is_tool_result_message(message_data):
    """Check if a message contains only tool_result blocks."""
    content = message_data.get("content", [])
    if not isinstance(content, list):
        return False
    if not content:
        return False
    return all(
        isinstance(block, dict) and block.get("type") == "tool_result"
        for block in content
    )


def render_message(log_type, message_json, timestamp):
    if not message_json:
        return ""
    try:
        message_data = json.loads(message_json)
    except json.JSONDecodeError:
        return ""
    msg_id = make_msg_id(timestamp)
    if log_type == "user":
        content_html = render_user_message_content(message_data)
        # Check if this is a tool result message
        if is_tool_result_message(message_data):
            role_class, role_label = "tool-reply", "Tool reply"
        else:
            role_class, role_label = "user", "User"
    elif log_type == "assistant":
        content_html = render_assistant_message(message_data, msg_id)
        role_class, role_label = "assistant", "Assistant"
    else:
        return ""
    if not content_html.strip():
        return ""
    return _macros.message(role_class, role_label, msg_id, timestamp, content_html)


def render_logline(entry):
    """Render one tail logline to a complete `.message` HTML fragment.

    Reuses render_message (the same renderer the static pages use). Returns "" for
    entries with no visible content so the caller can skip them. Continuation
    summaries get the same collapsed <details> wrapper as the static output.
    """
    message_json = json.dumps(entry.get("message", {}))
    fragment = render_message(
        entry.get("type"), message_json, entry.get("timestamp", "")
    )
    if not fragment:
        return ""
    if entry.get("isCompactSummary"):
        return _macros.continuation(fragment)
    return fragment


def prompt_preview(text):
    """Collapse whitespace and cap at 100 chars — the canonical prompt preview
    shared by the live TOC, the static index, and the session card, so all
    three render identical previews for the same prompt."""
    preview = " ".join(text.split())
    if len(preview) > 100:
        preview = preview[:97] + "..."
    return preview


def compute_usage_totals(loglines):
    """Token totals for the session info card.

    ``context_tokens``: input + cache_creation + cache_read of the LATEST
    assistant entry carrying ``message.usage`` (≈ the conversation's current
    context size), or None when no usage was recorded (web JSON exports).
    ``output_tokens``: sum across all assistant entries. Absolute numbers
    only — the model's context *limit* is not recorded in session files.
    """
    context = None
    output = 0
    for entry in loglines:
        if entry.get("type") != "assistant":
            continue
        usage = entry.get("message", {}).get("usage")
        if not isinstance(usage, dict):
            continue
        output += usage.get("output_tokens", 0) or 0
        context = (
            (usage.get("input_tokens", 0) or 0)
            + (usage.get("cache_creation_input_tokens", 0) or 0)
            + (usage.get("cache_read_input_tokens", 0) or 0)
        )
    return {"context_tokens": context, "output_tokens": output}


def last_assistant_snippet(loglines, max_length=280):
    """Whitespace-collapsed text of the last assistant text block, or None.

    The session card's recap fallback for sessions with no away-summary yet.
    """
    for entry in reversed(loglines):
        if entry.get("type") != "assistant":
            continue
        text = extract_text_from_content(entry.get("message", {}).get("content", ""))
        if not text:
            continue
        return _truncate(" ".join(text.split()), max_length)
    return None


def build_card_data(
    title,
    prompt_num,
    total_messages,
    total_tool_calls,
    total_commits,
    loglines,
    recap,
    card_prompts,
    latest_link,
):
    """Assemble the session-card payload shared by both static generators.

    Recap precedence: an explicit away-summary recap (source "recap"), else
    the last assistant text snippet (source "assistant") so the card always
    answers "where did this session leave off", else null. ``usage`` comes
    from compute_usage_totals; ``context_tokens`` is null for web JSON
    exports, which the card hides.
    """
    usage = compute_usage_totals(loglines)
    if recap:
        recap_obj = {"text": _truncate(recap, 400), "source": "recap"}
    else:
        snippet = last_assistant_snippet(loglines)
        recap_obj = {"text": snippet, "source": "assistant"} if snippet else None
    return {
        "title": title,
        "stats": {
            "prompts": prompt_num,
            "messages": total_messages,
            "tool_calls": total_tool_calls,
            "commits": total_commits,
        },
        "usage": usage,
        "recap": recap_obj,
        "prompts": card_prompts,
        "latest_link": latest_link,
    }


def index_prompt(entry):
    """If `entry` is a real user prompt (for the live TOC), return (True, preview);
    otherwise (False, "").

    Mirrors generate_html's timeline predicate: a user message with non-empty
    text, excluding continuation summaries and "Stop hook feedback:" prompts.
    """
    if entry.get("type") != "user" or entry.get("isCompactSummary"):
        return False, ""
    text = extract_text_from_content(entry.get("message", {}).get("content", ""))
    if not text or text.startswith("Stop hook feedback:"):
        return False, ""
    return True, prompt_preview(text)


def new_live_stats():
    """Fresh cumulative-counter state for one SSE connection."""
    return {
        "prompts": 0,
        "messages": 0,
        "tool_counts": {},
        "commits": 0,
        "context_tokens": None,
        "output_tokens": 0,
    }


def accumulate_live_stats(state, entry):
    """Fold one *shown* logline entry into the running live counters.

    Counts what the live view displays: every rendered message, its tool_use
    blocks, detected commits, and real user prompts. Reuses analyze_conversation.
    """
    state["messages"] += 1
    delta = analyze_conversation(
        [
            (
                entry.get("type"),
                json.dumps(entry.get("message", {})),
                entry.get("timestamp", ""),
            )
        ]
    )
    for name, count in delta["tool_counts"].items():
        state["tool_counts"][name] = state["tool_counts"].get(name, 0) + count
    state["commits"] += len(delta["commits"])
    is_prompt, _ = index_prompt(entry)
    if is_prompt:
        state["prompts"] += 1
    if entry.get("type") == "assistant":
        usage = entry.get("message", {}).get("usage")
        if isinstance(usage, dict):
            # Same semantics as compute_usage_totals: latest context, summed
            # output.
            state["output_tokens"] += usage.get("output_tokens", 0) or 0
            state["context_tokens"] = (
                (usage.get("input_tokens", 0) or 0)
                + (usage.get("cache_creation_input_tokens", 0) or 0)
                + (usage.get("cache_read_input_tokens", 0) or 0)
            )
    return state


def live_stats_payload(state):
    """Project live-counter state to the SSE wire dict."""
    return {
        "prompts": state["prompts"],
        "messages": state["messages"],
        "tool_calls": sum(state["tool_counts"].values()),
        "commits": state["commits"],
        "context_tokens": state["context_tokens"],
        "output_tokens": state["output_tokens"],
    }


# Extra styles for the live view (status dot, stats bar, table of contents).
# Appended to the shared CSS so the static output's CSS constant stays untouched.
LIVE_CSS = """
.live-header { display: flex; align-items: baseline; flex-wrap: wrap; gap: 12px; }
.live-status { font-size: 0.55rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase; }
.live-status.live { color: #2e7d32; }
.live-status.down { color: var(--text-muted); }
.stats-bar { color: var(--text-muted); margin: 0 0 20px; }
.toc { margin-bottom: 24px; border: 1px solid #e0e0e0; border-radius: 8px; background: var(--card-bg); padding: 8px 12px; }
.toc > summary { cursor: pointer; color: var(--text-muted); font-weight: 600; }
.toc ol { margin: 8px 0 4px; padding-left: 24px; }
.toc li { margin: 2px 0; }
.toc a { color: inherit; text-decoration: none; }
.toc a:hover { text-decoration: underline; }
"""

# Client-side script for the live view. The four DOM "enhancers" are refactored
# from the static JS constant into enhance(root) so they can run on each
# streamed fragment; the EventSource client appends fragments, builds the TOC,
# and updates the stats bar. The static JS constant is intentionally left
# untouched (keeps static-output snapshots stable).
LIVE_JS = r"""
(function () {
  function localizeTimes(root) {
    root.querySelectorAll('time[data-timestamp]').forEach(function (el) {
      var ts = el.getAttribute('data-timestamp');
      var date = new Date(ts);
      if (isNaN(date.getTime())) return;
      var now = new Date();
      var isToday = date.toDateString() === now.toDateString();
      var timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
      el.textContent = isToday ? timeStr : (date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + timeStr);
    });
  }
  function highlightJson(root) {
    root.querySelectorAll('pre.json').forEach(function (el) {
      if (el.dataset.hl) return;
      el.dataset.hl = '1';
      var text = el.textContent;
      text = text.replace(/"([^"]+)":/g, '<span style="color: #ce93d8">"$1"</span>:');
      text = text.replace(/: "([^"]*)"/g, ': <span style="color: #81d4fa">"$1"</span>');
      text = text.replace(/: (\d+)/g, ': <span style="color: #ffcc80">$1</span>');
      text = text.replace(/: (true|false|null)/g, ': <span style="color: #f48fb1">$1</span>');
      el.innerHTML = text;
    });
  }
  function setupTruncation(root) {
    root.querySelectorAll('.truncatable').forEach(function (wrapper) {
      if (wrapper.dataset.trunc) return;
      wrapper.dataset.trunc = '1';
      var content = wrapper.querySelector('.truncatable-content');
      var btn = wrapper.querySelector('.expand-btn');
      if (content && btn && content.scrollHeight > 250) {
        wrapper.classList.add('truncated');
        btn.addEventListener('click', function () {
          if (wrapper.classList.contains('truncated')) { wrapper.classList.remove('truncated'); wrapper.classList.add('expanded'); btn.textContent = 'Show less'; }
          else { wrapper.classList.remove('expanded'); wrapper.classList.add('truncated'); btn.textContent = 'Show more'; }
        });
      }
    });
  }
  function setupCopyButtons(root) {
    // A streamed fragment's root IS the .message element, so include it as well
    // as any descendants (querySelectorAll matches descendants only, not root).
    var msgs = root.matches && root.matches('.message') ? [root] : [];
    root.querySelectorAll('.message').forEach(function (m) { msgs.push(m); });
    msgs.forEach(function (msg) {
      if (msg.dataset.copyBound) return;
      msg.dataset.copyBound = '1';
      var textBtn = msg.querySelector('.copy-text');
      var mdBtn = msg.querySelector('.copy-md');
      if (!textBtn && !mdBtn) return;
      function flash(btn, ok) {
        var cls = ok ? 'copied' : 'failed';
        var original = btn.textContent;
        btn.classList.add(cls);
        btn.textContent = ok ? '✓' : '✗';
        setTimeout(function () { btn.classList.remove(cls); btn.textContent = original; }, 1200);
      }
      function writeClipboard(btn, text) {
        if (!text) { flash(btn, false); return; }
        if (navigator.clipboard && navigator.clipboard.writeText) {
          navigator.clipboard.writeText(text).then(function () { flash(btn, true); }, function () { flash(btn, false); });
          return;
        }
        try {
          var ta = document.createElement('textarea');
          ta.value = text; ta.style.position = 'fixed'; ta.style.left = '-9999px';
          document.body.appendChild(ta); ta.select();
          var ok = document.execCommand('copy'); document.body.removeChild(ta); flash(btn, ok);
        } catch (e) { flash(btn, false); }
      }
      if (textBtn) {
        textBtn.addEventListener('click', function (e) {
          e.preventDefault();
          var body = msg.querySelector('.message-content');
          var text = body ? (body.innerText || body.textContent || '').trim() : '';
          writeClipboard(textBtn, text);
        });
      }
      if (mdBtn) {
        mdBtn.addEventListener('click', function (e) {
          e.preventDefault();
          var blocks = [];
          msg.querySelectorAll('[data-markdown]').forEach(function (el) {
            var src = el.getAttribute('data-markdown');
            if (!src) return;
            if (el.classList.contains('thinking')) {
              var quoted = src.split('\n').map(function (line) { return line.length ? '> ' + line : '>'; }).join('\n');
              blocks.push(quoted);
            } else { blocks.push(src); }
          });
          writeClipboard(mdBtn, blocks.join('\n\n'));
        });
      }
    });
  }
  function enhance(root) {
    localizeTimes(root);
    highlightJson(root);
    setupTruncation(root);
    setupCopyButtons(root);
  }

  var messages = document.getElementById('messages');
  var tocList = document.getElementById('toc-list');
  var statusEl = document.getElementById('live-status');
  // Live mode: no embedded JSON payload — build an empty card and let the
  // SSE handlers below feed it.
  var card = window.sessionCard || null;
  if (card) card.init(null);

  function nearBottom() {
    return (window.innerHeight + window.scrollY) >= (document.body.scrollHeight - 120);
  }
  function setStatus(live) {
    if (!statusEl) return;
    statusEl.textContent = live ? '● live' : '● reconnecting…';
    statusEl.className = 'live-status ' + (live ? 'live' : 'down');
  }
  function setCount(id, n) { var el = document.getElementById(id); if (el) el.textContent = n; }

  var es = new EventSource('/events');
  es.addEventListener('open', function () { setStatus(true); });
  es.addEventListener('error', function () { setStatus(false); });

  es.addEventListener('reset', function () {
    if (messages) messages.innerHTML = '';
    if (tocList) tocList.innerHTML = '';
    setCount('stat-prompts', 0); setCount('stat-messages', 0); setCount('stat-tools', 0); setCount('stat-commits', 0);
    if (card) card.reset();
  });
  es.addEventListener('append', function (e) {
    var payload = JSON.parse(e.data);
    var stick = nearBottom();
    var tpl = document.createElement('template');
    tpl.innerHTML = payload.html;
    var node = tpl.content.firstElementChild;
    if (!node || !messages) return;
    messages.appendChild(node);
    enhance(node);
    // Client-side recap fallback: the latest assistant text stands in until
    // a real away-summary recap arrives (which then pins the section).
    if (card) {
      var ats = node.querySelectorAll('.assistant-text');
      if (ats.length) {
        var txt = (ats[ats.length - 1].innerText || '').replace(/\s+/g, ' ').trim();
        if (txt) card.setRecap(txt.length > 280 ? txt.slice(0, 277) + '...' : txt, false);
      }
    }
    if (stick) window.scrollTo(0, document.body.scrollHeight);
  });
  es.addEventListener('prompt', function (e) {
    var p = JSON.parse(e.data);
    if (card) card.addPrompt(p); // no link on the wire -> in-page '#'+id
    if (!tocList) return;
    var li = document.createElement('li');
    var a = document.createElement('a');
    a.href = '#' + p.id;
    a.textContent = '#' + p.num + '  ' + p.preview; // textContent: preview never parsed as HTML
    li.appendChild(a);
    tocList.appendChild(li);
  });
  es.addEventListener('stats', function (e) {
    var s = JSON.parse(e.data);
    setCount('stat-prompts', s.prompts);
    setCount('stat-messages', s.messages);
    setCount('stat-tools', s.tool_calls);
    setCount('stat-commits', s.commits);
    if (card) {
      card.setStats(s);
      card.setUsage({ context_tokens: s.context_tokens, output_tokens: s.output_tokens });
    }
  });
  es.addEventListener('title', function (e) {
    var t = JSON.parse(e.data).title;
    if (!t) return;
    document.title = t + ' (live)';
    var h = document.getElementById('session-title');
    if (h) h.textContent = t; // textContent: never parsed as HTML
    if (card) card.setTitle(t);
  });
  es.addEventListener('recap', function (e) {
    var r = JSON.parse(e.data);
    if (card && r.text) card.setRecap(r.text, true);
  });
})();
"""


class _LiveServer(ThreadingHTTPServer):
    """Threaded HTTP server that tails one session file. One handler thread per
    connected browser tab; daemon threads so Ctrl-C/shutdown never hangs."""

    daemon_threads = True
    allow_reuse_address = True

    def __init__(self, server_address, handler, session_file, repo, poll_interval):
        super().__init__(server_address, handler)
        self.session_file = Path(session_file)
        self.repo = repo
        self.poll_interval = poll_interval
        self.stop_event = threading.Event()

    def shutdown(self):
        # Signal handler poll-loops to exit first, then stop serve_forever.
        # MUST be called from a different thread than serve_forever().
        self.stop_event.set()
        super().shutdown()


class _LiveHandler(BaseHTTPRequestHandler):
    """Serves the live shell at `/` and a Server-Sent Events tail at `/events`."""

    def log_message(self, format, *args):  # noqa: A002 - keep the CLI output clean
        pass

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/":
            self._serve_shell()
        elif path == "/events":
            self._serve_events()
        else:
            self.send_error(404)

    def _serve_shell(self):
        try:
            # Pre-title the shell so the tab is identifiable before the SSE
            # replay arrives (and for sessions that never rename).
            session_title = get_session_title(self.server.session_file)
        except Exception:
            session_title = None  # file may not exist yet — title arrives live
        body = (
            get_template("live.html")
            .render(
                # CARD_JS before LIVE_JS: the SSE handlers call into
                # window.sessionCard, so the card API must exist first.
                css=CSS + LIVE_CSS + CARD_CSS,
                js=CARD_JS + LIVE_JS,
                session_title=session_title,
            )
            .encode("utf-8")
        )
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _sse_write(self, text):
        self.wfile.write(text.encode("utf-8"))
        self.wfile.flush()

    def _serve_events(self):
        server = self.server
        # render_message -> ... -> commit_card reads this module global for commit
        # links. `watch` bypasses generate_html (where it's normally set), so set
        # it here from the repo resolved once at server construction.
        global _github_repo
        _github_repo = server.repo

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.send_header("Cache-Control", "no-cache")
        self.send_header("Connection", "close")
        self.end_headers()

        stop_event = server.stop_event
        path = server.session_file
        poll = server.poll_interval
        try:
            self._sse_write(format_sse_event("reset", {}))
            offset = 0
            state = new_live_stats()
            last_title = None
            last_recap = None
            while not stop_event.is_set():
                if not path.exists():
                    if stop_event.wait(poll):
                        break
                    continue
                if path.stat().st_size < offset:
                    # File truncated/compacted: re-sync from the top.
                    self._sse_write(format_sse_event("reset", {}))
                    offset = 0
                    state = new_live_stats()
                    last_title = None
                    last_recap = None
                loglines, offset = read_new_loglines(path, offset)
                changed = False
                for entry in loglines:
                    if entry.get("type") == "ai-title":
                        # Deduped: Claude Code re-writes the same title often.
                        if entry["title"] != last_title:
                            last_title = entry["title"]
                            self._sse_write(
                                format_sse_event("title", {"title": last_title})
                            )
                        continue
                    if entry.get("type") == "away-summary":
                        if entry["text"] and entry["text"] != last_recap:
                            last_recap = entry["text"]
                            self._sse_write(
                                format_sse_event("recap", {"text": last_recap})
                            )
                        continue
                    fragment = render_logline(entry)
                    if not fragment:
                        continue
                    self._sse_write(format_sse_event("append", {"html": str(fragment)}))
                    accumulate_live_stats(state, entry)
                    is_prompt, preview = index_prompt(entry)
                    if is_prompt:
                        ts = entry.get("timestamp", "")
                        self._sse_write(
                            format_sse_event(
                                "prompt",
                                {
                                    "id": make_msg_id(ts),
                                    "num": state["prompts"],
                                    "preview": preview,
                                    "timestamp": ts,
                                },
                            )
                        )
                    changed = True
                if changed:
                    self._sse_write(
                        format_sse_event("stats", live_stats_payload(state))
                    )
                self._sse_write(": keepalive\n\n")
                if stop_event.wait(poll):
                    break
        except (BrokenPipeError, ConnectionError, OSError):
            pass  # client disconnected; end this handler thread


def create_live_server(
    session_file, host="127.0.0.1", port=0, repo=None, poll_interval=0.3
):
    """Build (but do not start) a live-tail HTTP server for `session_file`.

    Binds immediately to ``host:port`` (use port 0 for an ephemeral port; read
    the real port from ``server.server_address``). Resolves the GitHub repo once
    for commit links unless one is supplied. Start with ``serve_forever()`` and
    stop with ``shutdown()`` then ``server_close()``. Does NOT open a browser —
    that stays in the command layer so tests never spawn one.
    """
    session_file = Path(session_file)
    if repo is None and session_file.exists():
        try:
            repo = detect_github_repo(
                parse_session_file(session_file).get("loglines", [])
            )
        except Exception:
            repo = None
    return _LiveServer((host, port), _LiveHandler, session_file, repo, poll_interval)


CSS = """
:root { --bg-color: #f5f5f5; --card-bg: #ffffff; --user-bg: #e3f2fd; --user-border: #1976d2; --assistant-bg: #f5f5f5; --assistant-border: #9e9e9e; --thinking-bg: #fff8e1; --thinking-border: #ffc107; --thinking-text: #666; --tool-bg: #f3e5f5; --tool-border: #9c27b0; --tool-result-bg: #e8f5e9; --tool-error-bg: #ffebee; --text-color: #212121; --text-muted: #757575; --code-bg: #263238; --code-text: #aed581; }
* { box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; background: var(--bg-color); color: var(--text-color); margin: 0; padding: 16px; line-height: 1.6; }
.container { max-width: 800px; margin: 0 auto; }
h1 { font-size: 1.5rem; margin-bottom: 24px; padding-bottom: 8px; border-bottom: 2px solid var(--user-border); }
.header-row { display: flex; justify-content: space-between; align-items: center; flex-wrap: wrap; gap: 12px; border-bottom: 2px solid var(--user-border); padding-bottom: 8px; margin-bottom: 24px; }
.header-row h1 { border-bottom: none; padding-bottom: 0; margin-bottom: 0; flex: 1; min-width: 200px; }
.message { margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.message.user { background: var(--user-bg); border-left: 4px solid var(--user-border); }
.message.assistant { background: var(--card-bg); border-left: 4px solid var(--assistant-border); }
.message.tool-reply { background: #fff8e1; border-left: 4px solid #ff9800; }
.tool-reply .role-label { color: #e65100; }
.tool-reply .tool-result { background: transparent; padding: 0; margin: 0; }
.tool-reply .tool-result .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff8e1); }
.message-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(0,0,0,0.03); font-size: 0.85rem; }
.message-header-actions { display: flex; align-items: center; gap: 6px; }
.copy-btn { background: transparent; border: 1px solid transparent; border-radius: 4px; padding: 2px 6px; cursor: pointer; font-size: 0.8rem; line-height: 1; color: var(--text-muted); opacity: 0.55; transition: opacity 0.15s, background 0.15s, color 0.15s; }
.copy-btn:hover { opacity: 1; background: rgba(0,0,0,0.06); color: var(--text-color); }
.copy-btn.copied { background: #c8e6c9; color: #1b5e20; opacity: 1; border-color: #81c784; }
.copy-btn.failed { background: #ffcdd2; color: #b71c1c; opacity: 1; border-color: #ef9a9a; }
.copy-md { font-family: monospace; font-weight: 600; }
.role-label { font-weight: 600; text-transform: uppercase; letter-spacing: 0.5px; }
.user .role-label { color: var(--user-border); }
time { color: var(--text-muted); font-size: 0.8rem; }
.timestamp-link { color: inherit; text-decoration: none; }
.timestamp-link:hover { text-decoration: underline; }
.message:target { animation: highlight 2s ease-out; }
@keyframes highlight { 0% { background-color: rgba(25, 118, 210, 0.2); } 100% { background-color: transparent; } }
.message-content { padding: 16px; }
.message-content p { margin: 0 0 12px 0; }
.message-content p:last-child { margin-bottom: 0; }
.thinking { background: var(--thinking-bg); border: 1px solid var(--thinking-border); border-radius: 8px; padding: 12px; margin: 12px 0; font-size: 0.9rem; color: var(--thinking-text); }
.thinking-label { font-size: 0.75rem; font-weight: 600; text-transform: uppercase; color: #f57c00; margin-bottom: 8px; }
.thinking p { margin: 8px 0; }
.assistant-text { margin: 8px 0; }
.tool-use { background: var(--tool-bg); border: 1px solid var(--tool-border); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-header { font-weight: 600; color: var(--tool-border); margin-bottom: 8px; display: flex; align-items: center; gap: 8px; }
.tool-icon { font-size: 1.1rem; }
.tool-description { font-size: 0.9rem; color: var(--text-muted); margin-bottom: 8px; font-style: italic; }
.tool-result { background: var(--tool-result-bg); border-radius: 8px; padding: 12px; margin: 12px 0; }
.tool-result.tool-error { background: var(--tool-error-bg); }
.file-tool { border-radius: 8px; padding: 12px; margin: 12px 0; }
.write-tool { background: linear-gradient(135deg, #e3f2fd 0%, #e8f5e9 100%); border: 1px solid #4caf50; }
.edit-tool { background: linear-gradient(135deg, #fff3e0 0%, #fce4ec 100%); border: 1px solid #ff9800; }
.file-tool-header { font-weight: 600; margin-bottom: 4px; display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
.write-header { color: #2e7d32; }
.edit-header { color: #e65100; }
.file-tool-icon { font-size: 1rem; }
.file-tool-path { font-family: monospace; background: rgba(0,0,0,0.08); padding: 2px 8px; border-radius: 4px; }
.file-tool-fullpath { font-family: monospace; font-size: 0.8rem; color: var(--text-muted); margin-bottom: 8px; word-break: break-all; }
.file-content { margin: 0; }
.edit-section { display: flex; margin: 4px 0; border-radius: 4px; overflow: hidden; }
.edit-label { padding: 8px 12px; font-weight: bold; font-family: monospace; display: flex; align-items: flex-start; }
.edit-old { background: #fce4ec; }
.edit-old .edit-label { color: #b71c1c; background: #f8bbd9; }
.edit-old .edit-content { color: #880e4f; }
.edit-new { background: #e8f5e9; }
.edit-new .edit-label { color: #1b5e20; background: #a5d6a7; }
.edit-new .edit-content { color: #1b5e20; }
.edit-content { margin: 0; flex: 1; background: transparent; font-size: 0.85rem; }
.edit-replace-all { font-size: 0.75rem; font-weight: normal; color: var(--text-muted); }
.write-tool .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #e6f4ea); }
.edit-tool .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff0e5); }
.todo-list { background: linear-gradient(135deg, #e8f5e9 0%, #f1f8e9 100%); border: 1px solid #81c784; border-radius: 8px; padding: 12px; margin: 12px 0; }
.todo-header { font-weight: 600; color: #2e7d32; margin-bottom: 10px; display: flex; align-items: center; gap: 8px; font-size: 0.95rem; }
.todo-items { list-style: none; margin: 0; padding: 0; }
.todo-item { display: flex; align-items: flex-start; gap: 10px; padding: 6px 0; border-bottom: 1px solid rgba(0,0,0,0.06); font-size: 0.9rem; }
.todo-item:last-child { border-bottom: none; }
.todo-icon { flex-shrink: 0; width: 20px; height: 20px; display: flex; align-items: center; justify-content: center; font-weight: bold; border-radius: 50%; }
.todo-completed .todo-icon { color: #2e7d32; background: rgba(46, 125, 50, 0.15); }
.todo-completed .todo-content { color: #558b2f; text-decoration: line-through; }
.todo-in-progress .todo-icon { color: #f57c00; background: rgba(245, 124, 0, 0.15); }
.todo-in-progress .todo-content { color: #e65100; font-weight: 500; }
.todo-pending .todo-icon { color: #757575; background: rgba(0,0,0,0.05); }
.todo-pending .todo-content { color: #616161; }
pre { background: var(--code-bg); color: var(--code-text); padding: 12px; border-radius: 6px; overflow-x: auto; font-size: 0.85rem; line-height: 1.5; margin: 8px 0; white-space: pre-wrap; word-wrap: break-word; }
pre.json { color: #e0e0e0; }
code { background: rgba(0,0,0,0.08); padding: 2px 6px; border-radius: 4px; font-size: 0.9em; }
pre code { background: none; padding: 0; }
.user-content { margin: 0; }
.truncatable { position: relative; }
.truncatable.truncated .truncatable-content { max-height: 200px; overflow: hidden; }
.truncatable.truncated::after { content: ''; position: absolute; bottom: 32px; left: 0; right: 0; height: 60px; background: linear-gradient(to bottom, transparent, var(--card-bg)); pointer-events: none; }
.message.user .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--user-bg)); }
.message.tool-reply .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, #fff8e1); }
.tool-use .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--tool-bg)); }
.tool-result .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--tool-result-bg)); }
.expand-btn { display: none; width: 100%; padding: 8px 16px; margin-top: 4px; background: rgba(0,0,0,0.05); border: 1px solid rgba(0,0,0,0.1); border-radius: 6px; cursor: pointer; font-size: 0.85rem; color: var(--text-muted); }
.expand-btn:hover { background: rgba(0,0,0,0.1); }
.truncatable.truncated .expand-btn, .truncatable.expanded .expand-btn { display: block; }
.pagination { display: flex; justify-content: center; gap: 8px; margin: 24px 0; flex-wrap: wrap; }
.pagination a, .pagination span { padding: 5px 10px; border-radius: 6px; text-decoration: none; font-size: 0.85rem; }
.pagination a { background: var(--card-bg); color: var(--user-border); border: 1px solid var(--user-border); }
.pagination a:hover { background: var(--user-bg); }
.pagination .current { background: var(--user-border); color: white; }
.pagination .disabled { color: var(--text-muted); border: 1px solid #ddd; }
.pagination .index-link { background: var(--user-border); color: white; }
details.continuation { margin-bottom: 16px; }
details.continuation summary { cursor: pointer; padding: 12px 16px; background: var(--user-bg); border-left: 4px solid var(--user-border); border-radius: 12px; font-weight: 500; color: var(--text-muted); }
details.continuation summary:hover { background: rgba(25, 118, 210, 0.15); }
details.continuation[open] summary { border-radius: 12px 12px 0 0; margin-bottom: 0; }
.index-item { margin-bottom: 16px; border-radius: 12px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); background: var(--user-bg); border-left: 4px solid var(--user-border); }
.index-item a { display: block; text-decoration: none; color: inherit; }
.index-item a:hover { background: rgba(25, 118, 210, 0.1); }
.index-item-header { display: flex; justify-content: space-between; align-items: center; padding: 8px 16px; background: rgba(0,0,0,0.03); font-size: 0.85rem; }
.index-item-number { font-weight: 600; color: var(--user-border); }
.index-item-content { padding: 16px; }
.index-item-stats { padding: 8px 16px 12px 32px; font-size: 0.85rem; color: var(--text-muted); border-top: 1px solid rgba(0,0,0,0.06); }
.index-item-commit { margin-top: 6px; padding: 4px 8px; background: #fff3e0; border-radius: 4px; font-size: 0.85rem; color: #e65100; }
.index-item-commit code { background: rgba(0,0,0,0.08); padding: 1px 4px; border-radius: 3px; font-size: 0.8rem; margin-right: 6px; }
.commit-card { margin: 8px 0; padding: 10px 14px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 6px; }
.commit-card a { text-decoration: none; color: #5d4037; display: block; }
.commit-card a:hover { color: #e65100; }
.commit-card-hash { font-family: monospace; color: #e65100; font-weight: 600; margin-right: 8px; }
.index-commit { margin-bottom: 12px; padding: 10px 16px; background: #fff3e0; border-left: 4px solid #ff9800; border-radius: 8px; box-shadow: 0 1px 2px rgba(0,0,0,0.05); }
.index-commit a { display: block; text-decoration: none; color: inherit; }
.index-commit a:hover { background: rgba(255, 152, 0, 0.1); margin: -10px -16px; padding: 10px 16px; border-radius: 8px; }
.index-commit-header { display: flex; justify-content: space-between; align-items: center; font-size: 0.85rem; margin-bottom: 4px; }
.index-commit-hash { font-family: monospace; color: #e65100; font-weight: 600; }
.index-commit-msg { color: #5d4037; }
.index-item-long-text { margin-top: 8px; padding: 12px; background: var(--card-bg); border-radius: 8px; border-left: 3px solid var(--assistant-border); }
.index-item-long-text .truncatable.truncated::after { background: linear-gradient(to bottom, transparent, var(--card-bg)); }
.index-item-long-text-content { color: var(--text-color); }
#search-box { display: none; align-items: center; gap: 8px; }
#search-box input { padding: 6px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; width: 180px; }
#search-box button, #modal-search-btn, #modal-close-btn { background: var(--user-border); color: white; border: none; border-radius: 6px; padding: 6px 10px; cursor: pointer; display: flex; align-items: center; justify-content: center; }
#search-box button:hover, #modal-search-btn:hover { background: #1565c0; }
#modal-close-btn { background: var(--text-muted); margin-left: 8px; }
#modal-close-btn:hover { background: #616161; }
#search-modal[open] { border: none; border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.2); padding: 0; width: 90vw; max-width: 900px; height: 80vh; max-height: 80vh; display: flex; flex-direction: column; }
#search-modal::backdrop { background: rgba(0,0,0,0.5); }
.search-modal-header { display: flex; align-items: center; gap: 8px; padding: 16px; border-bottom: 1px solid var(--assistant-border); background: var(--bg-color); border-radius: 12px 12px 0 0; }
.search-modal-header input { flex: 1; padding: 8px 12px; border: 1px solid var(--assistant-border); border-radius: 6px; font-size: 16px; }
#search-status { padding: 8px 16px; font-size: 0.85rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
#search-results { flex: 1; overflow-y: auto; padding: 16px; }
.search-result { margin-bottom: 16px; border-radius: 8px; overflow: hidden; box-shadow: 0 1px 3px rgba(0,0,0,0.1); }
.search-result a { display: block; text-decoration: none; color: inherit; }
.search-result a:hover { background: rgba(25, 118, 210, 0.05); }
.search-result-page { padding: 6px 12px; background: rgba(0,0,0,0.03); font-size: 0.8rem; color: var(--text-muted); border-bottom: 1px solid rgba(0,0,0,0.06); }
.search-result-content { padding: 12px; }
.search-result mark { background: #fff59d; padding: 1px 2px; border-radius: 2px; }
@media (max-width: 600px) { body { padding: 8px; } .message, .index-item { border-radius: 8px; } .message-content, .index-item-content { padding: 12px; } pre { font-size: 0.8rem; padding: 8px; } #search-box input { width: 120px; } #search-modal[open] { width: 95vw; height: 90vh; } }
"""

JS = """
document.querySelectorAll('time[data-timestamp]').forEach(function(el) {
    const timestamp = el.getAttribute('data-timestamp');
    const date = new Date(timestamp);
    const now = new Date();
    const isToday = date.toDateString() === now.toDateString();
    const timeStr = date.toLocaleTimeString(undefined, { hour: '2-digit', minute: '2-digit' });
    if (isToday) { el.textContent = timeStr; }
    else { el.textContent = date.toLocaleDateString(undefined, { month: 'short', day: 'numeric' }) + ' ' + timeStr; }
});
document.querySelectorAll('pre.json').forEach(function(el) {
    let text = el.textContent;
    text = text.replace(/"([^"]+)":/g, '<span style="color: #ce93d8">"$1"</span>:');
    text = text.replace(/: "([^"]*)"/g, ': <span style="color: #81d4fa">"$1"</span>');
    text = text.replace(/: (\\d+)/g, ': <span style="color: #ffcc80">$1</span>');
    text = text.replace(/: (true|false|null)/g, ': <span style="color: #f48fb1">$1</span>');
    el.innerHTML = text;
});
document.querySelectorAll('.truncatable').forEach(function(wrapper) {
    const content = wrapper.querySelector('.truncatable-content');
    const btn = wrapper.querySelector('.expand-btn');
    if (content.scrollHeight > 250) {
        wrapper.classList.add('truncated');
        btn.addEventListener('click', function() {
            if (wrapper.classList.contains('truncated')) { wrapper.classList.remove('truncated'); wrapper.classList.add('expanded'); btn.textContent = 'Show less'; }
            else { wrapper.classList.remove('expanded'); wrapper.classList.add('truncated'); btn.textContent = 'Show more'; }
        });
    }
});
document.querySelectorAll('.message').forEach(function(msg) {
    const textBtn = msg.querySelector('.copy-text');
    const mdBtn = msg.querySelector('.copy-md');
    if (!textBtn && !mdBtn) return;
    function flash(btn, ok) {
        const cls = ok ? 'copied' : 'failed';
        const original = btn.textContent;
        btn.classList.add(cls);
        btn.textContent = ok ? '\\u2713' : '\\u2717';
        setTimeout(function() {
            btn.classList.remove(cls);
            btn.textContent = original;
        }, 1200);
    }
    function writeClipboard(btn, text) {
        if (!text) { flash(btn, false); return; }
        if (navigator.clipboard && navigator.clipboard.writeText) {
            navigator.clipboard.writeText(text).then(
                function() { flash(btn, true); },
                function() { flash(btn, false); }
            );
            return;
        }
        try {
            const ta = document.createElement('textarea');
            ta.value = text;
            ta.style.position = 'fixed';
            ta.style.left = '-9999px';
            document.body.appendChild(ta);
            ta.select();
            const ok = document.execCommand('copy');
            document.body.removeChild(ta);
            flash(btn, ok);
        } catch (e) {
            flash(btn, false);
        }
    }
    if (textBtn) {
        textBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const body = msg.querySelector('.message-content');
            const text = body ? (body.innerText || body.textContent || '').trim() : '';
            writeClipboard(textBtn, text);
        });
    }
    if (mdBtn) {
        mdBtn.addEventListener('click', function(e) {
            e.preventDefault();
            const blocks = [];
            msg.querySelectorAll('[data-markdown]').forEach(function(el) {
                const src = el.getAttribute('data-markdown');
                if (!src) return;
                if (el.classList.contains('thinking')) {
                    const quoted = src.split('\\n').map(function(line) {
                        return line.length ? '> ' + line : '>';
                    }).join('\\n');
                    blocks.push(quoted);
                } else {
                    blocks.push(src);
                }
            });
            writeClipboard(mdBtn, blocks.join('\\n\\n'));
        });
    }
});
"""

# Floating session info card. One implementation, two data feeds: static
# pages embed a JSON payload (session_card.html) that auto-inits the card;
# the live view drives the same API from SSE events. All text lands via
# textContent, so neither feed needs an HTML-escaping path.
CARD_CSS = """
#session-card { position: fixed; right: 16px; bottom: 16px; z-index: 1000; font-size: 0.85rem; }
#session-card .card-pill { display: flex; align-items: center; gap: 8px; background: var(--user-border); color: white; border: none; border-radius: 999px; padding: 8px 14px; cursor: pointer; box-shadow: 0 2px 8px rgba(0,0,0,0.25); font-size: 0.85rem; }
#session-card .card-pill:hover { background: #1565c0; }
#session-card .card-panel { display: none; width: 340px; max-width: calc(100vw - 32px); max-height: 70vh; overflow-y: auto; background: var(--card-bg); border: 1px solid var(--assistant-border); border-radius: 12px; box-shadow: 0 4px 24px rgba(0,0,0,0.25); padding: 12px 14px; }
#session-card.card-open .card-pill { display: none; }
#session-card.card-open .card-panel { display: block; }
#session-card .card-head { display: flex; align-items: flex-start; justify-content: space-between; gap: 8px; margin-bottom: 8px; }
#session-card .card-title { font-weight: 600; line-height: 1.3; }
#session-card .card-close { background: transparent; border: none; cursor: pointer; font-size: 1.1rem; color: var(--text-muted); padding: 0 2px; line-height: 1; }
#session-card .card-close:hover { color: var(--text-color); }
#session-card .card-stats { color: var(--text-muted); margin-bottom: 6px; }
#session-card .card-usage { color: var(--text-muted); margin-bottom: 8px; }
#session-card .card-section-label { font-size: 0.7rem; font-weight: 600; text-transform: uppercase; letter-spacing: 0.04em; color: var(--text-muted); margin: 8px 0 4px; }
#session-card .card-recap-text { background: var(--thinking-bg); border-left: 3px solid var(--thinking-border); border-radius: 6px; padding: 8px 10px; }
#session-card .card-prompt-list { margin: 0; padding-left: 22px; max-height: 32vh; overflow-y: auto; }
#session-card .card-prompt-list li { margin: 2px 0; }
#session-card .card-prompt-list a { color: inherit; text-decoration: none; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; display: block; max-width: 100%; }
#session-card .card-prompt-list a:hover { text-decoration: underline; }
#session-card .card-nav { display: flex; gap: 8px; margin-top: 10px; }
#session-card .card-nav-btn { flex: 1; text-align: center; background: var(--user-bg); color: var(--user-border); border: 1px solid var(--user-border); border-radius: 6px; padding: 6px 8px; text-decoration: none; cursor: pointer; }
#session-card .card-nav-btn:hover { background: rgba(25, 118, 210, 0.15); }
@media print { #session-card { display: none; } }
"""

CARD_JS = r"""
(function () {
  var STORAGE_KEY = 'cct-card-expanded';
  var root = null;
  var refs = {};
  var statsState = { prompts: 0, messages: 0, tool_calls: 0, commits: 0 };
  var lastCtx = null;
  var recapPinned = false; // a real recap beats assistant-text fallbacks

  function el(tag, cls, text) {
    var e = document.createElement(tag);
    if (cls) e.className = cls;
    if (text != null) e.textContent = text;
    return e;
  }
  function formatTokens(n) {
    if (n == null || isNaN(n)) return null;
    return n >= 1000 ? Math.round(n / 1000) + 'k' : String(n);
  }
  function isExpanded() {
    try { return localStorage.getItem(STORAGE_KEY) === '1'; } catch (e) { return false; }
  }
  function setExpanded(on) {
    try { localStorage.setItem(STORAGE_KEY, on ? '1' : '0'); } catch (e) {}
    if (root) root.classList.toggle('card-open', !!on);
  }

  function build() {
    root = document.getElementById('session-card');
    if (!root) return false;
    root.innerHTML = '';

    var pill = el('button', 'card-pill');
    pill.type = 'button';
    pill.setAttribute('aria-label', 'Session info');
    refs.pillStats = el('span', 'card-pill-stats', '…');
    pill.appendChild(refs.pillStats);
    pill.appendChild(el('span', 'card-pill-icon', 'ⓘ'));
    pill.addEventListener('click', function () { setExpanded(true); });

    var panel = el('div', 'card-panel');
    var head = el('div', 'card-head');
    var h1 = document.getElementById('session-title');
    refs.title = el('div', 'card-title', h1 ? h1.textContent : '');
    var close = el('button', 'card-close', '×');
    close.type = 'button';
    close.setAttribute('aria-label', 'Collapse session info');
    close.addEventListener('click', function () { setExpanded(false); });
    head.appendChild(refs.title);
    head.appendChild(close);
    panel.appendChild(head);

    refs.stats = el('div', 'card-stats', '');
    panel.appendChild(refs.stats);
    refs.usage = el('div', 'card-usage', '');
    refs.usage.hidden = true;
    panel.appendChild(refs.usage);

    refs.recapWrap = el('div', 'card-recap');
    refs.recapWrap.hidden = true;
    refs.recapLabel = el('div', 'card-section-label', 'Recap');
    refs.recapText = el('div', 'card-recap-text', '');
    refs.recapWrap.appendChild(refs.recapLabel);
    refs.recapWrap.appendChild(refs.recapText);
    panel.appendChild(refs.recapWrap);

    var promptsWrap = el('div', 'card-prompts');
    promptsWrap.appendChild(el('div', 'card-section-label', 'Prompts'));
    refs.promptList = el('ol', 'card-prompt-list');
    promptsWrap.appendChild(refs.promptList);
    panel.appendChild(promptsWrap);

    var nav = el('div', 'card-nav');
    var topBtn = el('a', 'card-nav-btn', '▲ top');
    topBtn.href = '#';
    topBtn.addEventListener('click', function (e) {
      e.preventDefault();
      window.scrollTo(0, 0);
    });
    refs.latestBtn = el('a', 'card-nav-btn', '⤓ latest');
    refs.latestBtn.href = '#';
    refs.latestBtn.addEventListener('click', function (e) {
      if (refs.latestBtn.dataset.href) return; // static: real link navigates
      e.preventDefault();
      window.scrollTo(0, document.body.scrollHeight); // live: jump to tail
    });
    nav.appendChild(topBtn);
    nav.appendChild(refs.latestBtn);
    panel.appendChild(nav);

    root.appendChild(pill);
    root.appendChild(panel);
    root.hidden = false;
    if (isExpanded()) root.classList.add('card-open');
    return true;
  }

  function updatePill() {
    if (!refs.pillStats) return;
    var bits = [statsState.prompts + 'p'];
    if (lastCtx) bits.push('ctx ' + lastCtx);
    refs.pillStats.textContent = bits.join(' · ');
  }
  function setTitle(t) {
    if (t && refs.title) refs.title.textContent = t;
  }
  function setStats(s) {
    if (!s) return;
    statsState = s;
    if (refs.stats) {
      refs.stats.textContent = s.prompts + ' prompts · ' + s.messages +
        ' messages · ' + s.tool_calls + ' tools · ' + s.commits + ' commits';
    }
    updatePill();
  }
  function setUsage(u) {
    if (!refs.usage) return;
    var ctx = u ? formatTokens(u.context_tokens) : null;
    var out = u ? formatTokens(u.output_tokens) : null;
    lastCtx = ctx;
    if (!ctx && !out) { refs.usage.hidden = true; updatePill(); return; }
    var parts = [];
    if (ctx) parts.push('Context ' + ctx);
    if (out) parts.push('Output ' + out);
    refs.usage.textContent = parts.join(' · ');
    refs.usage.hidden = false;
    updatePill();
  }
  function setRecap(text, isReal) {
    if (!refs.recapWrap || !text) return;
    if (recapPinned && !isReal) return;
    recapPinned = recapPinned || !!isReal;
    refs.recapLabel.textContent = isReal ? 'Recap' : 'Latest reply';
    refs.recapText.textContent = text;
    refs.recapWrap.hidden = false;
  }
  function addPrompt(p) {
    if (!refs.promptList || !p) return;
    var li = el('li');
    var a = el('a', null, '#' + p.num + '  ' + p.preview);
    a.href = p.link || ('#' + p.id);
    li.appendChild(a);
    refs.promptList.appendChild(li);
  }
  function setPrompts(list) {
    if (!refs.promptList) return;
    refs.promptList.innerHTML = '';
    (list || []).forEach(addPrompt);
  }
  function setLatestLink(href) {
    if (!refs.latestBtn || !href) return;
    refs.latestBtn.href = href;
    refs.latestBtn.dataset.href = href;
  }
  function reset() {
    recapPinned = false;
    lastCtx = null;
    setStats({ prompts: 0, messages: 0, tool_calls: 0, commits: 0 });
    if (refs.usage) refs.usage.hidden = true;
    if (refs.recapWrap) refs.recapWrap.hidden = true;
    if (refs.promptList) refs.promptList.innerHTML = '';
  }
  function init(data) {
    if (!build()) return;
    reset();
    if (!data) return;
    setTitle(data.title);
    setStats(data.stats);
    setUsage(data.usage);
    if (data.recap && data.recap.text) {
      setRecap(data.recap.text, data.recap.source === 'recap');
    }
    setPrompts(data.prompts);
    setLatestLink(data.latest_link);
  }

  window.sessionCard = {
    init: init, reset: reset, setTitle: setTitle, setStats: setStats,
    setUsage: setUsage, setRecap: setRecap, addPrompt: addPrompt,
    setPrompts: setPrompts, setLatestLink: setLatestLink
  };

  // Static pages: auto-init from the embedded JSON payload.
  var dataEl = document.getElementById('session-card-data');
  if (dataEl) {
    try { init(JSON.parse(dataEl.textContent)); } catch (e) {}
  }
})();
"""

# JavaScript to fix relative URLs when served via gisthost.github.io or gistpreview.github.io
# Fixes issue #26: Pagination links broken on gisthost.github.io
GIST_PREVIEW_JS = r"""
(function() {
    var hostname = window.location.hostname;
    if (hostname !== 'gisthost.github.io' && hostname !== 'gistpreview.github.io') return;
    // URL format: https://gisthost.github.io/?GIST_ID/filename.html
    var match = window.location.search.match(/^\?([^/]+)/);
    if (!match) return;
    var gistId = match[1];

    function rewriteLinks(root) {
        (root || document).querySelectorAll('a[href]').forEach(function(link) {
            var href = link.getAttribute('href');
            // Skip already-rewritten links (issue #26 fix)
            if (href.startsWith('?')) return;
            // Skip external links and anchors
            if (href.startsWith('http') || href.startsWith('#') || href.startsWith('//')) return;
            // Handle anchor in relative URL (e.g., page-001.html#msg-123)
            var parts = href.split('#');
            var filename = parts[0];
            var anchor = parts.length > 1 ? '#' + parts[1] : '';
            link.setAttribute('href', '?' + gistId + '/' + filename + anchor);
        });
    }

    // Run immediately
    rewriteLinks();

    // Also run on DOMContentLoaded in case DOM isn't ready yet
    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', function() { rewriteLinks(); });
    }

    // Use MutationObserver to catch dynamically added content
    // gistpreview.github.io may add content after initial load
    var observer = new MutationObserver(function(mutations) {
        mutations.forEach(function(mutation) {
            mutation.addedNodes.forEach(function(node) {
                if (node.nodeType === 1) { // Element node
                    rewriteLinks(node);
                    // Also check if the node itself is a link
                    if (node.tagName === 'A' && node.getAttribute('href')) {
                        var href = node.getAttribute('href');
                        if (!href.startsWith('?') && !href.startsWith('http') &&
                            !href.startsWith('#') && !href.startsWith('//')) {
                            var parts = href.split('#');
                            var filename = parts[0];
                            var anchor = parts.length > 1 ? '#' + parts[1] : '';
                            node.setAttribute('href', '?' + gistId + '/' + filename + anchor);
                        }
                    }
                }
            });
        });
    });

    // Start observing once body exists
    function startObserving() {
        if (document.body) {
            observer.observe(document.body, { childList: true, subtree: true });
        } else {
            setTimeout(startObserving, 10);
        }
    }
    startObserving();

    // Handle fragment navigation after dynamic content loads
    // gisthost.github.io/gistpreview.github.io loads content dynamically, so the browser's
    // native fragment navigation fails because the element doesn't exist yet
    function scrollToFragment() {
        var hash = window.location.hash;
        if (!hash) return false;
        var targetId = hash.substring(1);
        var target = document.getElementById(targetId);
        if (target) {
            target.scrollIntoView({ behavior: 'smooth', block: 'start' });
            return true;
        }
        return false;
    }

    // Try immediately in case content is already loaded
    if (!scrollToFragment()) {
        // Retry with increasing delays to handle dynamic content loading
        var delays = [100, 300, 500, 1000, 2000];
        delays.forEach(function(delay) {
            setTimeout(scrollToFragment, delay);
        });
    }
})();
"""


def inject_gist_preview_js(output_dir):
    """Inject gist preview JavaScript into all HTML files in the output directory."""
    output_dir = Path(output_dir)
    for html_file in output_dir.glob("*.html"):
        content = html_file.read_text(encoding="utf-8")
        # Insert the gist preview JS before the closing </body> tag
        if "</body>" in content:
            content = content.replace(
                "</body>", f"<script>{GIST_PREVIEW_JS}</script>\n</body>"
            )
            html_file.write_text(content, encoding="utf-8")


def create_gist(output_dir, public=False):
    """Create a GitHub gist from the HTML files in output_dir.

    Returns the gist ID on success, or raises click.ClickException on failure.
    """
    output_dir = Path(output_dir)
    html_files = list(output_dir.glob("*.html"))
    if not html_files:
        raise click.ClickException("No HTML files found to upload to gist.")

    # Build the gh gist create command
    # gh gist create file1 file2 ... --public/--private
    cmd = ["gh", "gist", "create"]
    cmd.extend(str(f) for f in sorted(html_files))
    if public:
        cmd.append("--public")

    try:
        result = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=True,
        )
        # Output is the gist URL, e.g., https://gist.github.com/username/GIST_ID
        gist_url = result.stdout.strip()
        # Extract gist ID from URL
        gist_id = gist_url.rstrip("/").split("/")[-1]
        return gist_id, gist_url
    except subprocess.CalledProcessError as e:
        error_msg = e.stderr.strip() if e.stderr else str(e)
        raise click.ClickException(f"Failed to create gist: {error_msg}")
    except FileNotFoundError:
        raise click.ClickException(
            "gh CLI not found. Install it from https://cli.github.com/ and run 'gh auth login'."
        )


def generate_pagination_html(current_page, total_pages):
    return _macros.pagination(current_page, total_pages)


def generate_index_pagination_html(total_pages):
    """Generate pagination for index page where Index is current (first page)."""
    return _macros.index_pagination(total_pages)


def _build_conversations(loglines):
    """Group loglines into prompt-led conversations.

    Each conversation starts at a user message with visible text and carries
    every following entry until the next one. Shared by both generators.
    """
    conversations = []
    current_conv = None
    for entry in loglines:
        log_type = entry.get("type")
        timestamp = entry.get("timestamp", "")
        is_compact_summary = entry.get("isCompactSummary", False)
        message_data = entry.get("message", {})
        if not message_data:
            continue
        # Convert message dict to JSON string for compatibility with existing render functions
        message_json = json.dumps(message_data)
        is_user_prompt = False
        user_text = None
        if log_type == "user":
            content = message_data.get("content", "")
            text = extract_text_from_content(content)
            if text:
                is_user_prompt = True
                user_text = text
        if is_user_prompt:
            if current_conv:
                conversations.append(current_conv)
            current_conv = {
                "user_text": user_text,
                "timestamp": timestamp,
                "messages": [(log_type, message_json, timestamp)],
                "is_continuation": bool(is_compact_summary),
            }
        elif current_conv:
            current_conv["messages"].append((log_type, message_json, timestamp))
    if current_conv:
        conversations.append(current_conv)
    return conversations


def _render_session_pages(loglines, output_dir, title, recap, echo=print):
    """Shared session-page core: stats, timeline, card payload, pages + index.

    Callers resolve title/recap and set the _github_repo global first; `echo`
    is print (file path) or click.echo (web path) for progress lines.
    """
    conversations = _build_conversations(loglines)

    total_convs = len(conversations)
    total_pages = (total_convs + PROMPTS_PER_PAGE - 1) // PROMPTS_PER_PAGE

    # Stats, commits, and the prompt timeline are computed BEFORE page
    # rendering so the session-card payload can be embedded in every page.
    total_tool_counts = {}
    total_messages = 0
    all_commits = []  # (timestamp, hash, message, page_num, conv_index)
    for i, conv in enumerate(conversations):
        total_messages += len(conv["messages"])
        stats = analyze_conversation(conv["messages"])
        for tool, count in stats["tool_counts"].items():
            total_tool_counts[tool] = total_tool_counts.get(tool, 0) + count
        page_num = (i // PROMPTS_PER_PAGE) + 1
        for commit_hash, commit_msg, commit_ts in stats["commits"]:
            all_commits.append((commit_ts, commit_hash, commit_msg, page_num, i))
    total_tool_calls = sum(total_tool_counts.values())
    total_commits = len(all_commits)

    # Build timeline items: prompts and commits merged by timestamp
    timeline_items = []
    card_prompts = []

    # Add prompts
    prompt_num = 0
    for i, conv in enumerate(conversations):
        if conv.get("is_continuation"):
            continue
        if conv["user_text"].startswith("Stop hook feedback:"):
            continue
        prompt_num += 1
        page_num = (i // PROMPTS_PER_PAGE) + 1
        msg_id = make_msg_id(conv["timestamp"])
        link = f"page-{page_num:03d}.html#{msg_id}"
        rendered_content = render_markdown_text(conv["user_text"])
        card_prompts.append(
            {
                "num": prompt_num,
                "id": msg_id,
                "link": link,
                "preview": prompt_preview(conv["user_text"]),
                "timestamp": conv["timestamp"],
            }
        )

        # Collect all messages including from subsequent continuation conversations
        # This ensures long_texts from continuations appear with the original prompt
        all_messages = list(conv["messages"])
        for j in range(i + 1, len(conversations)):
            if not conversations[j].get("is_continuation"):
                break
            all_messages.extend(conversations[j]["messages"])

        # Analyze conversation for stats (excluding commits from inline display now)
        stats = analyze_conversation(all_messages)
        tool_stats_str = format_tool_stats(stats["tool_counts"])

        long_texts_html = ""
        for lt in stats["long_texts"]:
            rendered_lt = render_markdown_text(lt)
            long_texts_html += _macros.index_long_text(rendered_lt)

        stats_html = _macros.index_stats(tool_stats_str, long_texts_html)

        item_html = _macros.index_item(
            prompt_num, link, conv["timestamp"], rendered_content, stats_html
        )
        timeline_items.append((conv["timestamp"], "prompt", item_html))

    # Add commits as separate timeline items
    for commit_ts, commit_hash, commit_msg, page_num, conv_idx in all_commits:
        item_html = _macros.index_commit(
            commit_hash, commit_msg, commit_ts, _github_repo
        )
        timeline_items.append((commit_ts, "commit", item_html))

    # Sort by timestamp
    timeline_items.sort(key=lambda x: x[0])
    index_items = [item[2] for item in timeline_items]

    # Jump-to-latest targets the last message on the last page.
    if conversations:
        last_ts = conversations[-1]["messages"][-1][2]
        latest_page = (len(conversations) - 1) // PROMPTS_PER_PAGE + 1
        latest_link = f"page-{latest_page:03d}.html#{make_msg_id(last_ts)}"
    else:
        latest_link = None

    card_data = build_card_data(
        title,
        prompt_num,
        total_messages,
        total_tool_calls,
        total_commits,
        loglines,
        recap,
        card_prompts,
        latest_link,
    )
    # "</" must not appear raw inside a <script> block; "<\/" is the
    # equivalent JSON escape, preventing </script> breakout.
    card_json = json.dumps(card_data).replace("</", "<\\/")

    for page_num in range(1, total_pages + 1):
        start_idx = (page_num - 1) * PROMPTS_PER_PAGE
        end_idx = min(start_idx + PROMPTS_PER_PAGE, total_convs)
        page_convs = conversations[start_idx:end_idx]
        messages_html = []
        for conv in page_convs:
            is_first = True
            for log_type, message_json, timestamp in conv["messages"]:
                msg_html = render_message(log_type, message_json, timestamp)
                if msg_html:
                    # Wrap continuation summaries in collapsed details
                    if is_first and conv.get("is_continuation"):
                        msg_html = f'<details class="continuation"><summary>Session continuation summary</summary>{msg_html}</details>'
                    messages_html.append(msg_html)
                is_first = False
        pagination_html = generate_pagination_html(page_num, total_pages)
        page_template = get_template("page.html")
        page_content = page_template.render(
            css=CSS + CARD_CSS,
            js=JS + CARD_JS,
            session_title=title,
            card_json=card_json,
            page_num=page_num,
            total_pages=total_pages,
            pagination_html=pagination_html,
            messages_html="".join(messages_html),
        )
        (output_dir / f"page-{page_num:03d}.html").write_text(
            page_content, encoding="utf-8"
        )
        echo(f"Generated page-{page_num:03d}.html")

    index_pagination = generate_index_pagination_html(total_pages)
    index_template = get_template("index.html")
    index_content = index_template.render(
        css=CSS + CARD_CSS,
        js=JS + CARD_JS,
        session_title=title,
        card_json=card_json,
        pagination_html=index_pagination,
        prompt_num=prompt_num,
        total_messages=total_messages,
        total_tool_calls=total_tool_calls,
        total_commits=total_commits,
        total_pages=total_pages,
        index_items_html="".join(index_items),
    )
    index_path = output_dir / "index.html"
    index_path.write_text(index_content, encoding="utf-8")
    echo(
        f"Generated {index_path.resolve()} ({total_convs} prompts, {total_pages} pages)"
    )


def generate_html(json_path, output_dir, github_repo=None, title=None, recap=None):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)

    if title is None:
        # CLI path: one scan yields both the name and the recap. Batch callers
        # (generate_batch_html) pass both in from the scan they already did.
        json_path_p = Path(json_path)
        if json_path_p.suffix == ".jsonl":
            meta = scan_session_metadata(json_path_p, max_length=80)
            if meta.title and meta.title != "(no summary)":
                title = meta.title
            if recap is None:
                recap = meta.recap
        else:
            title = get_session_title(json_path)

    # Load session file (supports both JSON and JSONL)
    data = parse_session_file(json_path)

    loglines = data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        github_repo = detect_github_repo(loglines)
        if github_repo:
            print(f"Auto-detected GitHub repo: {github_repo}")
        else:
            print(
                "Warning: Could not auto-detect GitHub repo. Commit links will be disabled."
            )

    # Set module-level variable for render functions
    global _github_repo
    _github_repo = github_repo

    _render_session_pages(loglines, output_dir, title, recap, echo=print)


@click.group(cls=DefaultGroup, default="local", default_if_no_args=True)
@click.version_option(None, "-v", "--version", package_name="claude-code-transcripts")
def cli():
    """Convert Claude Code session JSON to mobile-friendly HTML pages."""
    pass


@cli.command("local")
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSONL session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
@click.option(
    "--limit",
    default=10,
    help="Maximum number of sessions to show (default: 10)",
)
def local_cmd(output, output_auto, repo, gist, include_json, open_browser, limit):
    """Select and convert a local Claude Code session to HTML."""
    projects_folder = Path.home() / ".claude" / "projects"

    if not projects_folder.exists():
        click.echo(f"Projects folder not found: {projects_folder}")
        click.echo("No local Claude Code sessions available.")
        return

    click.echo("Loading local sessions...")
    choices = build_session_choices(projects_folder, limit=limit)

    if not choices:
        click.echo("No local sessions found.")
        return

    selected = questionary.select(
        "Select a session to convert:",
        choices=choices,
    ).ask()

    if selected is None:
        click.echo("No session selected.")
        return

    session_file = selected

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / session_file.stem
    elif output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_file.stem}"

    output = Path(output)
    generate_html(session_file, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSONL file to output directory if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / session_file.name
        shutil.copy(session_file, json_dest)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSONL: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


@cli.command("watch")
@click.option(
    "--session",
    type=click.Path(),
    help="Tail a specific session file instead of the newest.",
)
@click.option(
    "--pick",
    is_flag=True,
    help="Choose the session from a list instead of auto-selecting the newest.",
)
@click.option(
    "--limit",
    default=10,
    help="Maximum sessions to show with --pick (default: 10).",
)
@click.option(
    "-s",
    "--source",
    type=click.Path(),
    help="Projects folder to search (default: ~/.claude/projects).",
)
@click.option(
    "--port",
    default=0,
    help="Port to serve on (default: an OS-assigned free port).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected if omitted.",
)
@click.option(
    "--open/--no-open",
    "open_browser",
    default=True,
    help="Open the live view in your browser (default: yes).",
)
@click.option(
    "--poll-interval",
    default=0.3,
    help="Seconds between file polls (default: 0.3).",
)
def watch_cmd(session, pick, limit, source, port, repo, open_browser, poll_interval):
    """Tail an active Claude Code session live in your browser.

    Starts a local server and streams the session to the browser as Claude
    writes it. With no options it tails the most-recently-modified session.
    """
    projects_folder = Path(source) if source else (Path.home() / ".claude" / "projects")

    if pick and not session:
        # Same rows as the `local` picker (branch/project/command columns).
        choices = build_session_choices(projects_folder, limit=limit)
        if not choices:
            click.echo("No local sessions found.")
            return
        session_file = questionary.select(
            "Select a session to watch:", choices=choices
        ).ask()
        if session_file is None:
            click.echo("No session selected.")
            return
    else:
        session_file = resolve_active_session(projects_folder, session=session)

    if session_file is None:
        click.echo("No active session found to watch.")
        return
    session_file = Path(session_file)
    if not session_file.exists():
        click.echo(f"Session file not found: {session_file}")
        return

    server = create_live_server(
        session_file, port=port, repo=repo, poll_interval=poll_interval
    )
    url = f"http://127.0.0.1:{server.server_address[1]}/"
    click.echo(f"Watching {session_file}")
    click.echo(f"Live at {url}  (press Ctrl-C to stop)")
    if open_browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        click.echo("\nStopping…")
    finally:
        server.stop_event.set()
        server.server_close()


def is_url(path):
    """Check if a path is a URL (starts with http:// or https://)."""
    return path.startswith("http://") or path.startswith("https://")


def fetch_url_to_tempfile(url):
    """Fetch a URL and save to a temporary file.

    Returns the Path to the temporary file.
    Raises click.ClickException on network errors.
    """
    try:
        response = httpx.get(url, timeout=60.0, follow_redirects=True)
        response.raise_for_status()
    except httpx.RequestError as e:
        raise click.ClickException(f"Failed to fetch URL: {e}")
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"Failed to fetch URL: {e.response.status_code} {e.response.reason_phrase}"
        )

    # Determine file extension from URL
    url_path = url.split("?")[0]  # Remove query params
    if url_path.endswith(".jsonl"):
        suffix = ".jsonl"
    elif url_path.endswith(".json"):
        suffix = ".json"
    else:
        suffix = ".jsonl"  # Default to JSONL

    # Extract a name from the URL for the temp file
    url_name = Path(url_path).stem or "session"

    temp_dir = Path(tempfile.gettempdir())
    temp_file = temp_dir / f"claude-url-{url_name}{suffix}"
    temp_file.write_text(response.text, encoding="utf-8")
    return temp_file


@cli.command("json")
@click.argument("json_file", type=click.Path())
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on filename (uses -o as parent, or current dir).",
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name) for commit links. Auto-detected from git push output if not specified.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the original JSON session file in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
def json_cmd(json_file, output, output_auto, repo, gist, include_json, open_browser):
    """Convert a Claude Code session JSON/JSONL file or URL to HTML."""
    # Handle URL input
    if is_url(json_file):
        click.echo(f"Fetching {json_file}...")
        temp_file = fetch_url_to_tempfile(json_file)
        json_file_path = temp_file
        # Use URL path for naming
        url_name = Path(json_file.split("?")[0]).stem or "session"
    else:
        # Validate that local file exists
        json_file_path = Path(json_file)
        if not json_file_path.exists():
            raise click.ClickException(f"File not found: {json_file}")
        url_name = None

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / (url_name or json_file_path.stem)
    elif output is None:
        output = (
            Path(tempfile.gettempdir())
            / f"claude-session-{url_name or json_file_path.stem}"
        )

    output = Path(output)
    generate_html(json_file_path, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Copy JSON file to output directory if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / json_file_path.name
        shutil.copy(json_file_path, json_dest)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def resolve_credentials(token, org_uuid):
    """Resolve token and org_uuid from arguments or auto-detect.

    Returns (token, org_uuid) tuple.
    Raises click.ClickException if credentials cannot be resolved.
    """
    # Get token
    if token is None:
        token = get_access_token_from_keychain()
        if token is None:
            if platform.system() == "Darwin":
                raise click.ClickException(
                    "Could not retrieve access token from macOS keychain. "
                    "Make sure you are logged into Claude Code, or provide --token."
                )
            else:
                raise click.ClickException(
                    "On non-macOS platforms, you must provide --token with your access token."
                )

    # Get org UUID
    if org_uuid is None:
        org_uuid = get_org_uuid_from_config()
        if org_uuid is None:
            raise click.ClickException(
                "Could not find organization UUID in ~/.claude.json. "
                "Provide --org-uuid with your organization UUID."
            )

    return token, org_uuid


def format_session_for_display(session_data):
    """Format a session for display in the list or picker.

    Shows repo first (if available), then date, then title.
    Returns a formatted string.
    """
    title = session_data.get("title", "Untitled")
    created_at = session_data.get("created_at", "")
    repo = session_data.get("repo")
    # Truncate title if too long
    if len(title) > 50:
        title = title[:47] + "..."
    # Format: repo (or placeholder)  date  title
    repo_display = repo if repo else "(no repo)"
    date_display = created_at[:19] if created_at else "N/A"
    return f"{repo_display:30}  {date_display:19}  {title}"


def generate_html_from_session_data(session_data, output_dir, github_repo=None):
    """Generate HTML from session data dict (instead of file path)."""
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)

    # Web sessions carry their own display title; they have no away-summary
    # recaps, so the card falls back to the last assistant snippet.
    title = session_data.get("title")
    recap = None

    loglines = session_data.get("loglines", [])

    # Auto-detect GitHub repo if not provided
    if github_repo is None:
        github_repo = detect_github_repo(loglines)
        if github_repo:
            click.echo(f"Auto-detected GitHub repo: {github_repo}")

    # Set module-level variable for render functions
    global _github_repo
    _github_repo = github_repo

    _render_session_pages(loglines, output_dir, title, recap, echo=click.echo)


@cli.command("web")
@click.argument("session_id", required=False)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    help="Output directory. If not specified, writes to temp dir and opens in browser.",
)
@click.option(
    "-a",
    "--output-auto",
    is_flag=True,
    help="Auto-name output subdirectory based on session ID (uses -o as parent, or current dir).",
)
@click.option("--token", help="API access token (auto-detected from keychain on macOS)")
@click.option(
    "--org-uuid", help="Organization UUID (auto-detected from ~/.claude.json)"
)
@click.option(
    "--repo",
    help="GitHub repo (owner/name). Filters session list and sets default for commit links.",
)
@click.option(
    "--gist",
    is_flag=True,
    help="Upload to GitHub Gist and output a gisthost.github.io URL.",
)
@click.option(
    "--json",
    "include_json",
    is_flag=True,
    help="Include the JSON session data in the output directory.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated index.html in your default browser (default if no -o specified).",
)
def web_cmd(
    session_id,
    output,
    output_auto,
    token,
    org_uuid,
    repo,
    gist,
    include_json,
    open_browser,
):
    """Select and convert a web session from the Claude API to HTML.

    If SESSION_ID is not provided, displays an interactive picker to select a session.
    """
    try:
        token, org_uuid = resolve_credentials(token, org_uuid)
    except click.ClickException:
        raise

    # If no session ID provided, show interactive picker
    if session_id is None:
        try:
            sessions_data = fetch_sessions(token, org_uuid)
        except httpx.HTTPStatusError as e:
            raise click.ClickException(
                f"API request failed: {e.response.status_code} {e.response.text}"
            )
        except httpx.RequestError as e:
            raise click.ClickException(f"Network error: {e}")

        sessions = sessions_data.get("data", [])
        if not sessions:
            raise click.ClickException("No sessions found.")

        # Enrich sessions with repo information (extracted from session metadata)
        sessions = enrich_sessions_with_repos(sessions)

        # Filter by repo if specified
        if repo:
            sessions = filter_sessions_by_repo(sessions, repo)
            if not sessions:
                raise click.ClickException(f"No sessions found for repo: {repo}")

        # Build choices for questionary
        choices = []
        for s in sessions:
            sid = s.get("id", "unknown")
            display = format_session_for_display(s)
            choices.append(questionary.Choice(title=display, value=sid))

        selected = questionary.select(
            "Select a session to import:",
            choices=choices,
        ).ask()

        if selected is None:
            # User cancelled
            raise click.ClickException("No session selected.")

        session_id = selected

    # Fetch the session
    click.echo(f"Fetching session {session_id}...")
    try:
        session_data = fetch_session(token, org_uuid, session_id)
    except httpx.HTTPStatusError as e:
        raise click.ClickException(
            f"API request failed: {e.response.status_code} {e.response.text}"
        )
    except httpx.RequestError as e:
        raise click.ClickException(f"Network error: {e}")

    # Determine output directory and whether to open browser
    # If no -o specified, use temp dir and open browser by default
    auto_open = output is None and not gist and not output_auto
    if output_auto:
        # Use -o as parent dir (or current dir), with auto-named subdirectory
        parent_dir = Path(output) if output else Path(".")
        output = parent_dir / session_id
    elif output is None:
        output = Path(tempfile.gettempdir()) / f"claude-session-{session_id}"

    output = Path(output)
    click.echo(f"Generating HTML in {output}/...")
    generate_html_from_session_data(session_data, output, github_repo=repo)

    # Show output directory
    click.echo(f"Output: {output.resolve()}")

    # Save JSON session data if requested
    if include_json:
        output.mkdir(exist_ok=True)
        json_dest = output / f"{session_id}.json"
        with open(json_dest, "w") as f:
            json.dump(session_data, f, indent=2)
        json_size_kb = json_dest.stat().st_size / 1024
        click.echo(f"JSON: {json_dest} ({json_size_kb:.1f} KB)")

    if gist:
        # Inject gist preview JS and create gist
        inject_gist_preview_js(output)
        click.echo("Creating GitHub gist...")
        gist_id, gist_url = create_gist(output)
        preview_url = f"https://gisthost.github.io/?{gist_id}/index.html"
        click.echo(f"Gist: {gist_url}")
        click.echo(f"Preview: {preview_url}")

    if open_browser or auto_open:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


@cli.command("all")
@click.option(
    "-s",
    "--source",
    type=click.Path(exists=True),
    help="Source directory containing Claude projects (default: ~/.claude/projects).",
)
@click.option(
    "-o",
    "--output",
    type=click.Path(),
    default="./claude-archive",
    help="Output directory for the archive (default: ./claude-archive).",
)
@click.option(
    "--include-agents",
    is_flag=True,
    help="Include agent-* session files (excluded by default).",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be converted without creating files.",
)
@click.option(
    "--check",
    "check_only",
    is_flag=True,
    help="Report which session outputs are stale (source modified or tool "
    "version changed) without writing. Exits non-zero if any are stale.",
)
@click.option(
    "--if-stale",
    "if_stale",
    is_flag=True,
    help="Only regenerate sessions whose source JSONL was modified since the "
    "last render, or whose recorded tool version differs from the current one.",
)
@click.option(
    "--open",
    "open_browser",
    is_flag=True,
    help="Open the generated archive in your default browser.",
)
@click.option(
    "-q",
    "--quiet",
    is_flag=True,
    help="Suppress all output except errors.",
)
def all_cmd(
    source,
    output,
    include_agents,
    dry_run,
    check_only,
    if_stale,
    open_browser,
    quiet,
):
    """Convert all local Claude Code sessions to a browsable HTML archive.

    Creates a directory structure with:
    - Master index listing all projects
    - Per-project pages listing sessions
    - Individual session transcripts
    """
    # --dry-run, --check, and --if-stale all alter the default "rebuild
    # everything" behavior in incompatible ways. Reject combinations up front
    # rather than letting them silently override each other.
    mode_flags = {
        "--dry-run": dry_run,
        "--check": check_only,
        "--if-stale": if_stale,
    }
    enabled = [name for name, on in mode_flags.items() if on]
    if len(enabled) > 1:
        raise click.UsageError(f"{' and '.join(enabled)} are mutually exclusive.")

    # Default source folder
    if source is None:
        source = Path.home() / ".claude" / "projects"
    else:
        source = Path(source)

    if not source.exists():
        raise click.ClickException(f"Source directory not found: {source}")

    output = Path(output)

    if not quiet:
        click.echo(f"Scanning {source}...")

    projects = find_all_sessions(source, include_agents=include_agents)

    if not projects:
        if not quiet:
            click.echo("No sessions found.")
        return

    # Calculate totals
    total_sessions = sum(len(p["sessions"]) for p in projects)

    if not quiet:
        click.echo(f"Found {len(projects)} projects with {total_sessions} sessions")

    if dry_run:
        # Dry-run always outputs (it's the point of dry-run), but respects --quiet
        if not quiet:
            click.echo("\nDry run - would convert:")
            for project in projects:
                click.echo(
                    f"\n  {project['name']} ({len(project['sessions'])} sessions)"
                )
                for session in project["sessions"][:3]:  # Show first 3
                    mod_time = datetime.fromtimestamp(session["mtime"])
                    click.echo(
                        f"    - {session['path'].stem} ({mod_time.strftime('%Y-%m-%d')})"
                    )
                if len(project["sessions"]) > 3:
                    click.echo(f"    ... and {len(project['sessions']) - 3} more")
        return

    if check_only:
        current_version = _get_tool_version()
        stale = []
        for project in projects:
            for session in project["sessions"]:
                session_dir = output / project["name"] / session["path"].stem
                is_stale, reason = _session_is_stale(
                    session, session_dir, current_version
                )
                if is_stale:
                    stale.append((project["name"], session["path"].stem, reason))
        if not quiet:
            for project_name, stem, reason in stale:
                click.echo(f"STALE  {project_name}/{stem}  {reason}")
            click.echo(f"{len(stale)} of {total_sessions} stale.")
        if stale:
            raise click.exceptions.Exit(1)
        return

    if not quiet:
        click.echo(f"\nGenerating archive in {output}...")

    # Progress callback for non-quiet mode
    def on_progress(project_name, session_name, current, total):
        if not quiet and current % 10 == 0:
            click.echo(f"  Processed {current}/{total} sessions...")

    # Generate the archive using the library function
    stats = generate_batch_html(
        source,
        output,
        include_agents=include_agents,
        progress_callback=on_progress,
        only_stale=if_stale,
    )

    # Report any failures
    if stats["failed_sessions"]:
        click.echo(f"\nWarning: {len(stats['failed_sessions'])} session(s) failed:")
        for failure in stats["failed_sessions"]:
            click.echo(
                f"  {failure['project']}/{failure['session']}: {failure['error']}"
            )

    if not quiet:
        if if_stale:
            click.echo(
                f"\nregenerated {stats['regenerated_sessions']}, "
                f"skipped {stats['skipped_sessions']}, "
                f"failed {len(stats['failed_sessions'])}"
            )
        else:
            click.echo(
                f"\nGenerated archive with {stats['total_projects']} projects, "
                f"{stats['total_sessions']} sessions"
            )
        click.echo(f"Output: {output.resolve()}")

    if open_browser:
        index_url = (output / "index.html").resolve().as_uri()
        webbrowser.open(index_url)


def main():
    cli()
