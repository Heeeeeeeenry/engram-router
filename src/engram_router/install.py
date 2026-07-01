"""engram install — one-command agent integration.

Auto-detects installed AI agents and configures their MCP settings
so engram-router is available as a memory tool everywhere.

Usage:
    engram install              # auto-detect all agents
    engram install --hermes     # only Hermes
    engram install --dry-run    # show what would be done
    engram status               # show which agents are connected
    engram uninstall            # remove all engram MCP configs

Supported agents:
    Hermes Agent    — hermes mcp add
    Claude Desktop  — claude_desktop_config.json
    Claude Code     — .claude/mcp.json (project or global)
    OpenClaw        — openclaw mcp add
    Codex (OpenAI)  — .codex/mcp.json
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


# ═══════════════════════════════════════════════════════════════════════════
# Agent detectors
# ═══════════════════════════════════════════════════════════════════════════

@dataclass
class AgentAdapter:
    """How to install/uninstall engram MCP for one agent type."""

    name: str
    key: str
    description: str

    detect: Any = None       # callable() -> bool
    install: Any = None      # callable(dry_run: bool) -> str
    uninstall: Any = None    # callable(dry_run: bool) -> str
    status: Any = None       # callable() -> str

    installed: bool = False
    status_text: str = ""


# ═══════════════════════════════════════════════════════════════════════════
# Hermes Agent
# ═══════════════════════════════════════════════════════════════════════════

def _detect_hermes() -> bool:
    return shutil.which("hermes") is not None

def _install_hermes(dry_run: bool = False) -> str:
    """Write MCP config directly to Hermes config.yaml (avoids interactive prompts)."""
    engram_mcp = shutil.which("engram-mcp") or "engram-mcp"
    config_path = Path.home() / ".hermes" / "config.yaml"

    if not config_path.exists():
        return "✗ Hermes: config.yaml not found"

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except ImportError:
        # Fallback: use simple string check
        with open(config_path) as f:
            content = f.read()
        if "engram-mcp" in content:
            return "✓ Hermes: engram already configured"
        if dry_run:
            return f"would add engram MCP server to {config_path}"
        # Append MCP config
        with open(config_path, "a") as f:
            f.write(f"\n# engram-router MCP (added by engram-install)\n")
            f.write(f"mcp_servers:\n")
            f.write(f"  engram:\n")
            f.write(f"    command: {engram_mcp}\n")
            f.write(f"    args:\n")
            f.write(f"      - --db\n")
            f.write(f"      - {_default_db()}\n")
        return f"✓ Hermes: engram added to {config_path}"

    servers = config.setdefault("mcp_servers", config.setdefault("mcp", {}))
    if isinstance(servers, list):
        for s in servers:
            if isinstance(s, dict) and s.get("name") == "engram":
                return "✓ Hermes: engram already configured"
    elif isinstance(servers, dict):
        if "engram" in servers:
            return "✓ Hermes: engram already configured"

    if dry_run:
        return f"would add engram MCP server to {config_path}"

    entry = {"name": "engram", "command": engram_mcp, "args": ["--db", _default_db()]}
    if isinstance(servers, list):
        servers.append(entry)
    else:
        servers["engram"] = entry

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return f"✓ Hermes: engram added to {config_path}"

def _uninstall_hermes(dry_run: bool = False) -> str:
    config_path = Path.home() / ".hermes" / "config.yaml"
    if not config_path.exists():
        return "✓ Hermes: not configured"

    try:
        import yaml
        with open(config_path) as f:
            config = yaml.safe_load(f) or {}
    except ImportError:
        with open(config_path) as f:
            content = f.read()
        if "engram-mcp" not in content and "engram-router" not in content:
            return "✓ Hermes: engram not configured"
        if dry_run:
            return f"would remove engram from {config_path}"
        # Remove engram block
        lines = content.split("\n")
        new_lines = []
        skip = False
        for line in lines:
            if "engram-router MCP" in line or "engram-mcp" in line:
                skip = True
                continue
            if skip and (line.startswith("mcp") or line.startswith("#") or not line.strip()):
                continue
            skip = False
            new_lines.append(line)
        with open(config_path, "w") as f:
            f.write("\n".join(new_lines))
        return f"✓ Hermes: engram removed from {config_path}"

    removed = False
    for key in ("mcp_servers", "mcp"):
        servers = config.get(key)
        if isinstance(servers, list):
            config[key] = [s for s in servers if not (isinstance(s, dict) and s.get("name") == "engram")]
            if len(config[key]) != len(servers):
                removed = True
        elif isinstance(servers, dict) and "engram" in servers:
            del servers["engram"]
            removed = True

    if not removed:
        return "✓ Hermes: engram not configured"
    if dry_run:
        return f"would remove engram from {config_path}"
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
    return f"✓ Hermes: engram removed from {config_path}"

def _status_hermes() -> str:
    try:
        result = subprocess.run(
            ["hermes", "mcp", "list"], capture_output=True, text=True, timeout=10)
        if "engram" in result.stdout.lower():
            return "connected (engram MCP server registered)"
        return "not connected"
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Claude Desktop
# ═══════════════════════════════════════════════════════════════════════════

def _claude_desktop_config_path() -> Path | None:
    """Return the Claude Desktop config path for this OS."""
    if sys.platform == "darwin":
        return Path.home() / "Library" / "Application Support" / "Claude" / "claude_desktop_config.json"
    elif sys.platform == "win32":
        return Path(os.environ.get("APPDATA", "")) / "Claude" / "claude_desktop_config.json"
    else:
        return None  # Linux: not sure of standard path

def _detect_claude_desktop() -> bool:
    path = _claude_desktop_config_path()
    return path is not None and path.exists()

def _read_json(path: Path) -> dict:
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, FileNotFoundError):
        return {}

def _write_json(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(data, f, indent=2)

def _install_claude_desktop(dry_run: bool = False) -> str:
    path = _claude_desktop_config_path()
    if not path:
        return "✗ Claude Desktop: config path unknown for this OS"

    config = _read_json(path)
    servers = config.setdefault("mcpServers", {})

    if "engram" in servers:
        return "✓ Claude Desktop: engram already configured"

    if dry_run:
        return f"would add engram MCP server to {path}"

    servers["engram"] = {
        "command": "engram-mcp",
        "args": ["--db", _default_db()],
    }
    _write_json(path, config)
    return f"✓ Claude Desktop: engram added to {path}"

def _uninstall_claude_desktop(dry_run: bool = False) -> str:
    path = _claude_desktop_config_path()
    if not path or not path.exists():
        return "✗ Claude Desktop: config not found"

    config = _read_json(path)
    servers = config.get("mcpServers", {})
    if "engram" not in servers:
        return "✓ Claude Desktop: engram not configured (nothing to remove)"

    if dry_run:
        return f"would remove engram from {path}"

    del servers["engram"]
    _write_json(path, config)
    return f"✓ Claude Desktop: engram removed from {path}"

def _status_claude_desktop() -> str:
    path = _claude_desktop_config_path()
    if not path or not path.exists():
        return "not installed"
    config = _read_json(path)
    if "engram" in config.get("mcpServers", {}):
        return "connected"
    return "not connected"


# ═══════════════════════════════════════════════════════════════════════════
# Claude Code
# ═══════════════════════════════════════════════════════════════════════════

def _claude_code_paths() -> list[Path]:
    """Return all possible Claude Code MCP config locations."""
    paths = []
    # Project-local
    cwd = Path.cwd()
    if (cwd / ".claude").exists():
        paths.append(cwd / ".claude" / "mcp.json")
    # Global
    paths.append(Path.home() / ".claude" / "mcp.json")
    return paths

def _detect_claude_code() -> bool:
    return shutil.which("claude") is not None

def _install_claude_code(dry_run: bool = False) -> str:
    # Install to global config
    path = Path.home() / ".claude" / "mcp.json"
    config = _read_json(path) if path.exists() else {}
    servers = config.setdefault("mcpServers", {})

    if "engram" in servers:
        return "✓ Claude Code (global): engram already configured"

    if dry_run:
        return f"would add engram to {path}"

    servers["engram"] = {
        "command": "engram-mcp",
        "args": ["--db", _default_db()],
    }
    _write_json(path, config)
    return f"✓ Claude Code: engram added to {path}"

def _uninstall_claude_code(dry_run: bool = False) -> str:
    results = []
    for path in _claude_code_paths():
        if not path.exists():
            continue
        config = _read_json(path)
        if "engram" in config.get("mcpServers", {}):
            if dry_run:
                results.append(f"would remove engram from {path}")
            else:
                del config["mcpServers"]["engram"]
                _write_json(path, config)
                results.append(f"✓ Claude Code: removed from {path}")
    return " | ".join(results) if results else "✓ Claude Code: not configured"

def _status_claude_code() -> str:
    for path in _claude_code_paths():
        if path.exists():
            config = _read_json(path)
            if "engram" in config.get("mcpServers", {}):
                return f"connected ({path})"
    return "not connected"


# ═══════════════════════════════════════════════════════════════════════════
# OpenClaw
# ═══════════════════════════════════════════════════════════════════════════

def _detect_openclaw() -> bool:
    return shutil.which("openclaw") is not None

def _install_openclaw(dry_run: bool = False) -> str:
    cmd = ["openclaw", "mcp", "add", "engram",
           "--command", f"engram-mcp --db {_default_db()}"]
    if dry_run:
        return f"would run: {' '.join(cmd)}"
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return f"✓ OpenClaw: engram added" if result.returncode == 0 else f"⚠ OpenClaw: {result.stderr}"
    except Exception as e:
        return f"✗ OpenClaw: {e}"

def _uninstall_openclaw(dry_run: bool = False) -> str:
    cmd = ["openclaw", "mcp", "remove", "engram"]
    if dry_run:
        return f"would run: {' '.join(cmd)}"
    try:
        subprocess.run(cmd, capture_output=True, text=True, timeout=15)
        return "✓ OpenClaw: removed"
    except Exception as e:
        return f"✗ OpenClaw: {e}"

def _status_openclaw() -> str:
    try:
        result = subprocess.run(
            ["openclaw", "mcp", "list"], capture_output=True, text=True, timeout=10)
        return "connected" if "engram" in result.stdout.lower() else "not connected"
    except Exception:
        return "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# Codex (OpenAI)
# ═══════════════════════════════════════════════════════════════════════════

def _detect_codex() -> bool:
    return shutil.which("codex") is not None

def _install_codex(dry_run: bool = False) -> str:
    path = Path.home() / ".codex" / "mcp.json"
    config = _read_json(path) if path.exists() else {}
    servers = config.setdefault("mcpServers", {})

    if "engram" in servers:
        return "✓ Codex: engram already configured"

    if dry_run:
        return f"would add engram to {path}"

    servers["engram"] = {
        "command": "engram-mcp",
        "args": ["--db", _default_db()],
    }
    _write_json(path, config)
    return f"✓ Codex: engram added to {path}"

def _uninstall_codex(dry_run: bool = False) -> str:
    path = Path.home() / ".codex" / "mcp.json"
    if not path.exists():
        return "✓ Codex: not configured"
    config = _read_json(path)
    if "engram" in config.get("mcpServers", {}):
        if dry_run:
            return f"would remove engram from {path}"
        del config["mcpServers"]["engram"]
        _write_json(path, config)
        return f"✓ Codex: removed from {path}"
    return "✓ Codex: not configured"

def _status_codex() -> str:
    path = Path.home() / ".codex" / "mcp.json"
    if path.exists():
        config = _read_json(path)
        if "engram" in config.get("mcpServers", {}):
            return f"connected ({path})"
    return "not connected"


# ═══════════════════════════════════════════════════════════════════════════
# Registry
# ═══════════════════════════════════════════════════════════════════════════

def _default_db() -> str:
    return str(Path.home() / ".engram" / "memory.db")


AGENTS: list[AgentAdapter] = [
    AgentAdapter(
        name="Hermes Agent",
        key="hermes",
        description="Hermes Agent (hermes CLI)",
        detect=_detect_hermes,
        install=_install_hermes,
        uninstall=_uninstall_hermes,
        status=_status_hermes,
    ),
    AgentAdapter(
        name="Claude Desktop",
        key="claude-desktop",
        description="Anthropic Claude Desktop app",
        detect=_detect_claude_desktop,
        install=_install_claude_desktop,
        uninstall=_uninstall_claude_desktop,
        status=_status_claude_desktop,
    ),
    AgentAdapter(
        name="Claude Code",
        key="claude-code",
        description="Anthropic Claude Code CLI",
        detect=_detect_claude_code,
        install=_install_claude_code,
        uninstall=_uninstall_claude_code,
        status=_status_claude_code,
    ),
    AgentAdapter(
        name="OpenClaw",
        key="openclaw",
        description="OpenClaw agent framework",
        detect=_detect_openclaw,
        install=_install_openclaw,
        uninstall=_uninstall_openclaw,
        status=_status_openclaw,
    ),
    AgentAdapter(
        name="Codex",
        key="codex",
        description="OpenAI Codex CLI",
        detect=_detect_codex,
        install=_install_codex,
        uninstall=_uninstall_codex,
        status=_status_codex,
    ),
]


# ═══════════════════════════════════════════════════════════════════════════
# Commands
# ═══════════════════════════════════════════════════════════════════════════

def cmd_install(agents: list[str] | None = None, dry_run: bool = False) -> int:
    """Install engram MCP config for detected (or specified) agents."""
    if agents is None:
        agents = []

    # Ensure DB directory exists
    db_dir = Path(_default_db()).parent
    if not dry_run:
        db_dir.mkdir(parents=True, exist_ok=True)

    print("engram-router MCP installer\n")

    configured = 0
    for agent in AGENTS:
        if agents and agent.key not in agents:
            continue

        detected = agent.detect() if agent.detect else False
        if not detected and agents:
            print(f"  {agent.name}: not found (skipping)")
            continue
        if not detected:
            continue

        result = agent.install(dry_run)
        print(f"  {result}")
        if "✓" in result:
            configured += 1

    if configured == 0 and not dry_run:
        print("\nNo agents detected. Install one of: hermes, claude, codex, openclaw")
        return 1

    print(f"\n{configured} agent(s) configured.")
    if not dry_run:
        print(f"Memory DB: {_default_db()}")
        print("Restart your agents to pick up the new MCP server.")
    return 0


def cmd_status() -> int:
    """Show connection status for all agents."""
    print("engram-router status\n")
    print(f"{'Agent':<20} {'Status':<30} {'Detected'}")
    print("-" * 65)

    for agent in AGENTS:
        detected = agent.detect() if agent.detect else False
        status = agent.status() if agent.status else "unknown"
        print(f"{agent.name:<20} {status:<30} {'✓' if detected else '✗'}")

    return 0


def cmd_uninstall(agents: list[str] | None = None, dry_run: bool = False) -> int:
    """Remove engram MCP config from agents."""
    if agents is None:
        agents = []

    print("engram-router MCP uninstall\n")

    removed = 0
    for agent in AGENTS:
        if agents and agent.key not in agents:
            continue
        result = agent.uninstall(dry_run)
        print(f"  {result}")
        if "✓" in result:
            removed += 1

    print(f"\n{removed} agent(s) cleaned up.")
    return 0


# ═══════════════════════════════════════════════════════════════════════════
# CLI entry point
# ═══════════════════════════════════════════════════════════════════════════

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(
        prog="engram-install",
        description="One-command engram-router MCP integration for AI agents",
    )
    sub = parser.add_subparsers(dest="command")

    inst = sub.add_parser("install", help="Auto-detect and configure all agents")
    inst.add_argument("--hermes", action="store_true")
    inst.add_argument("--claude-desktop", action="store_true")
    inst.add_argument("--claude-code", action="store_true")
    inst.add_argument("--openclaw", action="store_true")
    inst.add_argument("--codex", action="store_true")
    inst.add_argument("--dry-run", action="store_true", help="Show what would be done")

    sub.add_parser("status", help="Show agent connection status")

    uninst = sub.add_parser("uninstall", help="Remove engram MCP config")
    uninst.add_argument("--hermes", action="store_true")
    uninst.add_argument("--claude-desktop", action="store_true")
    uninst.add_argument("--claude-code", action="store_true")
    uninst.add_argument("--openclaw", action="store_true")
    uninst.add_argument("--codex", action="store_true")
    uninst.add_argument("--dry-run", action="store_true")

    args = parser.parse_args()

    if args.command == "install":
        selected = [k for k in ["hermes", "claude-desktop", "claude-code", "openclaw", "codex"]
                    if getattr(args, k.replace("-", "_"), False)]
        sys.exit(cmd_install(agents=selected or None, dry_run=args.dry_run))
    elif args.command == "status":
        sys.exit(cmd_status())
    elif args.command == "uninstall":
        selected = [k for k in ["hermes", "claude-desktop", "claude-code", "openclaw", "codex"]
                    if getattr(args, k.replace("-", "_"), False)]
        sys.exit(cmd_uninstall(agents=selected or None, dry_run=args.dry_run))
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
