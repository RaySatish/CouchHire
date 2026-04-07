"""CDP session manager — launches Chromium with remote debugging for Playwright + human takeover.

Launches a real Chromium instance with --remote-debugging-port so both Playwright
(via connect_over_cdp()) and a human (via chrome://inspect) can connect to the
same browser session simultaneously.

Public API:
    launch_browser(session_id, headless) -> dict   — start Chromium, return connection info
    get_cdp_url(session_id) -> str                 — HTTP URL for a running session
    get_takeover_instructions(session_id) -> str    — human-readable takeover guide
    close_browser(session_id) -> None              — terminate and clean up
"""

import json
import logging
import platform
import shutil
import subprocess
import time
import urllib.error
import urllib.request
from pathlib import Path
from subprocess import Popen

from config import CDP_PORT

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level state — tracks running browser sessions
# ---------------------------------------------------------------------------
# {session_id: {"process": Popen, "cdp_port": int, "headless": bool}}
_SESSIONS: dict[str, dict] = {}

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
_CDP_STARTUP_TIMEOUT: float = 10.0  # seconds to wait for CDP to become available
_CDP_POLL_INTERVAL: float = 0.5     # seconds between polls
_TERMINATE_TIMEOUT: float = 5.0     # seconds to wait after terminate() before kill()


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _find_chromium_executable() -> str:
    """Locate a Chromium/Chrome executable. Returns the path string.

    Priority:
        1. Playwright's bundled Chromium (most reliable — known compatible version)
        2. System chromium / chromium-browser
        3. System google-chrome / google-chrome-stable
        4. macOS: /Applications/Google Chrome.app/Contents/MacOS/Google Chrome
    """
    # 1. Playwright's bundled Chromium
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            pw_path = p.chromium.executable_path
            if pw_path and Path(pw_path).exists():
                logger.debug("Using Playwright bundled Chromium: %s", pw_path)
                return pw_path
    except Exception as exc:
        logger.debug("Playwright Chromium lookup failed: %s", exc)

    # 2. System chromium
    for name in ("chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            logger.debug("Using system Chromium: %s", path)
            return path

    # 3. System Chrome
    for name in ("google-chrome", "google-chrome-stable"):
        path = shutil.which(name)
        if path:
            logger.debug("Using system Chrome: %s", path)
            return path

    # 4. macOS default location
    if platform.system() == "Darwin":
        mac_path = Path("/Applications/Google Chrome.app/Contents/MacOS/Google Chrome")
        if mac_path.exists():
            logger.debug("Using macOS Chrome: %s", mac_path)
            return str(mac_path)

    raise RuntimeError(
        "No Chromium/Chrome executable found. Install one of:\n"
        "  • Playwright Chromium: python -m playwright install chromium\n"
        "  • System Chromium: apt install chromium-browser\n"
        "  • Google Chrome: https://www.google.com/chrome/"
    )


def _wait_for_cdp(port: int, timeout: float = _CDP_STARTUP_TIMEOUT) -> str:
    """Poll the CDP /json/version endpoint until it responds.

    Returns the webSocketDebuggerUrl from the response.
    Raises RuntimeError if the endpoint doesn't respond within timeout.
    """
    url = f"http://localhost:{port}/json/version"
    deadline = time.monotonic() + timeout

    while time.monotonic() < deadline:
        try:
            resp = urllib.request.urlopen(url, timeout=2)
            data = json.loads(resp.read().decode())
            ws_url = data.get("webSocketDebuggerUrl", "")
            if ws_url:
                return ws_url
            # Response came back but no WS URL yet — keep polling
        except (urllib.error.URLError, ConnectionRefusedError, OSError):
            pass  # CDP not ready yet
        time.sleep(_CDP_POLL_INTERVAL)

    raise RuntimeError(
        f"CDP endpoint at localhost:{port} did not respond within {timeout}s. "
        f"Check if port {port} is already in use: lsof -i :{port}"
    )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def launch_browser(session_id: str = "default", headless: bool = True) -> dict:
    """Launch Chromium with --remote-debugging-port and return connection info.

    Args:
        session_id: Identifier for this browser session (for logging/tracking).
        headless: If True, runs with --headless=new. If False, shows the
                  browser window (useful for local development).

    Returns:
        dict with keys:
            - "cdp_url": str — HTTP URL (e.g. "http://localhost:9222")
            - "ws_url": str — WebSocket debugger URL from /json/version
            - "session_id": str
            - "pid": int — Chromium process ID (for cleanup)

    Raises:
        RuntimeError: If Chromium fails to start or CDP port is unavailable.
    """
    # Check if session already exists
    if session_id in _SESSIONS:
        existing = _SESSIONS[session_id]
        if existing["process"].poll() is None:
            raise RuntimeError(
                f"Browser session '{session_id}' is already running "
                f"(pid={existing['process'].pid}). Close it first or use a "
                f"different session_id."
            )
        # Process died — clean up stale entry
        logger.warning(
            "Stale session '%s' found (process exited). Cleaning up.", session_id
        )
        _SESSIONS.pop(session_id, None)

    port = CDP_PORT
    executable = _find_chromium_executable()
    user_data_dir = Path(f"/tmp/couchhire-chrome-{session_id}")

    # Build command
    cmd = [
        executable,
        f"--remote-debugging-port={port}",
        f"--user-data-dir={user_data_dir}",
        "--no-first-run",
        "--no-default-browser-check",
        "--disable-background-networking",
        "--disable-sync",
        "--disable-translate",
    ]
    if headless:
        cmd.append("--headless=new")

    # Launch
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            f"Failed to launch Chromium at '{executable}': {exc}"
        ) from exc
    except OSError as exc:
        raise RuntimeError(
            f"OS error launching Chromium at '{executable}': {exc}"
        ) from exc

    # Wait for CDP to become available
    try:
        ws_url = _wait_for_cdp(port)
    except RuntimeError:
        # CDP never came up — kill the process and re-raise
        process.kill()
        process.wait(timeout=5)
        raise

    cdp_url = f"http://localhost:{port}"

    # Store session
    _SESSIONS[session_id] = {
        "process": process,
        "cdp_port": port,
        "headless": headless,
    }

    logger.info(
        "Browser launched (session=%s, pid=%d, cdp=%s)",
        session_id, process.pid, cdp_url,
    )

    return {
        "cdp_url": cdp_url,
        "ws_url": ws_url,
        "session_id": session_id,
        "pid": process.pid,
    }


def get_cdp_url(session_id: str = "default") -> str:
    """Return the CDP HTTP URL for a running session."""
    if session_id not in _SESSIONS:
        raise RuntimeError(
            f"No browser session '{session_id}' found. "
            f"Call launch_browser(session_id='{session_id}') first."
        )

    session = _SESSIONS[session_id]
    process = session["process"]

    if process.poll() is not None:
        # Process has exited
        _SESSIONS.pop(session_id, None)
        raise RuntimeError(
            f"Browser session '{session_id}' has exited "
            f"(exit code={process.returncode}). Relaunch with launch_browser()."
        )

    return f"http://localhost:{session['cdp_port']}"


def get_takeover_instructions(session_id: str = "default") -> str:
    """Return human-readable instructions for connecting to the browser session."""
    if session_id not in _SESSIONS:
        raise RuntimeError(
            f"No browser session '{session_id}' found. "
            f"Call launch_browser(session_id='{session_id}') first."
        )

    session = _SESSIONS[session_id]
    port = session["cdp_port"]

    if not session["headless"]:
        return (
            "🔗 Browser Takeover Instructions\n"
            "\n"
            f"Session: {session_id} | Port: {port}\n"
            "\n"
            "The browser window is visible on your screen. Interact directly.\n"
            "\n"
            "After you're done, tap [Done] in Telegram to resume the agent."
        )

    return (
        "🔗 Browser Takeover Instructions\n"
        "\n"
        f"Session: {session_id} | Port: {port}\n"
        "\n"
        "1. Open Google Chrome on your machine\n"
        "2. Navigate to: chrome://inspect/#devices\n"
        f"3. Click \"Configure...\" and add: localhost:{port}\n"
        "4. Your session should appear under \"Remote Target\"\n"
        "5. Click \"inspect\" to take control\n"
        "\n"
        "After you're done, tap [Done] in Telegram to resume the agent."
    )


def close_browser(session_id: str = "default") -> None:
    """Terminate the Chromium process and clean up."""
    if session_id not in _SESSIONS:
        logger.warning("No browser session '%s' to close.", session_id)
        return

    session = _SESSIONS.pop(session_id)
    process = session["process"]
    port = session["cdp_port"]

    # Terminate gracefully, then force-kill if needed
    if process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=_TERMINATE_TIMEOUT)
        except subprocess.TimeoutExpired:
            logger.warning(
                "Browser session '%s' did not terminate in %.0fs, killing.",
                session_id, _TERMINATE_TIMEOUT,
            )
            process.kill()
            process.wait(timeout=5)

    # Clean up user-data-dir
    user_data_dir = Path(f"/tmp/couchhire-chrome-{session_id}")
    if user_data_dir.exists():
        shutil.rmtree(user_data_dir, ignore_errors=True)
        logger.debug("Cleaned up user-data-dir: %s", user_data_dir)

    logger.info("Browser closed (session=%s)", session_id)
