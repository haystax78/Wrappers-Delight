"""Update check and in-place install from the GitHub releases.

On start-up (when enabled in the add-on preferences and Blender is allowed
online access) the latest release of the repository is fetched in a
background thread.  If its version is newer than the installed one, the
sidebar shows a short notice; clicking it downloads that release and copies
it over the installed add-on (saved templates are kept).  The new code runs
after Blender restarts.

Release versions come from the tag (``v1.2``, ``1.2.0`` ...).  The release's
first ``.zip`` asset is used if there is one, else GitHub's source zip.
"""

import io
import json
import os
import re
import shutil
import tempfile
import threading
import urllib.request
import zipfile

import bpy

REPO = "haystax78/Wrappers-Delight"
API_LATEST = f"https://api.github.com/repos/{REPO}/releases/latest"
RELEASES_PAGE = f"https://github.com/{REPO}/releases"
TIMEOUT = 15
KEEP = ("templates",)            # never overwritten by an update

ADDON_DIR = os.path.dirname(os.path.abspath(__file__))

# shared with the UI; written by worker threads, read on the main thread
state = {
    "status": "IDLE",   # IDLE CHECKING UP_TO_DATE AVAILABLE DOWNLOADING
                        # INSTALLED ERROR OFFLINE
    "latest": None,     # (major, minor, patch)
    "tag": "",
    "url": "",          # zip to download
    "message": "",
}
_lock = threading.Lock()


def installed_version():
    from . import bl_info
    return tuple(bl_info["version"])


def parse_version(text):
    """(major, minor, patch) from a tag such as 'v1.2' or '1.2.3'."""
    m = re.search(r"(\d+)(?:\.(\d+))?(?:\.(\d+))?", text or "")
    if not m:
        return None
    return tuple(int(g or 0) for g in m.groups())


def version_text(v):
    if not v:
        return ""
    return ".".join(str(n) for n in (v if v[2] else v[:2]))


def _set(**kw):
    """Update the shared state (safe from worker threads: no bpy here)."""
    with _lock:
        state.update(kw)


_BUSY = {"CHECKING", "DOWNLOADING"}


def _redraw():
    wm = getattr(bpy.context, "window_manager", None)
    if wm is not None:
        for win in wm.windows:
            for area in win.screen.areas:
                if area.type in {'VIEW_3D', 'PREFERENCES'}:
                    area.tag_redraw()


def _poll():
    """Main-thread timer: repaint while a worker runs, once more after."""
    _redraw()
    return 0.5 if state["status"] in _BUSY else None


def _start_polling():
    if not bpy.app.timers.is_registered(_poll):
        bpy.app.timers.register(_poll, first_interval=0.2)


def is_dev_checkout():
    """Installed straight from a git working copy: never overwrite it."""
    return os.path.exists(os.path.join(ADDON_DIR, ".git"))


def _request(url, accept):
    req = urllib.request.Request(url, headers={
        "Accept": accept, "User-Agent": "wrappers-delight-updater"})
    return urllib.request.urlopen(req, timeout=TIMEOUT)


def online_allowed():
    return bool(getattr(bpy.app, "online_access", True))


# ------------------------------------------------------------------ check --

def check(force=False):
    """Start a background check (once per session unless ``force``)."""
    if state["status"] in {"CHECKING", "DOWNLOADING", "INSTALLED"}:
        return
    if not force and state["status"] != "IDLE":
        return
    if not online_allowed():
        _set(status="OFFLINE", message="Online access is off in "
             "Preferences > System")
        _redraw()
        return
    _set(status="CHECKING", message="Checking for updates...")
    threading.Thread(target=_check_worker, daemon=True).start()
    _start_polling()


def _check_worker():
    try:
        with _request(API_LATEST, "application/vnd.github+json") as resp:
            data = json.load(resp)
    except Exception as exc:  # noqa: BLE001 - network: report, never raise
        code = getattr(exc, "code", None)
        msg = ("No public release found" if code == 404
               else f"Update check failed: {exc}")
        _set(status="ERROR", message=msg)
        return
    tag = data.get("tag_name") or data.get("name") or ""
    latest = parse_version(tag)
    url = next((a.get("browser_download_url") for a in data.get("assets", [])
                if str(a.get("name", "")).lower().endswith(".zip")),
               data.get("zipball_url", ""))
    if latest is None:
        _set(status="ERROR", message=f"Can't read a version from '{tag}'")
    elif latest > installed_version():
        _set(status="AVAILABLE", latest=latest, tag=tag, url=url,
             message=f"Update available: {version_text(latest)}")
    else:
        _set(status="UP_TO_DATE", latest=latest, tag=tag,
             message=f"Up to date ({version_text(installed_version())})")


# ---------------------------------------------------------------- install --

def install():
    """Download the newer release and install it over this add-on."""
    if state["status"] != "AVAILABLE" or not state["url"]:
        return False
    if is_dev_checkout():
        _set(message="This copy is a git checkout: update it with git "
                     "(pull) instead")
        _redraw()
        return False
    _set(status="DOWNLOADING",
         message=f"Downloading {version_text(state['latest'])}...")
    threading.Thread(target=_install_worker, args=(state["url"],),
                     daemon=True).start()
    _start_polling()
    return True


def _addon_root(folder):
    """The folder holding the add-on's __init__.py inside an extracted zip."""
    for root, _dirs, files in os.walk(folder):
        if "__init__.py" in files:
            with open(os.path.join(root, "__init__.py"),
                      encoding="utf-8", errors="ignore") as fh:
                if "bl_info" in fh.read():
                    return root
    return None


def _install_worker(url):
    try:
        with _request(url, "application/octet-stream") as resp:
            payload = resp.read()
        with tempfile.TemporaryDirectory() as tmp:
            with zipfile.ZipFile(io.BytesIO(payload)) as zf:
                base = os.path.realpath(tmp)
                for name in zf.namelist():
                    dest = os.path.realpath(os.path.join(tmp, name))
                    if not dest.startswith(base + os.sep) and dest != base:
                        raise ValueError(f"Unsafe path in zip: {name}")
                zf.extractall(tmp)
            src = _addon_root(tmp)
            if src is None:
                raise ValueError("The release zip has no add-on in it")
            _copy_over(src, ADDON_DIR)
    except Exception as exc:  # noqa: BLE001 - report in the UI
        _set(status="AVAILABLE", message=f"Update failed: {exc}")
        return
    _set(status="INSTALLED",
         message=f"Updated to {version_text(state['latest'])}: restart "
                 "Blender to finish")


def _copy_over(src, dst):
    for name in os.listdir(src):
        if name in KEEP or name.startswith(".") or name == "__pycache__":
            continue
        s, d = os.path.join(src, name), os.path.join(dst, name)
        if os.path.isdir(s):
            shutil.copytree(s, d, dirs_exist_ok=True,
                            ignore=shutil.ignore_patterns("__pycache__"))
        else:
            shutil.copy2(s, d)


# --------------------------------------------------------------- operators --

class MESHWRAP_OT_update_install(bpy.types.Operator):
    """Download the newer release from GitHub and install it over this
    add-on (restart Blender afterwards)"""
    bl_idname = "meshwrap.update_install"
    bl_label = "Install Update"

    @classmethod
    def poll(cls, context):
        return state["status"] == "AVAILABLE"

    def execute(self, context):
        if not install():
            self.report({'WARNING'}, state["message"])
            return {'CANCELLED'}
        self.report({'INFO'}, state["message"])
        return {'FINISHED'}


class MESHWRAP_OT_update_check(bpy.types.Operator):
    """Check GitHub for a newer release now"""
    bl_idname = "meshwrap.update_check"
    bl_label = "Check for Updates"

    def execute(self, context):
        check(force=True)
        return {'FINISHED'}


CLASSES = (MESHWRAP_OT_update_install, MESHWRAP_OT_update_check)


def _startup_check():
    from . import preferences
    if getattr(preferences.get(), "auto_update", True):
        check()
    return None


def register():
    for cls in CLASSES:
        bpy.utils.register_class(cls)
    # after start-up, so preferences and online access are settled
    bpy.app.timers.register(_startup_check, first_interval=3.0)


def unregister():
    for fn in (_startup_check, _poll):
        if bpy.app.timers.is_registered(fn):
            bpy.app.timers.unregister(fn)
    for cls in reversed(CLASSES):
        bpy.utils.unregister_class(cls)
