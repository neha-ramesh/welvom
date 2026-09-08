"""Make ffmpeg and ffprobe available, without needing apt.

THE PROBLEM THIS SOLVES

Everything in this pipeline shells out to `ffmpeg` and `ffprobe` by bare name,
which requires them to be installed system-wide. Locally that is one `winget
install`. On Streamlit Cloud it means a `packages.txt` and an `apt-get` run at
build time — and that build fails outright whenever Debian's security mirror is
serving an expired release file, which it does periodically. The whole app then
refuses to deploy over a stale package index nobody involved can influence.

`static-ffmpeg` ships both binaries as a pip package. Putting it in
requirements.txt removes apt from the picture entirely: the same dependency
mechanism that installs pydantic installs ffmpeg.

HOW IT IS USED

`ensure_ffmpeg()` runs once at startup and prepends the binaries' directory to
PATH, so every existing `subprocess.run(["ffmpeg", ...])` call keeps working
untouched. A system install still wins if there is one — it is faster to start
and usually a newer build.
"""

from __future__ import annotations

import os
import shutil

_READY: bool | None = None


def have_ffmpeg() -> bool:
    """Both binaries present. ffprobe is optional — see have_ffprobe."""
    return bool(shutil.which("ffmpeg") and shutil.which("ffprobe"))


def have_ffprobe() -> bool:
    return bool(shutil.which("ffprobe"))


def _expose(exe: str, as_name: str) -> None:
    """Put a binary on PATH under the plain name callers use.

    The wheel names its binary `ffmpeg-linux-x86_64-v7.0.2`, and every
    subprocess call in this project says `ffmpeg`. Rather than rewrite those,
    link it into a small directory of our own and prepend that to PATH.
    """
    import tempfile

    d = os.path.join(tempfile.gettempdir(), "welvom-bin")
    os.makedirs(d, exist_ok=True)
    target = os.path.join(d, as_name + (".exe" if os.name == "nt" else ""))
    if not os.path.exists(target):
        try:
            os.symlink(exe, target)
        except (OSError, NotImplementedError):
            shutil.copy2(exe, target)   # Windows without developer mode
            os.chmod(target, 0o755)
    if d not in os.environ.get("PATH", "").split(os.pathsep):
        os.environ["PATH"] = d + os.pathsep + os.environ.get("PATH", "")


def ensure_ffmpeg(verbose: bool = False) -> bool:
    """Put ffmpeg and ffprobe on PATH. Returns whether they are usable.

    Cached, because static_ffmpeg downloads its binaries on first call and there
    is no reason to pay that on every Streamlit rerun.
    """
    global _READY
    if _READY is not None:
        return _READY

    # A real system install is preferable: faster to start, and generally a
    # newer build than the bundled one.
    if have_ffmpeg():
        _READY = True
        return True

    # static-ffmpeg gives BOTH binaries, which is the tidiest outcome — but it
    # fetches them from GitHub on first use, so it fails on any host that blocks
    # or throttles that.
    try:
        import static_ffmpeg

        static_ffmpeg.add_paths()
    except Exception as e:
        if verbose:
            print(f"  static-ffmpeg unavailable: {e}")

    if have_ffmpeg():
        _READY = True
        return True

    # imageio-ffmpeg ships the binary INSIDE the wheel, so there is nothing to
    # download and nothing to fail. It provides ffmpeg only, which is why
    # probe.py can read metadata without ffprobe.
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and os.path.exists(exe):
            _expose(exe, "ffmpeg")
    except Exception as e:
        if verbose:
            print(f"  imageio-ffmpeg unavailable: {e}")

    _READY = bool(shutil.which("ffmpeg"))
    return _READY


def which_report() -> str:
    """One line naming where the binaries came from, for diagnostics."""
    ff, fp = shutil.which("ffmpeg"), shutil.which("ffprobe")
    if not ff:
        return "ffmpeg not found"
    return f"ffmpeg at {ff}" + ("" if fp else "  (no ffprobe — reading metadata from ffmpeg)")