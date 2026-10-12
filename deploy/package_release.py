"""Build a source-only release archive with a SHA-256 manifest.

    python deploy/package_release.py            -> dist/fedtshdiff-<digest12>.zip (+ .sha256)

The archive contains ``revision_code/`` (src, tests, docs, deploy, pyproject, README) and
``revision_code/release/SOURCE_MANIFEST.json`` listing every file's hash; ``setup_gcore.sh``
verifies both before installing.
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
INCLUDE = ("src", "tests", "docs", "deploy", "pyproject.toml", "README.md")
SKIP_PARTS = {"__pycache__", ".pytest_cache", "dist", ".egg-info"}


def files() -> list[Path]:
    out = []
    for item in INCLUDE:
        p = ROOT / item
        cands = [p] if p.is_file() else sorted(q for q in p.rglob("*") if q.is_file())
        for q in cands:
            if any(part in SKIP_PARTS or part.endswith(".egg-info") for part in q.relative_to(ROOT).parts):
                continue
            out.append(q)
    return out


def main() -> int:
    sys.path.insert(0, str(ROOT / "src"))
    from fedtshdiff.utils import code_digest

    entries = []
    for f in files():
        data = f.read_bytes().replace(b"\r\n", b"\n") if f.suffix in (".py", ".sh", ".md", ".yaml", ".toml", ".txt") else f.read_bytes()
        entries.append((f.relative_to(ROOT).as_posix(), data))
    digest = code_digest()
    manifest = {"package": "fedtshdiff", "code_digest": digest, "created": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "files": [{"path": p, "sha256": hashlib.sha256(d).hexdigest()} for p, d in entries]}
    dist = ROOT / "dist"
    dist.mkdir(exist_ok=True)
    name = f"fedtshdiff-{digest[:12]}"
    archive = dist / f"{name}.zip"
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as z:
        for p, d in entries:
            info = zipfile.ZipInfo(f"revision_code/{p}", date_time=(2026, 1, 1, 0, 0, 0))
            info.create_system = 3  # Unix, so the mode bits are meaningful
            info.external_attr = (0o755 if p.endswith(".sh") else 0o644) << 16
            z.writestr(info, d)
        z.writestr("revision_code/release/SOURCE_MANIFEST.json", json.dumps(manifest, indent=2) + "\n")
    sha = hashlib.sha256(archive.read_bytes()).hexdigest()
    (dist / f"{name}.zip.sha256").write_bytes(f"{sha}  {archive.name}\n".encode())  # LF for sha256sum -c
    print(json.dumps({"archive": str(archive), "sha256": sha, "code_digest": digest, "files": len(entries)}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
