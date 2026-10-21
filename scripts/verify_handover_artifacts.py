#!/usr/bin/env python3
"""Verify transferred model/geometry files without connecting to hardware."""
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    manifest = json.loads((ROOT / 'docs/artifact-manifest.json').read_text())
    failures = []
    for entry in manifest['files']:
        path = ROOT / entry['path']
        if not path.is_file():
            failures.append(f"Missing: {entry['path']}")
        elif path.stat().st_size != entry['bytes'] or hashlib.sha256(path.read_bytes()).hexdigest() != entry['sha256']:
            failures.append(f"Changed: {entry['path']}")
    if failures:
        print('\n'.join(failures))
        return 1
    print(f"Verified {len(manifest['files'])} artifact and geometry files; no robot connection.")
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
