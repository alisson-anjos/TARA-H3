"""Check that every Kitchen/Sol binary pinned in tara_h3/h3_temporal_binaries.json is published with the pinned size and SHA-256."""
import json
import os
import sys
import urllib.request
from pathlib import Path

manifest = json.loads((Path(__file__).resolve().parents[1] / 'tara_h3' / 'h3_temporal_binaries.json').read_text())
url = f"https://api.github.com/repos/alisson-anjos/comfy-kitchen/releases/tags/{manifest['release_tag']}"
headers = {'Accept': 'application/vnd.github+json', 'User-Agent': 'TARA-H3-CI'}
if os.environ.get('GITHUB_TOKEN'):
    headers['Authorization'] = 'Bearer ' + os.environ['GITHUB_TOKEN']
assets = {a['name']: a for a in json.load(urllib.request.urlopen(urllib.request.Request(url, headers=headers)))['assets']}
bad = 0
for b in manifest['binaries']:
    a = assets.get(b['file'])
    if a is None:
        print(f"MISSING {b['file']}"); bad += 1; continue
    digest = (a.get('digest') or '').removeprefix('sha256:')
    ok = a['size'] == b['bytes'] and (not digest or digest == b['sha256'])
    print(f"{'ok' if ok else 'MISMATCH'} {b['platform']} sm{b['arch']} {b['file']} ({a['size']} bytes{', sha256 checked' if digest else ''})")
    bad += not ok
sys.exit(1 if bad else 0)
