"""Pin the optional Kitchen/Sol one-pass extension to a comfy-kitchen commit.

Reads BUILD_MANIFEST.json from the per-commit release that alisson-anjos/comfy-kitchen CI publishes
(tag temporal-native-<commit>), rewrites tara_h3/h3_temporal_binaries.json and refreshes the SHA-256
of the Python/LICENSE/NOTICE files the bootstrap downloads from that commit.

Usage: python tools/update_kitchen_manifest.py <commit-sha>
"""
import hashlib
import json
import re
import sys
import urllib.request
from pathlib import Path

REPO = 'alisson-anjos/comfy-kitchen'
ROOT = Path(__file__).resolve().parents[1] / 'tara_h3'


def fetch(url):
    return urllib.request.urlopen(urllib.request.Request(url, headers={'User-Agent': 'TARA-H3-tools'}), timeout=60).read()


commit = sys.argv[1]
tag = f'temporal-native-{commit}'
build = json.loads(fetch(f'https://github.com/{REPO}/releases/download/{tag}/BUILD_MANIFEST.json'))
if build['source_commit'] != commit:
    sys.exit(f"release {tag} was built from {build['source_commit']}")
path = ROOT / 'h3_temporal_binaries.json'
manifest = json.loads(path.read_text())
binaries = []
for b in sorted(build['builds'], key=lambda b: (b['platform'] != 'Linux', b['arch'])):
    ext = 'so' if b['platform'] == 'Linux' else 'dll'
    binaries.append(dict(b, file=f"libck_temporal_sm{b['arch']}.{ext}"))
# Sources for a local build on GPUs without a prebuilt (needs nvcc): pinned file by file.
tree = json.loads(fetch(f'https://api.github.com/repos/{REPO}/git/trees/{commit}?recursive=1'))['tree']
TEMPORAL, SAGE = 'comfy_kitchen/backends/cuda/native_temporal/', 'comfy_kitchen/backends/cuda/sage_attention/'


def needed(path_):
    if path_.startswith(TEMPORAL) and '/vendor/' not in path_:
        return path_.endswith(('.cu', '.cuh', '.h', '.cpp', 'build.py'))
    if path_.startswith(SAGE) and path_.count('/') == SAGE.count('/'):
        return path_.endswith(('.cuh', '.h')) or (path_.endswith('.cu') and path_[len(SAGE):].startswith('sol_attn_'))
    return False


sources = []
for item in tree:
    path_ = item['path']
    if item['type'] != 'blob' or not needed(path_):
        continue
    data = fetch(f'https://raw.githubusercontent.com/{REPO}/{commit}/{path_}')
    sources.append({'path': path_, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)})
manifest.update(release_tag=tag, source_commit=commit, binaries=binaries, sources=sorted(sources, key=lambda x: x['path']))
path.write_text(json.dumps(manifest, indent=2) + '\n')

boot = ROOT / 'minimax_h3_temporal_bootstrap.py'
text = boot.read_text()
for name, rel in (('temporal_attention.py', 'comfy_kitchen/temporal_attention.py'), ('LICENSE', 'LICENSE'), ('NOTICE', 'NOTICE')):
    digest = hashlib.sha256(fetch(f'https://raw.githubusercontent.com/{REPO}/{commit}/{rel}')).hexdigest()
    text, n = re.subn(rf"(\('{re.escape(name)}', RAW\+'/{re.escape(rel)}', ')[0-9a-f]{{64}}(')", rf'\g<1>{digest}\g<2>', text)
    if n != 1:
        sys.exit(f'could not update the pinned hash of {name}')
boot.write_text(text)
print(f'pinned {tag}: {len(binaries)} binaries, {len(sources)} source files')
